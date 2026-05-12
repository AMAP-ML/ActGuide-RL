"""
DeepResearch 双工具 API 服务：仅提供 search 与 visit，基于 Redis 缓存并做优化。
对应 DeepResearch/inference/tool_search.py 与 tool_visit.py 的逻辑。

环境变量:
  SERPER_KEY_ID / SERPER_API_KEY: Serper API 密钥（search + visit 都用）
  REDIS_HOST, REDIS_PORT: Redis 地址（默认 localhost:6379）
  USE_CACHE: 1 启用 Redis 缓存（默认 1）
  ENABLE_CACHE_SYNC: 1 启用周期将 Redis 落盘到 CACHE_FILE
  WARM_CACHE_ON_START: 1 启动时从 WARM_START_FILE 加载缓存到 Redis
  WARM_START_FILE, CACHE_FILE: 暖启/落盘 JSON 路径
  DEEP_RESEARCH_CACHE_PREFIX: Redis 键前缀，默认 "dr:"（仅同步该前缀的 key）
  DEEP_RESEARCH_SEARCH_CACHE_TTL, DEEP_RESEARCH_VISIT_CACHE_TTL: 缓存 TTL（秒）
  DEEP_RESEARCH_L1_CACHE_MAXSIZE: 进程内 LRU 条数，0 关闭（默认 500）
  DEEP_RESEARCH_VISIT_CACHE_COMPRESS: 1 时 visit 原始内容压缩后存 Redis（默认 0）
  DEEP_RESEARCH_VISIT_FAIL_CACHE_TTL: visit 抓取失败时的负缓存 TTL（秒），避免同一 URL 反复打满超时（默认 600）
  DEEP_RESEARCH_RETRY_AFTER_SECONDS: 429 时 Retry-After 头秒数（默认 2）
  PORT: 服务端口，默认 8010
  UVICORN_TIMEOUT_WORKER_HEALTHCHECK: 多 worker 时子进程 pipe 健康检查超时（秒）。同步大文件暖启会长时间占 GIL，
    默认 5s 会导致父进程误判僵死并反复 kill leader；启动脚本会传入更大值，也可用本变量覆盖。

启动: python api_server_deepresearch_redis.py  或  uvicorn api_server_deepresearch_redis:app --host 0.0.0.0 --port 8010
"""

import os
import time
import pathlib
import argparse
import uvicorn
import threading
import atexit
import hashlib
import zlib
import base64
import asyncio
from collections import OrderedDict
from typing import List, Optional, Dict, Any
import json
import httpx
from urllib.parse import urlencode, urlparse, urlunparse
from filelock import FileLock, Timeout

try:
    import redis
except ImportError:
    print("Redis is not installed. Please run: pip install redis")
    exit(1)

try:
    import tiktoken
except ImportError:
    tiktoken = None

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# --- 并发控制 ---
# 内部并发默认值（与 uvicorn --workers=4 配套，避免对外部 Serper 造成过大并发与 429）
# 估算总并发上限（粗略）：workers(4) * visit(10) = 40，workers(4) * search(20) = 80
SEARCH_SEM = threading.Semaphore(int(os.environ.get("DEEP_RESEARCH_SEARCH_CONCURRENCY", "20")))
VISIT_SEM = threading.Semaphore(int(os.environ.get("DEEP_RESEARCH_VISIT_CONCURRENCY", "10")))

LOCK_PATH = "/tmp/tool_cache_deepresearch_sync.leader.lock"
SYNC_LEADER_LOCK = FileLock(LOCK_PATH)


def acquire_leader_lock():
    try:
        SYNC_LEADER_LOCK.acquire(timeout=0)
        return True
    except Exception:
        return False


# --- 配置 ---
SERPER_KEY = os.environ.get("SERPER_KEY_ID") or os.environ.get("SERPER_API_KEY", "")
SERPER_SCRAPE_URL = os.environ.get("SERPER_SCRAPE_URL", "https://scrape.serper.dev")
VISIT_SERVER_TIMEOUT = int(os.environ.get("VISIT_SERVER_TIMEOUT", "50"))
WEBCONTENT_MAXLENGTH = int(os.environ.get("WEBCONTENT_MAXLENGTH", "20000"))

# Redis 键前缀，便于只同步/加载本服务缓存，避免与其他服务冲突
CACHE_KEY_PREFIX = os.environ.get("DEEP_RESEARCH_CACHE_PREFIX", "dr:")
SEARCH_CACHE_TTL = int(os.environ.get("DEEP_RESEARCH_SEARCH_CACHE_TTL", 7 * 24 * 3600))   # 7 天
VISIT_CACHE_TTL = int(os.environ.get("DEEP_RESEARCH_VISIT_CACHE_TTL", 24 * 3600))         # 1 天
# visit 失败负缓存（仅标记，命中则快速返回失败文案，不再请求 Serper）
VISIT_FAIL_CACHE_TTL = int(os.environ.get("DEEP_RESEARCH_VISIT_FAIL_CACHE_TTL", 600))
# 进程内 L1 缓存条数，0 表示关闭
L1_CACHE_MAXSIZE = int(os.environ.get("DEEP_RESEARCH_L1_CACHE_MAXSIZE", "500"))
# visit 原始内容是否压缩后存 Redis（大页面可省内存与带宽）
VISIT_CACHE_COMPRESS = os.environ.get("DEEP_RESEARCH_VISIT_CACHE_COMPRESS", "0") == "1"
# 429 时建议客户端等待秒数
RETRY_AFTER_SECONDS = int(os.environ.get("DEEP_RESEARCH_RETRY_AFTER_SECONDS", "2"))

# --- 请求级超时控制（避免长尾阻塞 worker）---
# 每个 API 请求的“总截止时间”。到点后立刻中断重试并返回 504。
# 为了和最初版本的 timeout 语义一致，这里把 /search 与 /visit 拆开：
# - /search：单次 Serper search timeout 默认 30s
# - /visit：单次 Serper scrape timeout 默认 50s
SEARCH_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("DEEP_RESEARCH_SEARCH_REQUEST_TIMEOUT_SECONDS", "30"))
VISIT_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("DEEP_RESEARCH_VISIT_REQUEST_TIMEOUT_SECONDS", "50"))
# 向后兼容：如果你只配置了旧变量 DEEP_RESEARCH_REQUEST_TIMEOUT_SECONDS，就用它覆盖两类默认值
LEGACY_REQUEST_TIMEOUT_SECONDS = os.environ.get("DEEP_RESEARCH_REQUEST_TIMEOUT_SECONDS", "").strip()
if LEGACY_REQUEST_TIMEOUT_SECONDS:
    try:
        legacy = float(LEGACY_REQUEST_TIMEOUT_SECONDS)
        SEARCH_REQUEST_TIMEOUT_SECONDS = legacy
        VISIT_REQUEST_TIMEOUT_SECONDS = legacy
    except Exception:
        pass
# 距离 deadline 小于该值时直接判定超时，避免 requests/反序列化等收尾再把时间拖过去
DEADLINE_MIN_REMAINING_SECONDS = float(os.environ.get("DEEP_RESEARCH_DEADLINE_MIN_REMAINING_SECONDS", "0.5"))
# 重试时每次失败的 sleep（deadline 很近时会自动缩短）
RETRY_SLEEP_SECONDS = float(os.environ.get("DEEP_RESEARCH_RETRY_SLEEP_SECONDS", "0.5"))

# visit/search 重试次数（仍保留原有默认语义，但会被 REQUEST_TIMEOUT_SECONDS 的 deadline 截断）
# 重试次数（deadline 到点会被截断；这里默认收敛重试次数以降低外部请求风暴）
SEARCH_MAX_ATTEMPTS = int(os.environ.get("DEEP_RESEARCH_SEARCH_MAX_ATTEMPTS", "3"))
VISIT_OUTER_ATTEMPTS = int(os.environ.get("DEEP_RESEARCH_VISIT_OUTER_ATTEMPTS", "4"))
VISIT_INNER_ATTEMPTS = int(os.environ.get("DEEP_RESEARCH_VISIT_INNER_ATTEMPTS", "2"))

INFLIGHT = 0
INFLIGHT_LOCK = threading.Lock()
START_TS = time.time()


class DeepResearchRequestTimeout(Exception):
    """Raised when a single API request reaches its global deadline."""


def _check_deadline(deadline_ts: Optional[float]) -> None:
    if deadline_ts is None:
        return
    # time.time() 在短时间内的误差不影响语义，这里偏向“宁可早点返回”
    if time.time() >= (deadline_ts - DEADLINE_MIN_REMAINING_SECONDS):
        raise DeepResearchRequestTimeout()


def _effective_http_timeout(timeout_s: int, deadline_ts: Optional[float]) -> float:
    """Compute a per-attempt requests timeout that won't overshoot the request deadline."""
    if deadline_ts is None:
        return float(timeout_s)
    remaining = deadline_ts - time.time()
    # requests.timeout 必须是正数；给一个最小值避免立即抛异常死循环
    return float(max(0.2, min(timeout_s, remaining)))


async def _sleep_with_deadline(seconds: float, deadline_ts: Optional[float]) -> None:
    if seconds <= 0:
        return
    if deadline_ts is None:
        await asyncio.sleep(seconds)
        return
    remaining = deadline_ts - time.time()
    # 剩余时间不足以再 sleep，直接抛 timeout 让外层立即返回
    if remaining <= DEADLINE_MIN_REMAINING_SECONDS:
        raise DeepResearchRequestTimeout()
    await asyncio.sleep(min(seconds, max(0.0, remaining - DEADLINE_MIN_REMAINING_SECONDS)))


def _normalize_query_for_key(q: str) -> str:
    """归一化查询字符串，用于缓存 key；过长则用 hash。"""
    q = (q or "").strip()
    if len(q) > 500:
        return hashlib.sha256(q.encode("utf-8")).hexdigest()
    return q


def _visit_raw_is_failure(raw: Optional[str]) -> bool:
    if not raw:
        return True
    if raw.startswith("[visit] Failed to read page.") or raw == "[visit] Empty content.":
        return True
    if raw.startswith("[document_parser]"):
        return True
    return False


def _normalize_url_for_key(url: str) -> str:
    """归一化 URL 用于缓存 key（去 fragment、统一 scheme），提高命中率。"""
    url = (url or "").strip()
    if not url:
        return ""
    try:
        p = urlparse(url)
        if not p.scheme:
            url = "https://" + url
            p = urlparse(url)
        # 去掉 fragment，有时同一页带 #section 会重复抓
        netloc = (p.netloc or "").lower()
        path = (p.path or "/").rstrip("/") or "/"
        query = p.query
        normalized = urlunparse((p.scheme.lower(), netloc, path, "", query, ""))
        if len(normalized) > 1000:
            return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return normalized
    except Exception:
        if len(url) > 1000:
            return hashlib.sha256(url.encode("utf-8")).hexdigest()
        return url


_tiktoken_encoding = None

def _get_tiktoken_encoding():
    global _tiktoken_encoding
    if _tiktoken_encoding is None and tiktoken is not None:
        try:
            _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            pass
    return _tiktoken_encoding

def truncate_to_tokens(text: str, max_tokens: int = 20000) -> str:
    if not text:
        return text
    enc = _get_tiktoken_encoding()
    if enc is None:
        return text[:max_tokens * 4] if len(text) > max_tokens * 4 else text
    try:
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])
    except Exception:
        return text[:max_tokens * 4] if len(text) > max_tokens * 4 else text


# --- L1 进程内 LRU 缓存（减少 Redis 往返）---
class L1LRUCache:
    """线程安全的 LRU，仅缓存 value 为 str，用于热点 key。"""

    def __init__(self, maxsize: int):
        self.maxsize = max(maxsize, 0)
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        if self.maxsize == 0:
            return None
        with self._lock:
            val = self._cache.pop(key, None)
            if val is not None:
                self._cache[key] = val
            return val

    def set(self, key: str, value: str) -> None:
        if self.maxsize == 0:
            return
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = value


# --- Redis 持久化与缓存管理（仅处理本服务前缀的 key）---
class RedisCacheManager:
    """
    仅同步/加载指定 key 前缀的 Redis 键，避免与其他服务混用；
    支持周期性后台落盘与退出时落盘。
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        warm_start_file: str,
        cache_file: str,
        key_prefix: str = CACHE_KEY_PREFIX,
        sync_interval_seconds: float = 3600.0,
    ):
        if not redis_client:
            raise ValueError("A valid Redis client must be provided.")
        self.redis_client = redis_client
        self.warm_start_file = pathlib.Path(warm_start_file)
        self.cache_file = pathlib.Path(cache_file)
        self.key_prefix = key_prefix
        self.sync_interval = sync_interval_seconds
        self._stop_event = threading.Event()
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.sync_file_lock = FileLock(str(self.cache_file) + ".sync.lock")
        self._sync_thread = threading.Thread(target=self._periodic_sync, daemon=True)
        self._sync_thread.start()
        atexit.register(self.stop_and_sync)
        print(f"RedisCacheManager initialized (prefix={self.key_prefix}). Sync every {self.sync_interval}s -> '{self.cache_file}'.")

    def _periodic_sync(self):
        while not self._stop_event.wait(self.sync_interval):
            print("Performing periodic Redis to JSON sync (DeepResearch)...")
            self.sync_to_json()

    def sync_to_json(self, batch_size: int = 5000, max_keys: Optional[int] = None):
        try:
            self.sync_file_lock.acquire(timeout=0)
        except Timeout:
            print("[sync] already running, skip", flush=True)
            return
        try:
            print("Starting sync from Redis to JSON (prefix=%s)..." % self.key_prefix, flush=True)
            start_time = time.time()
            temp_file = self.cache_file.with_suffix(".tmp")
            n = 0
            match = self.key_prefix + "*"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write("{")
                first = True
                batch = []
                for key in self.redis_client.scan_iter(match=match, count=1000):
                    batch.append(key)
                    if len(batch) >= batch_size:
                        vals = self.redis_client.mget(batch)
                        for k, v in zip(batch, vals):
                            if v is None:
                                continue
                            if not first:
                                f.write(",")
                            f.write(json.dumps(k, ensure_ascii=False))
                            f.write(":")
                            f.write(json.dumps(v, ensure_ascii=False))
                            first = False
                            n += 1
                            if max_keys and n >= max_keys:
                                break
                        batch.clear()
                    if max_keys and n >= max_keys:
                        break
                if batch:
                    vals = self.redis_client.mget(batch)
                    for k, v in zip(batch, vals):
                        if v is None:
                            continue
                        if not first:
                            f.write(",")
                        f.write(json.dumps(k, ensure_ascii=False))
                        f.write(":")
                        f.write(json.dumps(v, ensure_ascii=False))
                        first = False
                        n += 1
                f.write("}")
            temp_file.replace(self.cache_file)
            print(f"Synced {n} entries to '{self.cache_file}' in {time.time() - start_time:.2f}s", flush=True)
        except Exception as e:
            print(f"Error during sync_to_json: {e}", flush=True)
        finally:
            try:
                self.sync_file_lock.release()
            except Exception:
                pass

    def load_from_json(self):
        if not self.warm_start_file.exists():
            print(f"Warm start file '{self.warm_start_file}' not found. Skipping.")
            return
        print(f"Warming up cache from '{self.warm_start_file}' (prefix={self.key_prefix})...")
        start_time = time.time()
        try:
            with open(self.warm_start_file, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Cache JSON top-level must be a dict.")
            time.sleep(0)  # 大 json.load 之后让出 GIL，避免 uvicorn worker 健康检查 ping 超时
            pipe = self.redis_client.pipeline()
            n = 0
            default_ttl = max(SEARCH_CACHE_TTL, VISIT_CACHE_TTL)
            for k, v in data.items():
                if not isinstance(k, str) or not k.startswith(self.key_prefix):
                    continue
                pipe.set(k, v, ex=default_ttl)
                n += 1
                if n % 5000 == 0:
                    pipe.execute()
                    pipe = self.redis_client.pipeline()
                    time.sleep(0)
            pipe.execute()
            print(f"Loaded {n} keys into Redis in {time.time() - start_time:.2f}s")
        except Exception as e:
            print(f"Error during load_from_json: {e}", flush=True)

    def stop_and_sync(self):
        print("Stopping Redis cache manager and performing final sync...")
        self._stop_event.set()
        try:
            self.sync_to_json()
        finally:
            try:
                self._sync_thread.join(timeout=30)
            except Exception:
                pass


# --- Search 逻辑（对应 DeepResearch tool_search.py）---
def _contains_chinese(text: str) -> bool:
    return any("\u4E00" <= c <= "\u9FFF" for c in text)


# 复用 HTTP 连接（在 on_startup 中初始化）
_http_client: Optional[httpx.AsyncClient] = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=200),
        )
    return _http_client

async def _google_search_serper(query: str, timeout: int = 30, deadline_ts: Optional[float] = None) -> str:
    if not SERPER_KEY:
        return "Google search failed: SERPER_KEY_ID or SERPER_API_KEY is not set."
    if _contains_chinese(query):
        payload = {"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"}
    else:
        payload = {"q": query, "location": "United States", "gl": "us", "hl": "en"}
    headers = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}
    client = _get_http_client()
    for attempt in range(SEARCH_MAX_ATTEMPTS):
        _check_deadline(deadline_ts)
        effective_timeout = _effective_http_timeout(timeout, deadline_ts)
        try:
            r = await client.post(
                "https://google.serper.dev/search",
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(effective_timeout),
            )
            r.raise_for_status()
            data = r.json()
            if "organic" not in data:
                return f"No results found for query: '{query}'. Use a less specific query."
            snippets = []
            for i, page in enumerate(data["organic"], 1):
                date_pub = "\nDate published: " + page.get("date", "") if page.get("date") else ""
                src = "\nSource: " + page.get("source", "") if page.get("source") else ""
                snip = "\n" + page.get("snippet", "") if page.get("snippet") else ""
                line = f"{i}. [{page.get('title', '')}]({page.get('link', '')}){date_pub}{src}\n{snip}".replace(
                    "Your browser can't play this video.", ""
                )
                snippets.append(line)
            return (
                f"A Google search for '{query}' found {len(snippets)} results:\n\n## Web Results\n"
                + "\n\n".join(snippets)
            )
        except Exception as e:
            if attempt == (SEARCH_MAX_ATTEMPTS - 1):
                return f"Google search failed for '{query}': {e}. Please try again later."
            await _sleep_with_deadline(RETRY_SLEEP_SECONDS, deadline_ts)
    return f"No results found for '{query}'."


# --- Visit 逻辑（对应 DeepResearch tool_visit.py）---
async def _serper_readpage(url: str, timeout: int = 50, deadline_ts: Optional[float] = None) -> str:
    if not SERPER_KEY:
        return "[visit] Failed to read page: SERPER_API_KEY is not set."
    scrape_url = f"{SERPER_SCRAPE_URL}?{urlencode({'url': url, 'apiKey': SERPER_KEY})}"
    client = _get_http_client()
    for attempt in range(VISIT_INNER_ATTEMPTS):
        _check_deadline(deadline_ts)
        effective_timeout = _effective_http_timeout(timeout, deadline_ts)
        try:
            r = await client.get(scrape_url, timeout=httpx.Timeout(effective_timeout))
            if r.status_code == 200:
                return r.text
            raise ValueError("serper scrape error")
        except Exception as e:
            if attempt == (VISIT_INNER_ATTEMPTS - 1):
                return "[visit] Failed to read page."
            await _sleep_with_deadline(RETRY_SLEEP_SECONDS, deadline_ts)
    return "[visit] Failed to read page."


async def _html_readpage_serper(url: str, deadline_ts: Optional[float] = None) -> str:
    for _ in range(VISIT_OUTER_ATTEMPTS):
        _check_deadline(deadline_ts)
        content = await _serper_readpage(url, timeout=VISIT_SERVER_TIMEOUT, deadline_ts=deadline_ts)
        if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content.":
            if not content.startswith("[document_parser]"):
                return content
    return "[visit] Failed to read page."


async def _readpage_serper(url: str, goal: str) -> str:
    content = await _html_readpage_serper(url, deadline_ts=None)
    if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
        content = truncate_to_tokens(content, max_tokens=WEBCONTENT_MAXLENGTH // 4 or 5000)
        return (
            f"The useful information in {url} for user goal {goal} as follows:\n\nContent:\n{content}\n\n"
        )
    return (
        f"The useful information in {url} for user goal {goal} as follows:\n\n"
        "Evidence in page: The provided webpage content could not be accessed. Please check the URL or file format.\n\n"
        "Summary: The webpage content could not be processed, and therefore, no information is available.\n\n"
    )


# 存 Redis 时 visit 大内容可选压缩，读时据此解压
_COMPRESS_PREFIX = "Z:"

def _compress_visit_value(value: str) -> str:
    return _COMPRESS_PREFIX + base64.b64encode(zlib.compress(value.encode("utf-8"), level=6)).decode("ascii")

def _decompress_visit_value(value: str) -> str:
    if not value.startswith(_COMPRESS_PREFIX):
        return value
    try:
        return zlib.decompress(base64.b64decode(value[len(_COMPRESS_PREFIX):])).decode("utf-8")
    except Exception:
        return value


# --- 带缓存的执行器 ---
class DeepResearchEngine:
    def __init__(self, redis_client: redis.Redis, use_cache: bool = True):
        self.redis = redis_client
        self.use_cache = use_cache
        self.l1 = L1LRUCache(L1_CACHE_MAXSIZE)

    def _cache_get(self, key: str, decompress_visit: bool = False) -> Optional[str]:
        if self.l1.maxsize > 0:
            val = self.l1.get(key)
            if val is not None:
                return val
        if not self.use_cache:
            return None
        try:
            val = self.redis.get(key)
            if val is None:
                return None
            if decompress_visit and val.startswith(_COMPRESS_PREFIX):
                val = _decompress_visit_value(val)
            if self.l1.maxsize > 0:
                self.l1.set(key, val)
            return val
        except Exception as e:
            print(f"Cache get failed: {e}")
            return None

    def _cache_set(self, key: str, value: str, ttl: int, compress_visit: bool = False) -> None:
        if self.l1.maxsize > 0:
            self.l1.set(key, value)
        if not self.use_cache:
            return
        try:
            if compress_visit and VISIT_CACHE_COMPRESS and len(value) > 2048:
                value = _compress_visit_value(value)
            self.redis.set(key, value, ex=ttl)
        except Exception as e:
            print(f"Cache set failed: {e}")

    async def search(self, query: str, timeout: int = 30, deadline_ts: Optional[float] = None) -> str:
        if not (query or "").strip():
            return "Empty query provided."
        key = CACHE_KEY_PREFIX + "search:" + _normalize_query_for_key(query)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        _check_deadline(deadline_ts)
        result = await _google_search_serper(query, timeout=timeout, deadline_ts=deadline_ts)
        self._cache_set(key, result, SEARCH_CACHE_TTL)
        return result

    async def search_batch(self, queries: List[str], timeout: int = 30, deadline_ts: Optional[float] = None) -> str:
        if not queries:
            return "Empty query list."
        keys = [CACHE_KEY_PREFIX + "search:" + _normalize_query_for_key(q) for q in queries]
        cached_list: List[Optional[str]] = [None] * len(queries)
        # L1
        for i in range(len(queries)):
            if self.l1.maxsize > 0:
                v = self.l1.get(keys[i])
                if v is not None:
                    cached_list[i] = v
        # Redis 批量 get
        need_redis_idx = [i for i in range(len(queries)) if cached_list[i] is None]
        if need_redis_idx and self.use_cache:
            try:
                rkeys = [keys[i] for i in need_redis_idx]
                rvals = self.redis.mget(rkeys)
                for j, i in enumerate(need_redis_idx):
                    if rvals[j] is not None:
                        cached_list[i] = rvals[j]
                        if self.l1.maxsize > 0:
                            self.l1.set(keys[i], rvals[j])
            except Exception as e:
                print(f"Batch cache get failed: {e}")
        # 未命中则请求并批量 set
        still_missing = [i for i in range(len(queries)) if cached_list[i] is None]
        if still_missing:
            pipe = self.redis.pipeline() if self.use_cache else None
            for i in still_missing:
                _check_deadline(deadline_ts)
                result = await _google_search_serper(queries[i], timeout=timeout, deadline_ts=deadline_ts)
                cached_list[i] = result
                if pipe is not None:
                    pipe.set(keys[i], result, ex=SEARCH_CACHE_TTL)
            if pipe is not None:
                try:
                    pipe.execute()
                except Exception as e:
                    print(f"Batch cache set failed: {e}")
        return "\n=======\n".join(cached_list[i] or "" for i in range(len(queries)))

    async def visit_url(self, url: str, goal: str, deadline_ts: Optional[float] = None) -> str:
        if not (url or "").strip():
            return "[Visit] Empty url provided."
        norm = _normalize_url_for_key(url)
        raw_key = CACHE_KEY_PREFIX + "visit_raw:" + norm
        fail_key = CACHE_KEY_PREFIX + "visit_fail:" + norm
        # 负缓存：近期已判定失败的 URL，直接返回，避免重复 50s 长尾
        if self.use_cache:
            try:
                if self.redis.get(fail_key):
                    goal = goal or ""
                    return (
                        f"The useful information in {url} for user goal {goal} as follows:\n\n"
                        "Evidence in page: The provided webpage content could not be accessed. Please check the URL or file format.\n\n"
                        "Summary: The webpage content could not be processed, and therefore, no information is available.\n\n"
                    )
            except Exception as e:
                print(f"visit_fail cache get failed: {e}", flush=True)
        raw_content = self._cache_get(raw_key, decompress_visit=True)
        if raw_content is None:
            _check_deadline(deadline_ts)
            raw_content = await _html_readpage_serper(url, deadline_ts=deadline_ts)
            if raw_content and not raw_content.startswith("[visit] Failed to read page.") and raw_content != "[visit] Empty content." and not raw_content.startswith("[document_parser]"):
                self._cache_set(raw_key, raw_content, VISIT_CACHE_TTL, compress_visit=True)
                if self.use_cache:
                    try:
                        self.redis.delete(fail_key)
                    except Exception as e:
                        print(f"visit_fail cache delete after success failed: {e}", flush=True)
            elif _visit_raw_is_failure(raw_content) and self.use_cache:
                try:
                    self.redis.set(fail_key, "1", ex=VISIT_FAIL_CACHE_TTL)
                except Exception as e:
                    print(f"visit_fail cache set failed: {e}", flush=True)
        goal = goal or ""
        if raw_content and not raw_content.startswith("[visit] Failed to read page.") and raw_content != "[visit] Empty content." and not raw_content.startswith("[document_parser]"):
            content = truncate_to_tokens(raw_content, max_tokens=WEBCONTENT_MAXLENGTH // 4 or 5000)
            return f"The useful information in {url} for user goal {goal} as follows:\n\nContent:\n{content}\n\n"
        return (
            f"The useful information in {url} for user goal {goal} as follows:\n\n"
            "Evidence in page: The provided webpage content could not be accessed. Please check the URL or file format.\n\n"
            "Summary: The webpage content could not be processed, and therefore, no information is available.\n\n"
        )

    async def visit(self, url: str, goal: str, deadline_ts: Optional[float] = None) -> str:
        return await self.visit_url(url, goal, deadline_ts=deadline_ts)

    async def visit_batch(self, urls: List[str], goal: str, deadline_ts: Optional[float] = None) -> str:
        if not urls:
            return "[Visit] Empty url list."
        results = []
        for u in urls:
            try:
                _check_deadline(deadline_ts)
                results.append(await self.visit_url(u, goal, deadline_ts=deadline_ts))
            except DeepResearchRequestTimeout:
                # 全局 deadline 超时要向外传播，确保 endpoint 能返回 504
                raise
            except Exception as e:
                results.append(f"Error fetching {u}: {e}")
        return "\n=======\n".join(results)


# --- FastAPI ---
app = FastAPI(title="DeepResearch API (Search + Visit)")

class SearchRequest(BaseModel):
    query: str | List[str]

class VisitRequest(BaseModel):
    url: str | List[str]
    goal: str

engine: Optional[DeepResearchEngine] = None
cache_manager: Optional[RedisCacheManager] = None
redis_client: Optional[redis.Redis] = None


@app.middleware("http")
async def log_requests(request: Request, call_next):
    global INFLIGHT
    with INFLIGHT_LOCK:
        INFLIGHT += 1
        cur = INFLIGHT
    t0 = time.time()
    try:
        return await call_next(request)
    finally:
        with INFLIGHT_LOCK:
            INFLIGHT -= 1
        print(f"[srv] pid={os.getpid()} inflight={cur} {request.method} {request.url.path} cost_ms={(time.time()-t0)*1000:.1f}", flush=True)


def _try_acquire(sem: threading.Semaphore, name: str):
    if not sem.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=f"too many {name} requests",
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        )


@app.post("/search")
async def search_endpoint(req: SearchRequest):
    _try_acquire(SEARCH_SEM, "search")
    deadline_ts = time.time() + SEARCH_REQUEST_TIMEOUT_SECONDS
    try:
        if isinstance(req.query, list):
            return await engine.search_batch(req.query, deadline_ts=deadline_ts)
        return await engine.search(req.query, deadline_ts=deadline_ts)
    except DeepResearchRequestTimeout:
        raise HTTPException(status_code=504, detail="search timeout")
    finally:
        SEARCH_SEM.release()


@app.post("/visit")
async def visit_endpoint(req: VisitRequest):
    _try_acquire(VISIT_SEM, "visit")
    deadline_ts = time.time() + VISIT_REQUEST_TIMEOUT_SECONDS
    try:
        if isinstance(req.url, list):
            return await engine.visit_batch(req.url, req.goal, deadline_ts=deadline_ts)
        return await engine.visit(req.url, req.goal, deadline_ts=deadline_ts)
    except DeepResearchRequestTimeout:
        raise HTTPException(status_code=504, detail="visit timeout")
    finally:
        VISIT_SEM.release()


@app.post("/sync_cache")
def sync_cache_endpoint(background: bool = True):
    if cache_manager is None:
        raise HTTPException(status_code=503, detail="cache_manager disabled or not leader")

    def _run():
        cache_manager.sync_to_json()

    if background:
        threading.Thread(target=_run, daemon=True).start()
        return {"message": "sync started (background)"}
    _run()
    return {"message": "sync finished"}


@app.get("/healthz")
def healthz():
    try:
        if redis_client:
            redis_client.ping()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/diag")
def diag():
    try:
        load = os.getloadavg()
    except Exception:
        load = (None, None, None)
    with INFLIGHT_LOCK:
        inflight = INFLIGHT
    return {
        "pid": os.getpid(),
        "uptime_s": round(time.time() - START_TS, 1),
        "inflight": inflight,
        "threads": threading.active_count(),
        "loadavg": list(load),
    }


def connect_redis(host: str, port: int, retries: int = 20, delay: float = 3) -> redis.Redis:
    for i in range(retries):
        try:
            client = redis.Redis(
                host=host,
                port=port,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            client.ping()
            print(f"Connected to Redis at {host}:{port}")
            return client
        except redis.exceptions.BusyLoadingError as e:
            print(f"Attempt {i+1}/{retries}: Redis loading. Retry in {delay}s: {e}")
            time.sleep(delay)
        except redis.exceptions.ConnectionError as e:
            print(f"Attempt {i+1}/{retries}: Redis connection failed. Retry in {delay}s: {e}")
            time.sleep(delay)
    print(f"FATAL: Could not connect to Redis at {host}:{port}")
    exit(1)


def parse_args():
    args = argparse.Namespace()
    args.redis_host = os.environ.get("REDIS_HOST", "localhost")
    args.redis_port = int(os.environ.get("REDIS_PORT", "6397"))
    args.warm_start_file = os.environ.get("WARM_START_FILE", "deepresearch_cache_warm.json")
    args.cache_file = os.environ.get("CACHE_FILE", "/tmp/deepresearch_cache_redis_dump.json")
    args.cache_sync_interval = float(os.environ.get("CACHE_SYNC_INTERVAL", "3600"))
    args.enable_cache_sync = os.environ.get("ENABLE_CACHE_SYNC", "0") == "1"
    args.warm_cache_on_start = os.environ.get("WARM_CACHE_ON_START", "0") == "1"
    args.use_cache = os.environ.get("USE_CACHE", "1") == "1"
    return args


@app.on_event("startup")
def on_startup():
    global engine, cache_manager, redis_client, _http_client
    args = parse_args()
    redis_client = connect_redis(args.redis_host, args.redis_port)
    warm_start_file = str(pathlib.Path(args.warm_start_file).expanduser())
    cache_file = str(pathlib.Path(args.cache_file).expanduser())
    cache_manager = None
    if args.enable_cache_sync and acquire_leader_lock():
        print(f"[startup] cache sync leader acquired pid={os.getpid()}", flush=True)
        cache_manager = RedisCacheManager(
            redis_client=redis_client,
            warm_start_file=warm_start_file,
            cache_file=cache_file,
            key_prefix=CACHE_KEY_PREFIX,
            sync_interval_seconds=args.cache_sync_interval,
        )
        if args.warm_cache_on_start:
            # 勿在 lifespan 里同步跑大文件 load：会长时间占 GIL，uvicorn 多 worker 父进程默认 5s 内收不到 pipe pong 会 kill 子进程。
            threading.Thread(
                target=cache_manager.load_from_json,
                name="dr-warm-redis",
                daemon=True,
            ).start()
    else:
        print(f"[startup] cache sync disabled or not leader pid={os.getpid()}", flush=True)
    # 预热 HTTP 连接池，避免首请求建连延迟
    _get_http_client()
    engine = DeepResearchEngine(redis_client=redis_client, use_cache=args.use_cache)
    print(f"[startup] engine ready use_cache={args.use_cache} l1_size={L1_CACHE_MAXSIZE}", flush=True)


@app.on_event("shutdown")
async def on_shutdown():
    global cache_manager, _http_client
    try:
        if cache_manager:
            cache_manager.stop_and_sync()
    except Exception as e:
        print(f"[shutdown] cache_manager stop failed: {e}", flush=True)
    try:
        if _http_client is not None:
            await _http_client.aclose()
            _http_client = None
    except Exception:
        pass
    try:
        if SYNC_LEADER_LOCK.is_locked:
            SYNC_LEADER_LOCK.release()
    except Exception:
        pass


if __name__ == "__main__":
    uvicorn.run(
        "api_server_deepresearch_redis:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8010")),
        workers=1,
        access_log=True,
    )
