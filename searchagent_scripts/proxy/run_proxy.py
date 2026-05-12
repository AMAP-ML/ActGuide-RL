#!/usr/bin/env python3
"""
高并发流式反向代理：在 8000 上统一暴露多个后端，供外网/训练机访问。

核心特性:
  - 多 worker 进程（默认 8），充分利用多核
  - 流式转发（streaming），不缓冲完整响应，首字节即转发
  - httpx 异步连接池（500 并发连接），不阻塞事件循环

路径映射:
  /tools/          -> http://127.0.0.1:8010/  (DeepResearch search/visit)
  /reward1/ ~ /reward8/ -> http://127.0.0.1:7001/ ~ 7008  (Reward vLLM 1-8)

环境变量:
  PROXY_PORT:          监听端口，默认 8000
  PROXY_WORKERS:       worker 进程数，默认 8
  REWARD_NUM:          reward 后端数量，默认 8
  REWARD_BASE_PORT:    reward 起始端口，默认 7001
  TOOLS_BACKEND:       工具服务地址，默认 http://127.0.0.1:8010
  REWARD{N}_BACKEND:   可单独覆盖 rewardN 地址，如 REWARD3_BACKEND=http://host:port

依赖: pip install fastapi uvicorn httpx
运行: python run_proxy.py
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional, Tuple

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from starlette.requests import ClientDisconnect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("proxy")

PROXY_PORT = int(os.environ.get("PROXY_PORT", "8000"))
PROXY_WORKERS = int(os.environ.get("PROXY_WORKERS", "8"))
TOOLS_BACKEND = os.environ.get("TOOLS_BACKEND", "http://127.0.0.1:8010").rstrip("/")

REWARD_NUM = int(os.environ.get("REWARD_NUM", "8"))
REWARD_BASE_PORT = int(os.environ.get("REWARD_BASE_PORT", "7011"))

# /rewardN/ -> http://127.0.0.1:{REWARD_BASE_PORT + N - 1}
# 可通过 REWARD{N}_BACKEND 环境变量单独覆盖
REWARD_BACKENDS: dict[str, str] = {}
for _i in range(1, REWARD_NUM + 1):
    _default = f"http://127.0.0.1:{REWARD_BASE_PORT + _i - 1}"
    REWARD_BACKENDS[f"/reward{_i}/"] = os.environ.get(
        f"REWARD{_i}_BACKEND", _default
    ).rstrip("/")

FORWARD_REQ_HEADERS = {
    "content-type", "accept", "authorization",
    "openai-api-key", "openai-organization",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    limits = httpx.Limits(
        max_connections=500,
        max_keepalive_connections=100,
        keepalive_expiry=30,
    )
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=10.0),
        limits=limits,
        follow_redirects=True,
    )
    reward_summary = "  ".join(f"{k}->{v}" for k, v in REWARD_BACKENDS.items())
    logger.info(
        "Proxy worker started (pid=%s). tools=%s  rewards: %s",
        os.getpid(), TOOLS_BACKEND, reward_summary,
    )
    yield
    await app.state.client.aclose()


app = FastAPI(title="Verl Local Gateway", version="1.0", lifespan=lifespan)


def _backend_for_path(path: str) -> Optional[Tuple[str, str]]:
    """返回 (backend_base_url, stripped_path)."""
    if path.startswith("/tools/"):
        return TOOLS_BACKEND, path[len("/tools"):] or "/"
    for prefix, backend in REWARD_BACKENDS.items():
        if path.startswith(prefix):
            return backend, path[len(prefix) - 1:] or "/"
    return None


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def proxy(path: str, request: Request):
    full_path = "/" + path if path else "/"
    match = _backend_for_path(full_path)
    if not match:
        valid = "/tools/, " + ", ".join(f"/reward{i}/" for i in range(1, REWARD_NUM + 1))
        return Response(
            content=f'{{"error":"use one of: {valid}"}}\n'.encode(),
            status_code=404,
            media_type="application/json",
        )

    backend_base, backend_path = match
    backend_url = backend_base + backend_path
    if request.url.query:
        backend_url += "?" + request.url.query

    headers = {}
    for k, v in request.headers.items():
        kl = k.lower()
        if kl in FORWARD_REQ_HEADERS or kl.startswith("x-"):
            headers[k] = v

    try:
        body = b""
        if request.method in ("POST", "PUT", "PATCH") and request.headers.get("content-length"):
            body = await request.body()
    except ClientDisconnect:
        logger.warning("Client disconnected before body received: %s", backend_url)
        return Response(status_code=499)

    client: httpx.AsyncClient = request.app.state.client
    logger.info("--> %s %s", request.method, backend_url)
    t0 = time.monotonic()

    try:
        backend_req = client.build_request(
            method=request.method,
            url=backend_url,
            headers=headers,
            content=body if body else None,
        )
        backend_resp = await client.send(backend_req, stream=True)
    except httpx.ConnectError as e:
        logger.error("Connect REFUSED %s (%.2fs): %s", backend_url, time.monotonic() - t0, e)
        return Response(
            content=f'{{"error":"backend unreachable: {backend_url}"}}\n'.encode(),
            status_code=502,
            media_type="application/json",
        )
    except httpx.TimeoutException as e:
        logger.error("TIMEOUT %s (%.2fs): %s", backend_url, time.monotonic() - t0, e)
        return Response(
            content=f'{{"error":"backend timeout: {backend_url}"}}\n'.encode(),
            status_code=504,
            media_type="application/json",
        )
    except Exception as e:
        logger.exception("Proxy error %s (%.2fs)", backend_url, time.monotonic() - t0)
        return Response(
            content=f'{{"error":"proxy error: {e}"}}\n'.encode(),
            status_code=502,
            media_type="application/json",
        )

    logger.info(
        "<-- %s %s  status=%s  %.2fs (first byte)",
        request.method, backend_url, backend_resp.status_code, time.monotonic() - t0,
    )

    resp_content_type = backend_resp.headers.get("content-type")

    async def _stream():
        try:
            async for chunk in backend_resp.aiter_bytes(chunk_size=65536):
                yield chunk
        finally:
            await backend_resp.aclose()

    return StreamingResponse(
        content=_stream(),
        status_code=backend_resp.status_code,
        media_type=resp_content_type,
    )


@app.get("/health")
def health():
    return {"status": "ok", "port": PROXY_PORT, "pid": os.getpid()}


def main():
    import uvicorn

    uvicorn.run(
        "run_proxy:app",
        host="0.0.0.0",
        port=PROXY_PORT,
        workers=PROXY_WORKERS,
        access_log=False,
    )


if __name__ == "__main__":
    main()
