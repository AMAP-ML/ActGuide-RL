#!/bin/bash
# DeepResearch 双工具服务 (search + visit) 启动脚本，支持多机多卡：传入 RANK 区分实例
# 用法: ./run_deepresearch_api_server_1.sh <RANK>
# 多机时每台机器一个 RANK，缓存文件与端口按 RANK 区分；可配合 taskset 绑核
# 可选: START_REDIS_IN_SCRIPT=1 时由本脚本在本机启动 Redis，再启动 API（方式3）
set -e

RANK="$1"
if [[ -z "$RANK" ]]; then
  echo "Usage: $0 <RANK>"
  exit 1
fi

cd "$(dirname "$0")"

# Python 环境（可按需改成自己的 conda 路径或注释掉用当前环境）
export PYTHONHOME="${CONDA_PREFIX:-/path/to/conda/env}"
export PATH="${CONDA_PREFIX:-/path/to/conda/env}/bin:$PATH"

echo "DeepResearch API server RANK=$RANK python:"
which python

# ---------- 方式3：可选在脚本内启动 Redis ----------
# 默认 Redis 路径（与 run_api_server_1 一致，可按环境修改）
REDIS_DIR_DEFAULT="${REDIS_DIR:-/path/to/redis}"
REDIS_CONF_DEFAULT="${REDIS_DIR_DEFAULT}/my-redis.conf"
REDIS_BIN_DEFAULT="${REDIS_DIR_DEFAULT}/src/redis-server"
# 设置 START_REDIS_IN_SCRIPT=1 时，由本脚本在本机启动 Redis，再启动 API。
if [[ "${START_REDIS_IN_SCRIPT}" == "1" ]]; then
  REDIS_CLI="${REDIS_CLI:-redis-cli}"
  REDIS_PORT="${REDIS_PORT:-6397}"
  if [[ -n "${REDIS_START_CONF}" ]]; then
    REDIS_CONF="$REDIS_START_CONF"
    REDIS_BIN="${REDIS_BIN:-redis-server}"
  elif [[ -f "${REDIS_CONF_DEFAULT}" ]]; then
    REDIS_CONF="$REDIS_CONF_DEFAULT"
    REDIS_BIN="${REDIS_BIN:-$REDIS_BIN_DEFAULT}"
  else
    REDIS_CONF=""
  fi
  if [[ -n "$REDIS_CONF" ]]; then
    echo "[redis] RANK=$RANK starting Redis with config: $REDIS_CONF"
    "$REDIS_BIN" "$REDIS_CONF" 2>/dev/null || true
  else
    REDIS_BIN="${REDIS_BIN:-redis-server}"
    echo "[redis] RANK=$RANK starting Redis on port $REDIS_PORT (daemon)"
    "$REDIS_BIN" --port "$REDIS_PORT" --bind 127.0.0.1 --daemonize yes 2>/dev/null || true
  fi
  export REDIS_HOST="${REDIS_HOST:-localhost}"
  export REDIS_PORT="$REDIS_PORT"
  echo "[redis] RANK=$RANK waiting for Redis on ${REDIS_HOST}:${REDIS_PORT} ..."
  i=0
  while [[ $i -lt 30 ]]; do
    if "$REDIS_CLI" -h "$REDIS_HOST" -p "$REDIS_PORT" ping 2>/dev/null | grep -q PONG; then
      echo "[redis] RANK=$RANK Redis ready."
      break
    fi
    i=$((i + 1))
    if [[ $i -eq 30 ]]; then
      echo "[redis] RANK=$RANK WARNING: Redis did not respond after 30 attempts; API may fail."
    fi
    sleep 0.5
  done
fi

# 按 RANK 区分的缓存路径（多机多卡时每 rank 一份，避免互相覆盖）
BASE_DIR="${DEEP_RESEARCH_CACHE_DIR:-${PROJECT_DIR:-/path/to/ActGuide-RL}/tool_server/cache}"
warm_start_file="${BASE_DIR}/deepresearch_cache_warm_1.json"
cache_file="${BASE_DIR}/deepresearch_cache_redis_1_${RANK}.json"

# 端口：单机多卡时 8010+RANK 避免冲突；多机时可由外部统一设 PORT=8010
# export PORT="${PORT:-$((8010 + RANK))}"
export PORT="8010"

# Serper API（必须）
export SERPER_API_KEY="${SERPER_API_KEY:?Set SERPER_API_KEY}"

# Redis（未在脚本内启动时需本机或远程已有 Redis）
export REDIS_HOST="${REDIS_HOST:-localhost}"
export REDIS_PORT="${REDIS_PORT:-6397}"

# 缓存与同步
export WARM_START_FILE="$warm_start_file"
export CACHE_FILE="$cache_file"
export WARM_CACHE_ON_START="${WARM_CACHE_ON_START:-1}"
export USE_CACHE="${USE_CACHE:-1}"
export ENABLE_CACHE_SYNC="${ENABLE_CACHE_SYNC:-1}"
export CACHE_SYNC_INTERVAL="${CACHE_SYNC_INTERVAL:-3600}"

# 预留给 tool server 的核数（多机多卡时可按机器调）
SERVER_NCORES="${SERVER_NCORES:-8}"
CPU_ALLOWED_LIST="$(grep Cpus_allowed_list /proc/self/status 2>/dev/null | awk '{print $2}' || true)"
echo "[cpu] RANK=$RANK Cpus_allowed_list=${CPU_ALLOWED_LIST:-N/A}"

SERVER_CPUS=""
if [[ -n "$CPU_ALLOWED_LIST" ]]; then
  if read -r SERVER_CPUS TRAIN_CPUS < <(python3 - <<'PY'
import os, re

allowed = open("/proc/self/status").read()
m = re.search(r"Cpus_allowed_list:\s*(.*)", allowed)
s = (m.group(1) if m else "").strip()
if not s:
    print("0", "0")
    raise SystemExit(0)

def expand(rng: str):
    out = []
    for part in rng.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a,b = part.split("-")
            out.extend(range(int(a), int(b)+1))
        else:
            out.append(int(part))
    return out

def compress(nums):
    nums = sorted(set(nums))
    if not nums: return ""
    ranges=[]
    st=pr=nums[0]
    for x in nums[1:]:
        if x==pr+1:
            pr=x
        else:
            ranges.append((st,pr))
            st=pr=x
    ranges.append((st,pr))
    parts=[]
    for a,b in ranges:
        parts.append(str(a) if a==b else f"{a}-{b}")
    return ",".join(parts)

nums = expand(s)
n_server = int(os.environ.get("SERVER_NCORES","8"))
n_server = max(1, min(n_server, len(nums)-1))

server = nums[:n_server]
train  = nums[n_server:]
print(compress(server), compress(train))
PY
) 2>/dev/null && [[ -n "$SERVER_CPUS" ]]; then
    echo "[cpu] RANK=$RANK server cpus: ${SERVER_CPUS}"
    echo "[cpu] RANK=$RANK train  cpus: ${TRAIN_CPUS}"
  fi
fi
if [[ -n "$SERVER_CPUS" ]]; then
  TASKSET_ARGS="taskset -c $SERVER_CPUS"
else
  TASKSET_ARGS=""
fi

echo "[start] RANK=$RANK PORT=$PORT CACHE_FILE=$cache_file"

# uvicorn 多 worker：父进程对子进程做 pipe 健康检查，默认仅等 5s。大暖启仍可能在后台线程里长时间占 GIL，拉大窗口避免误判杀进程。
UVICORN_TIMEOUT_WORKER_HEALTHCHECK="${UVICORN_TIMEOUT_WORKER_HEALTHCHECK:-86400}"

# 启动 DeepResearch API（--workers 4：本机 4 个进程一起接请求；其中只有 1 个进程负责把 Redis 缓存定期落盘到 CACHE_FILE，由 api 内文件锁决定）
if [[ -n "$TASKSET_ARGS" ]]; then
  $TASKSET_ARGS python -m uvicorn api_server_deepresearch_redis:app --host 0.0.0.0 --port "$PORT" --workers 4 --access-log \
    --timeout-worker-healthcheck "$UVICORN_TIMEOUT_WORKER_HEALTHCHECK"
else
  python -m uvicorn api_server_deepresearch_redis:app --host 0.0.0.0 --port "$PORT" --workers 4 --access-log \
    --timeout-worker-healthcheck "$UVICORN_TIMEOUT_WORKER_HEALTHCHECK"
fi