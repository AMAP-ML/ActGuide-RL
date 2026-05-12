#!/bin/bash
# Probe the reward gateway. Override the host via REWARD_GATEWAY, e.g.
#   REWARD_GATEWAY=http://your-host:8000 ./test_proxy.sh
BASE_URL="${REWARD_GATEWAY:-http://127.0.0.1:8000}"
N_REWARD_BACKENDS="${REWARD_BACKENDS:-8}"

for i in $(seq 1 "${N_REWARD_BACKENDS}"); do
  curl -s "${BASE_URL%/}/reward${i}/v1/models" | head -100
done
