export PYTHONHOME="${CONDA_PREFIX:-/path/to/conda/env}"
export PATH="${CONDA_PREFIX:-/path/to/conda/env}/bin:$PATH"
echo "vllm use python:"
which python


CUDA_VISIBLE_DEVICES=0 vllm serve /mnt/workspace/common/models/Qwen3-30B-A3B-Instruct-2507 --host 0.0.0.0 --port 7011 --disable-log-requests &
CUDA_VISIBLE_DEVICES=1 vllm serve /mnt/workspace/common/models/Qwen3-30B-A3B-Instruct-2507 --host 0.0.0.0 --port 7012 --disable-log-requests &
CUDA_VISIBLE_DEVICES=2 vllm serve /mnt/workspace/common/models/Qwen3-30B-A3B-Instruct-2507 --host 0.0.0.0 --port 7013 --disable-log-requests &
CUDA_VISIBLE_DEVICES=3 vllm serve /mnt/workspace/common/models/Qwen3-30B-A3B-Instruct-2507 --host 0.0.0.0 --port 7014 --disable-log-requests &
CUDA_VISIBLE_DEVICES=4 vllm serve /mnt/workspace/common/models/Qwen3-30B-A3B-Instruct-2507 --host 0.0.0.0 --port 7015 --disable-log-requests &
CUDA_VISIBLE_DEVICES=5 vllm serve /mnt/workspace/common/models/Qwen3-30B-A3B-Instruct-2507 --host 0.0.0.0 --port 7016 --disable-log-requests &
CUDA_VISIBLE_DEVICES=6 vllm serve /mnt/workspace/common/models/Qwen3-30B-A3B-Instruct-2507 --host 0.0.0.0 --port 7017 --disable-log-requests &
CUDA_VISIBLE_DEVICES=7 vllm serve /mnt/workspace/common/models/Qwen3-30B-A3B-Instruct-2507 --host 0.0.0.0 --port 7018 --disable-log-requests &


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /mnt/workspace/common/models/Qwen3-235B-A22B-Instruct-2507 --host 0.0.0.0 --port 7001 --tensor-parallel-size 8  --disable-log-requests &

cd ${PROJECT_DIR:-/path/to/ActGuide-RL}/searchagent_scripts/proxy
python run_proxy.py


