# make sure your current working directory is the root of the project
set -x

ulimit -n 65535

# sudo mount -o size=400480M -o nr_inodes=4000000 -o noatime,nodiratime -o remount /dev/shm # 增加shm，否则容易出现OOM
export RAY_PLASMA_STORE_MEMORY=$((256*1024*1024*1024))

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export SWANLAB_API_KEY="${SWANLAB_API_KEY:?Set SWANLAB_API_KEY}"
export SWANLAB_MODE="cloud"

export WG_BACKEND="ray"
export RAY_gcs_rpc_server_reconnect_timeout_s=100
export RAY_GCS_RPC_TIMEOUT_S=60
export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1
export GRPC_SERVER_PORT=50051
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_BLOCKING_WAIT=1  # NCCL 允许阻塞等待，确保操作完成
export NCCL_TIMEOUT=600000  # 将超时时间增加到 10 分钟
export HYDRA_FULL_ERROR=1
export RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S=3600

PROJECT_NAME="DeepResearch"

export PYTHONHOME="${CONDA_PREFIX:-/path/to/conda/env}"
export PATH="${CONDA_PREFIX:-/path/to/conda/env}/bin:$PATH"
echo "trainer use python:"
which python

PROJECT_DIR="${PROJECT_DIR:-/path/to/ActGuide-RL}"
cd $PROJECT_DIR

CONFIG_PATH="$PROJECT_DIR/searchagent_scripts/config"
TOOL_CONFIG="$CONFIG_PATH/deepresearch_tool_config.yaml"

n_gpus_per_node=8


export BASE_MODEL="${MODEL_DIR:-/path/to/models}/Qwen3-4B-Instruct-2507"
export EXPERIMENT_NAME=deepresearch-1k-qwen3-4b-actguide


TRAIN_FILES_LIST=(
    "${DATA_DIR:-/path/to/data/deepsearch}/ASearcher-DeepResearch-sample1k-actguide.parquet"
)
TRAIN_FILES="["
for ((i = 0; i < ${#TRAIN_FILES_LIST[@]}; i++)); do
    TRAIN_FILES+="\"${TRAIN_FILES_LIST[i]}\""
    if (( i < ${#TRAIN_FILES_LIST[@]} - 1 )); then
        TRAIN_FILES+=","
    fi
done
TRAIN_FILES+="]"
echo "TRAIN_FILES: ${TRAIN_FILES}"
VALID_FILES_LIST=(
    "${DATA_DIR:-/path/to/data/deepsearch}/DeepSearch/gaia_lv1.parquet"
    "${DATA_DIR:-/path/to/data/deepsearch}/DeepSearch/gaia_lv2.parquet"
    "${DATA_DIR:-/path/to/data/deepsearch}/DeepSearch/gaia_lv3.parquet"
    "${DATA_DIR:-/path/to/data/deepsearch}/DeepSearch/webwalker_easy.parquet"
    "${DATA_DIR:-/path/to/data/deepsearch}/DeepSearch/webwalker_medium.parquet"
    "${DATA_DIR:-/path/to/data/deepsearch}/DeepSearch/webwalker_hard.parquet"
    "${DATA_DIR:-/path/to/data/deepsearch}/DeepSearch/xbench.parquet"
    "${DATA_DIR:-/path/to/data/deepsearch}/DeepSearch/browsecomp_zh.parquet"
)
VALID_FILES="["
for ((i = 0; i < ${#VALID_FILES_LIST[@]}; i++)); do
    VALID_FILES+="\"${VALID_FILES_LIST[i]}\""
    if (( i < ${#VALID_FILES_LIST[@]} - 1 )); then
        VALID_FILES+=","
    fi
done
VALID_FILES+="]"
echo "VALID_FILES: ${VALID_FILES}"

ulimit -n 65535

ip_address=$MASTER_ADDR

ray start --head --node-ip-address=$MASTER_ADDR --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-gpus=$n_gpus_per_node

echo "ray status"
ray status --address="$MASTER_ADDR:6379"
sleep 20

ray job submit --address=$MASTER_ADDR:6379 \
    --runtime-env=verl/trainer/runtime_env.yaml \
    -- \
    python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='multiturn_grpo_actguide' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=40960 \
    data.max_response_length=40960 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.rollout.max_model_len=61440 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=30 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=30 \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=8000 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=left \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.ray_wait_register_center_timeout=1600 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.logger="['console', 'swanlab']" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.test_freq=32 \
    trainer.validation_data_dir=${PROJECT_DIR}/verl_dump/$EXPERIMENT_NAME \
    trainer.default_local_dir=${PROJECT_DIR}/verl_checkpoints/$EXPERIMENT_NAME \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VALID_FILES} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.total_epochs=1 \
    2>&1 | tee logs/$EXPERIMENT_NAME.log

