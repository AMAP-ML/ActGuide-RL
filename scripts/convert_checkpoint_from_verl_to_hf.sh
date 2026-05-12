python ./scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --hf_model_path /path/to/models/Qwen3-4B-Instruct-2507 \
    --local_dir ${PROJECT_DIR:-/path/to/ActGuide-RL}/verl_checkpoints_local/deepresearch-1k-qwen25-3b-guidedv6-rescue-adaptive-offpolicy-turn30-n8/global_step_31/actor \
    --target_dir ${PROJECT_DIR:-/path/to/ActGuide-RL}/verl_checkpoints/merge/deepresearch_1k_qwen25_3b_actguide_turn30_n8