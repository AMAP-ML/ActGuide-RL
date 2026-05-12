# Action-guidance (ActGuide) RL preprocessing: unguided student prompt with
# action trajectory stored in extra_info for teacher distillation at training
# time.

python preprocess_deepresearch_actguide.py \
  --input_jsonl ${DATA_DIR:-/path/to/data/deepsearch}/raw/ASearcher-DeepResearch-sample1k.jsonl \
  --output_path ${DATA_DIR:-/path/to/data/deepsearch}/GuidedRL/ASearcher-DeepResearch-sample1k-actguide.parquet \
  --data_source DeepResearch \
  --split train
