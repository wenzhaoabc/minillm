#!/usr/bin/env bash
set -euo pipefail

# 全参数 SFT 基于预训练产物继续训练。
# 默认通过 script_config 里的 load_dir 直接读取 out/pretrain-small。
cd "$(dirname "$0")/../.."

if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

MODEL_CONFIG="examples/configs/common/model_config.small.json"
TRAIN_CONFIG="examples/configs/full_sft/train_config.json"
SCRIPT_CONFIG="examples/configs/full_sft/script_config.json"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  minillm/trainer/train_full_sft.py \
  --model_config "$MODEL_CONFIG" \
  --train_config "$TRAIN_CONFIG" \
  --script_config "$SCRIPT_CONFIG"