#!/usr/bin/env bash
set -euo pipefail

# DPO 通常接在 full SFT 之后。
# beta 使用一个保守值 0.1，先保证训练稳定，再按数据质量调整。
cd "$(dirname "$0")/../.."

if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

MODEL_CONFIG="examples/configs/common/model_config.small.json"
TRAIN_CONFIG="examples/configs/dpo/train_config.json"
SCRIPT_CONFIG="examples/configs/dpo/script_config.json"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  minillm/trainer/train_dpo.py \
  --model_config "$MODEL_CONFIG" \
  --train_config "$TRAIN_CONFIG" \
  --script_config "$SCRIPT_CONFIG" \
  --beta 0.1