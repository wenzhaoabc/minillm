#!/usr/bin/env bash
set -euo pipefail

# LoRA 示例适合快速验证领域适配，不需要改动整套底座权重。
# 这里默认从 full SFT 权重继续训练，也可以把 script_config 里的 load_dir 改成 DPO 产物。
cd "$(dirname "$0")/../.."

if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

MODEL_CONFIG="examples/configs/common/model_config.small.json"
TRAIN_CONFIG="examples/configs/lora/train_config.json"
SCRIPT_CONFIG="examples/configs/lora/script_config.json"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  minillm/trainer/train_lora.py \
  --model_config "$MODEL_CONFIG" \
  --train_config "$TRAIN_CONFIG" \
  --script_config "$SCRIPT_CONFIG" \
  --lora_name lora_identity \
  --lora_rank 8