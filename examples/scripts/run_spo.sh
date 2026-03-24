#!/usr/bin/env bash
set -euo pipefail

# SPO 的 baseline 机制和 PPO、GRPO 不同，但同样建议从 DPO 产物开始。
# 这个示例保留较小 batch 和较短上下文，优先保证单机可调试性。
cd "$(dirname "$0")/../.."

if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

MODEL_CONFIG="examples/configs/common/model_config.small.json"
TRAIN_CONFIG="examples/configs/spo/train_config.json"
SCRIPT_CONFIG="examples/configs/spo/script_config.json"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-models/internlm2-1_8b-reward}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  minillm/trainer/train_spo.py \
  --model_config "$MODEL_CONFIG" \
  --train_config "$TRAIN_CONFIG" \
  --script_config "$SCRIPT_CONFIG" \
  --reward_model_path "$REWARD_MODEL_PATH" \
  --reasoning false \
  --max_gen_len 512 \
  --beta 0.02