#!/usr/bin/env bash
set -euo pipefail

# GRPO 更适合一条 prompt 采多个回答再做组内比较。
# 为了让单卡练习更容易跑通，这里把 num_generations 设为 4，而不是默认的 8。
cd "$(dirname "$0")/../.."

if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

MODEL_CONFIG="examples/configs/common/model_config.small.json"
TRAIN_CONFIG="examples/configs/grpo/train_config.json"
SCRIPT_CONFIG="examples/configs/grpo/script_config.json"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-models/internlm2-1_8b-reward}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  minillm/trainer/train_grpo.py \
  --model_config "$MODEL_CONFIG" \
  --train_config "$TRAIN_CONFIG" \
  --script_config "$SCRIPT_CONFIG" \
  --reward_model_path "$REWARD_MODEL_PATH" \
  --reasoning false \
  --max_gen_len 512 \
  --num_generations 4 \
  --beta 0.02