#!/usr/bin/env bash
set -euo pipefail

# PPO 对显存和 reward model 依赖更强。
# 这里显式关闭 reasoning，避免依赖尚未接入统一配置链路的 reason 模型。
cd "$(dirname "$0")/../.."

if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

MODEL_CONFIG="examples/configs/common/model_config.small.json"
TRAIN_CONFIG="examples/configs/ppo/train_config.json"
SCRIPT_CONFIG="examples/configs/ppo/script_config.json"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-models/internlm2-1_8b-reward}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  minillm/trainer/train_ppo.py \
  --model_config "$MODEL_CONFIG" \
  --train_config "$TRAIN_CONFIG" \
  --script_config "$SCRIPT_CONFIG" \
  --reward_model_path "$REWARD_MODEL_PATH" \
  --reasoning false \
  --max_gen_len 512 \
  --clip_epsilon 0.1 \
  --vf_coef 0.5 \
  --kl_coef 0.02 \
  --update_old_actor_freq 4