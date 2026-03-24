#!/usr/bin/env bash
set -euo pipefail

# 预训练是整条链路的起点。
# 这里使用共享的小模型结构配置，训练参数和数据路径单独放在 stage 配置里。
cd "$(dirname "$0")/../.."

if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

MODEL_CONFIG="examples/configs/common/model_config.small.json"
TRAIN_CONFIG="examples/configs/pretrain/train_config.json"
SCRIPT_CONFIG="examples/configs/pretrain/script_config.json"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  minillm/trainer/train_pretrain.py \
  --model_config "$MODEL_CONFIG" \
  --train_config "$TRAIN_CONFIG" \
  --script_config "$SCRIPT_CONFIG"