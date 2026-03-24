#!/usr/bin/env bash
set -euo pipefail

# 从仓库根目录运行，确保脚本里的相对路径稳定。
cd "$(dirname "$0")/../.."

# 如果本地虚拟环境存在，就直接启用，避免依赖系统 Python。
if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

# tokenizer 脚本目前仍是常量驱动版本。
# 使用前请先按需修改 minillm/trainer/train_tokenizer.py 里的 DATA_PATH、TOKENIZER_DIR、VOCAB_SIZE。
python minillm/trainer/train_tokenizer.py