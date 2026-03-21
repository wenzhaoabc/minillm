# MiniLLM

MiniLLM is a lightweight LLM training and inference project.  
The codebase now uses the new `minillm/` layout (model + dataset + trainer + infer + test), and legacy paths from `minillm_old/` are no longer used in command examples.

## Folder Structure

```text
minillm/
├─dataset/
│  ├─__init__.py
│  └─lm_dataset.py
├─infer/
│  ├─__init__.py
│  ├─chat_openai_api.py
│  ├─convert_model.py
│  ├─serve_openai_api.py
│  └─web_demo.py
├─model/
│  ├─config.py
│  ├─model_lora.py
│  ├─model_minillm.py
│  └─triton_flash_attn.py
├─test/
│  ├─__init__.py
│  └─eval_llm.py
├─tokenizer/
│  ├─tokenizer.json
│  └─tokenizer_config.json
└─trainer/
   ├─__init__.py
   ├─trainer_utils.py
   ├─train_pretrain.py
   ├─train_full_sft.py
   ├─train_dpo.py
   ├─train_reason.py
   ├─train_distillation.py
   ├─train_lora.py
   ├─train_ppo.py
   ├─train_grpo.py
   ├─train_spo.py
   └─train_tokenizer.py
```

## Quick Start

```bash
# 1) create virtual env
uv venv --prompt minillm --python 3.13
source .venv/bin/activate

# 2) install dependencies
uv sync
```

## Training Entrypoints

Run from repository root using module mode (`python -m ...`).

### Pretraining

Script: [minillm/trainer/train_pretrain.py](minillm/trainer/train_pretrain.py)

```bash
python -m minillm.trainer.train_pretrain \
  --data_path ./dataset/pretrain_hq.jsonl \
  --save_dir ./out \
  --epochs 1 \
  --batch_size 32
```

### Full SFT

Script: [minillm/trainer/train_full_sft.py](minillm/trainer/train_full_sft.py)

```bash
python -m minillm.trainer.train_full_sft \
  --data_path ./dataset/sft_mini_512.jsonl \
  --save_dir ./out \
  --epochs 2 \
  --batch_size 16
```

### DPO

Script: [minillm/trainer/train_dpo.py](minillm/trainer/train_dpo.py)

```bash
python -m minillm.trainer.train_dpo \
  --data_path ./dataset/dpo.jsonl \
  --save_dir ./out \
  --epochs 1 \
  --batch_size 4
```

### Reason Distillation

Script: [minillm/trainer/train_reason.py](minillm/trainer/train_reason.py)

```bash
python -m minillm.trainer.train_reason \
  --data_path ./dataset/r1_mix_1024.jsonl \
  --save_dir ./out \
  --epochs 1 \
  --batch_size 8
```

### Knowledge Distillation

Script: [minillm/trainer/train_distillation.py](minillm/trainer/train_distillation.py)

```bash
python -m minillm.trainer.train_distillation \
  --data_path ./dataset/sft_mini_512.jsonl \
  --save_dir ./out \
  --epochs 6 \
  --batch_size 32
```

### LoRA Fine-tuning

Script: [minillm/trainer/train_lora.py](minillm/trainer/train_lora.py)

```bash
python -m minillm.trainer.train_lora \
  --data_path ./dataset/lora_identity.jsonl \
  --save_dir ./out/lora \
  --epochs 50 \
  --batch_size 32
```

### RLHF-style Trainers

- PPO: [minillm/trainer/train_ppo.py](minillm/trainer/train_ppo.py)
- GRPO: [minillm/trainer/train_grpo.py](minillm/trainer/train_grpo.py)
- SPO: [minillm/trainer/train_spo.py](minillm/trainer/train_spo.py)

Example:

```bash
python -m minillm.trainer.train_grpo \
  --data_path ./dataset/rlaif-mini.jsonl \
  --save_dir ./out \
  --epochs 1 \
  --batch_size 2
```

## Inference

### Local Chat / Eval

Script: [minillm/test/eval_llm.py](minillm/test/eval_llm.py)

```bash
python -m minillm.test.eval_llm \
  --load_from model \
  --save_dir out \
  --weight full_sft
```

### OpenAI-Compatible API Server

Script: [minillm/infer/serve_openai_api.py](minillm/infer/serve_openai_api.py)

```bash
python -m minillm.infer.serve_openai_api \
  --load_from ../model \
  --save_dir out \
  --weight full_sft \
  --hidden_size 512 \
  --num_hidden_layers 8
```

### Streamlit Web Demo

Script: [minillm/infer/web_demo.py](minillm/infer/web_demo.py)

```bash
streamlit run minillm/infer/web_demo.py
```

### Model Conversion

Script: [minillm/infer/convert_model.py](minillm/infer/convert_model.py)

```bash
python -m minillm.infer.convert_model
```

## Triton Flash Attention

Triton implementation has been migrated to [minillm/model/triton_flash_attn.py](minillm/model/triton_flash_attn.py) and integrated into [minillm/model/model_minillm.py](minillm/model/model_minillm.py).

The model will use Triton flash-attention only when all conditions are met:

- CUDA is available
- inference/eval mode (not training)
- Q/K/V are CUDA tensors with same dtype
- dtype is float16 or bfloat16
- head dimension is one of 16, 32, 64, 128
- attention mask is all-ones (or `None`)

If conditions are not met, it automatically falls back to SDPA.

## Acknowledgements

- [MiniMind](https://github.com/jingyaogong/minimind)
- [llm-tap-rl](https://github.com/wenzhaoabc/llm-tap-rl)
- [trl](https://github.com/huggingface/trl)
- [simple_grpo](https://github.com/lsdefine/simple_GRPO)

## License

MIT License. See [LICENSE](LICENSE).
