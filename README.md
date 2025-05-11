# MiniLLM

MiniLLM is a lightweight and efficient implementation of the Language model。Similar to LLaMA, it is designed to be easy to use and integrate into various applications. The model is built using PyTorch and is optimized for performance and memory usage. For more details, please refer to the [MiniMind](https://github.com/jingyaogong/minimind)

# Tokenizer

BPE.

chat_template

```txt
{% if messages[0]['role'] == 'system' %}
    {% set system_message = messages[0]['content'] %}
    {{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}
{% else %}
    {{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}
{% endif %}
{% for message in messages %}
    {% set content = message['content'] %}
    {% if message['role'] == 'user' %}
        {{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ content + '<|im_end|>' + '\\n' }}
    {% endif %}
{% endfor %}
```

# Preparation

```sh
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install python dependencies
uv install python 3.13

# git clone the repo
git clone https://github.com/wenzhaoabc/minillm.git

# sync the repo
uv sync
```

# Pretrain

**Single GPU**

```bash
# Command to run the pretraining on a single GPU
python -m minillm.train.pretrain \
    --data_path /root/autodl-tmp/data/pretrain_hq.jsonl \
    --tokenizer_path /root/autodl-tmp/minillm/minillm/tokenizer \
    --batch_size 64 \
    --save_interval 500 \
    --use_moe
```

# Supervised Fine-tuning

```bash
conda activate /root/autodl-tmp/envs/minillm/

modelscope download gongjy/minimind_dataset sft_mini_512.jsonl --local_dir /root/autodl-tmp/data --repo-type dataset


python -m minillm.train.sft \
    --out_dir /root/autodl-tmp/sft_out \
    --data_path /root/autodl-tmp/data/sft_mini_512.jsonl \
    --tokenizer_path /root/autodl-tmp/minillm/minillm/tokenizer \
    --model_path  /root/autodl-tmp/out/checkpoint_epoch_0_step_9999.pt \
    --epochs 2 \
    --batch_size 32 \
    --save_interval 1000 \
    --use_moe
```

# RLHF

```bash
python -m minillm.train.dpo \
    --out_dir /root/autodl-tmp/ckp/dpo/ \
    --data_path /root/autodl-tmp/data/dpo.jsonl \
    --tokenizer_path /root/minillm/minillm/tokenizer \
    --model_path  /root/autodl-tmp/ckp/sft/sft_ckp_epoch_1_step_4999.pt \
    --epochs 2 \
    --batch_size 64 \
    --save_interval 1000 \
    --use_moe
```

# Distillation

```bash
modelscope download gongjy/minimind_dataset r1_mix_1024.jsonl --local_dir /root/autodl-tmp/data --repo-type dataset

python -m minillm.train.distill_reason \
    --out_dir /root/autodl-tmp/ckp/dis_cp/ \
    --data_path /root/autodl-tmp/data/r1_mix_1024.jsonl \
    --tokenizer_path /root/minillm/minillm/tokenizer \
    --model_path  /root/autodl-tmp/ckp/dpo/dpo_ckp_epoch_1_step_4999.pt \
    --epochs 2 \
    --batch_size 64 \
    --save_interval 150 \
    --use_moe
```

# Inference

```bash
python -m minillm.inference.server_api \
    --model_path /root/autodl-tmp/dpo_cp_e1_s999.pt \
    --tokenizer_path /root/minillm/minillm/tokenizer \
    --port 6006 \
    --use_moe
```