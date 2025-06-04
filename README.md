# MiniLLM

MiniLLM is a lightweight and efficient implementation of the Language model。Similar to LLaMA, it is designed to be easy to use and integrate into various applications. The model is built using PyTorch and is optimized for performance and memory usage. For more details, please refer to the [MiniMind](https://github.com/jingyaogong/minimind)

## Folder Structure

```plaintext
minillm/
├─inference                 # Inference related files
│      server_api.py
├─model                     # Model architecture and configuration files    
│      config.py
│      lora.py
│      model_base.py
│      model_v1.py
├─rlhf                      # Reinforcement Learning from Human Feedback (RLHF) related files
│      ds_rm.py
│      ppo.py
│      reward_model.py
│      train_rm.py
├─tests
│      eval_model.py
│      pretrain_test.py
│      test_chat_template.py
├─tokenizer
│      tokenizer.json
│      tokenizer_config.json
├─train                      # Training related files    
│      dataset.py
│      distill_reason.py
│      dpo.py
│      pretrain.py
│      sft.py
│      sft_lora.py
│      train_sft.py
└─utils
        log_example.py
        mllog.py
```

## Quick Start

Install the dependencies:

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

## Model Architecture

MiniLLM is based on the Transformer architecture, specifically designed for language modeling tasks. The basic architecture is similar to that of LLaMA.

- **Embedding Layer**: Converts input tokens into dense vectors.
- **Transformer Blocks**: Each block consists of:
    - Rotary Positional Embedding: Adds positional information to the input tokens.
    - Multi-Group Self-Attention: Allows the model to focus on different parts of the input sequence.
    - Layer Normalization: Normalizes the output of each layer to stabilize training.
    - Feed-Forward Neural Network: Applies a non-linear transformation to the attention output.
    - Mixture of Experts (MoE): optional

For dense model, see [LLM-Dense](./images/LLM-structure.png).
For MoE model, see [LLM-MoE](./images/LLM-structure-moe.png).

## Tokenizer

BPE.

chat_template:

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

## Model Training

Datasets: gongjy/minimind_dataset

**Pretraining**

```bash
# Command to run the pretraining on a single GPU
python -m minillm.train.pretrain \
    --data_path /root/autodl-tmp/data/pretrain_hq.jsonl \
    --tokenizer_path /root/autodl-tmp/minillm/minillm/tokenizer \
    --batch_size 64 \
    --save_interval 500 \
    --use_moe
```

**Supervised Fine-tuning**

```bash
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

**DPO**

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

**Distillation from Reasoning Data**

```bash
modelscope download gongjy/minimind_dataset r1_mix_1024.jsonl --local_dir /root/autodl-tmp/data --repo-type dataset

python -m minillm.train.distill_reason \
    --out_dir /root/autodl-tmp/ckp/dis_cp/ \
    --data_path /root/autodl-tmp/data/r1_mix_1024.jsonl \
    --tokenizer_path /root/minillm/minillm/tokenizer \
    --model_path  /root/autodl-tmp/ckp/distill_reason_e0_s5249.pt \
    --epochs 1 \
    --batch_size 32 \
    --save_interval 100 \
    --use_moe
```

**Inference**

```bash
python -m minillm.inference.server_api \
    --model_path /root/autodl-tmp/ckp/dis_cp/checkpoint_epoch_0_step_3899.pt \
    --tokenizer_path /root/minillm/minillm/tokenizer \
    --port 6006 \
    --use_moe
```

## Reinforcement Learning from Human Feedback (RLHF)

Implementation of RLHF using PPO (Proximal Policy Optimization) algorithm.

**Train Reward Model**

```bash
python -m minillm.rlhf.train_rm \
    --out_dir outputs/ \
    --data_path data/reward_model.jsonl \
    --tokenizer_path minillm/tokenizer \
    --model_path  /root/autodl-tmp/ckp/dpo/dpo_ckp_epoch_1_step_4999.pt \
    --epochs 2 \
    --batch_size 2 \
    --hidden_size 128 \
    --num_hidden_layers 2 \
    --max_seq_len 64 \
    --save_interval 2 \
    --use_moe
```

**Train PPO Agent**

```bash
python -m minillm.rlhf.ppo \
    --out_dir outputs/ \
    --data_path data/ppo_data.jsonl \
    --tokenizer_path minillm/tokenizer \
    --model_path  /root/autodl-tmp/ckp/dpo/dpo_ckp_epoch_1_step_4999.pt \
    --reward_model_path /root/autodl-tmp/ckp/rm/rm_ckp_epoch_1_step_999.pt \
    --epochs 2 \
    --batch_size 2 \
    --hidden_size 128 \
    --num_hidden_layers 2 \
    --max_seq_len 64 \
    --save_interval 2 \
    --use_moe
```

## Acknowledgements

This project is inspired by the following works:

- [minimind](https://github.com/jingyaogong/minimind)
- [llm-tap-rl](https://github.com/wenzhaoabc/llm-tap-rl)
- [trl](https://github.com/huggingface/trl)
- [simple_grpo](https://github.com/lsdefine/simple_GRPO)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
