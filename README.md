# MiniLLM

MiniLLM is a lightweight and efficient implementation of the Language model. Similar to LLaMA, it is designed to be easy to use and integrate into various applications. The aim of this project is to implement the processes of structural design, pre-training, SFT, RLHF and deployment inference of a large language model from scratch. In addition, lora fine-tuning and model distillation have been achieved.

## Folder Structure

```plaintext
minillm/
├─__init__.py
├─inference                 # Inference related files
│      server_api.py
├─model                     # Model architecture and configuration files    
│      config.py
│      lora.py
│      model_base.py
│      model_v1.py
├─rlhf                      # Reinforcement Learning from Human Feedback (RLHF) related files
│      ds_rm.py
│      grpo.py
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
        train_util.py
```

## Quick Start

Install the dependencies:

```sh
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# git clone the repo
git clone https://github.com/wenzhaoabc/minillm.git

# create env
uv venv --prompt minillm --python 3.13

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
For MoE model, see [LLM-MoE](./images/LLM-structure-moe.png). The two structure images are from [MiniMind](https://github.com/jingyaogong/minimind).

## Tokenizer

**BPE (Byte Pair Encoding)**

BPE is a data-driven subword tokenization algorithm. The core idea:

1. **Initialize**: Start with a character-level vocabulary of all unique characters in the corpus.
2. **Count pairs**: Iteratively count the frequency of all adjacent symbol pairs in the corpus.
3. **Merge**: Merge the most frequent pair into a new symbol and add it to the vocabulary.
4. **Repeat**: Continue merging until the vocabulary reaches the target size.

This produces a vocabulary of subword units — common words become single tokens, while rare words are split into smaller subword pieces. BPE strikes a balance between character-level and word-level tokenization, handling out-of-vocabulary words gracefully.

The tokenizer in this project uses HuggingFace `PreTrainedTokenizerFast` with:

- **Vocabulary size**: 6,400
- **Merge rules**: 6,141
- **Special tokens**: `<|endoftext|>` (pad/unk), `<|im_start|>` (bos), `<|im_end|>` (eos)

**Related files**:

- [`minillm/tokenizer/tokenizer.json`](./minillm/tokenizer/tokenizer.json) — BPE vocabulary and merge rules
- [`minillm/tokenizer/tokenizer_config.json`](./minillm/tokenizer/tokenizer_config.json) — tokenizer configuration and chat template
- [`minillm/tests/test_chat_template.py`](./minillm/tests/test_chat_template.py) — chat template rendering test

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

Datasets: [gongjy/minimind_dataset](https://huggingface.co/datasets/gongjy/minimind_dataset)

---

### Pretraining

**Script**: [`minillm/train/pretrain.py`](./minillm/train/pretrain.py)

**Objective**: Next-token prediction over large-scale unlabeled text. The model learns the statistical structure of language by maximizing the likelihood of each token given its context.

$$\mathcal{L}_{\text{PT}} = -\frac{1}{N} \sum_{t=1}^{N} \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})$$

For MoE models, an auxiliary load-balancing loss is added to encourage uniform expert utilization:

$$\mathcal{L} = \mathcal{L}_{\text{PT}} + \mathcal{L}_{\text{aux}}$$

The learning rate follows a cosine decay schedule:

$$\text{lr}(t) = \frac{\text{lr}_{\min}}{10} + \frac{1}{2} \cdot \text{lr}_{\max} \cdot \left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

```bash
# Single-GPU pretraining
python -m minillm.train.pretrain \
    --data_path /root/autodl-tmp/data/pretrain_hq.jsonl \
    --tokenizer_path /root/autodl-tmp/minillm/minillm/tokenizer \
    --batch_size 64 \
    --save_interval 500 \
    --use_moe
```

---

### Supervised Fine-tuning (SFT)

**Script**: [`minillm/train/sft.py`](./minillm/train/sft.py)

**Objective**: Fine-tune on instruction-response pairs. Loss is computed **only on assistant response tokens** (controlled by `loss_mask`), teaching the model to follow instructions without forgetting the base language understanding.

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{|\mathcal{A}|} \sum_{t \in \mathcal{A}} \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})$$

where $\mathcal{A}$ is the set of token positions belonging to assistant responses (i.e., tokens between `<|im_start|>assistant` and `<|im_end|>`).

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

---

### Direct Preference Optimization (DPO)

**Script**: [`minillm/train/dpo.py`](./minillm/train/dpo.py)

**Objective**: Align the model with human preferences **without training a separate reward model**. Given a preferred response $y_w$ and a rejected response $y_l$ for the same prompt $x$, DPO directly optimizes the policy by implicitly treating the log-ratio as a reward signal.

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right) \right]$$

where $\pi_{\text{ref}}$ is a frozen copy of the pre-trained model and $\beta = 0.1$ controls the KL penalty strength. Log probabilities are averaged over response length.

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

---

### Distillation from Reasoning Data

**Script**: [`minillm/train/distill_reason.py`](./minillm/train/distill_reason.py)

**Objective**: Distill chain-of-thought reasoning capability from teacher-generated R1-style data. The training objective is the same cross-entropy as SFT, but **format tokens** (`<think>`, `</think>`, `<answer>`, `</answer>`) are assigned a **10× higher loss weight** to strongly enforce the reasoning format.

$$\mathcal{L}_{\text{distill}} = \frac{\sum_{t \in \mathcal{A}} w_t \cdot \text{CE}(x_t, \hat{x}_t)}{\sum_{t \in \mathcal{A}} w_t}, \quad w_t = \begin{cases} 10 & \text{if } x_t \in \text{format tokens} \\ 1 & \text{otherwise} \end{cases}$$

```bash
# Download reasoning dataset
modelscope download gongjy/minimind_dataset r1_mix_1024.jsonl \
    --local_dir /root/autodl-tmp/data --repo-type dataset

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

---

## Reinforcement Learning from Human Feedback (RLHF)

RLHF pipeline: train a reward model on human preference data, then optimize the policy with PPO or GRPO.

### Reward Model

**Script**: [`minillm/rlhf/train_rm.py`](./minillm/rlhf/train_rm.py)

**Objective**: Learn a scalar reward $r_\theta(x, y)$ from human preference pairs $(y_w \succ y_l \mid x)$ using the Bradley-Terry preference model. A mean-zero regularization term prevents reward hacking by keeping reward magnitudes bounded.

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right] + c \cdot \mathbb{E}\left[ \left( r_\theta(x, y_w) + r_\theta(x, y_l) \right)^2 \right]$$

where $c = 0.01$ is the centering coefficient ([ref](https://huggingface.co/papers/2312.09244)).

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

### PPO (Proximal Policy Optimization)

**Script**: [`minillm/rlhf/ppo.py`](./minillm/rlhf/ppo.py)

**Objective**: Optimize the policy using the PPO clipped surrogate objective with a value function baseline. The reward signal comes from the trained reward model. A KL penalty against the reference policy prevents over-optimization.

$$\mathcal{L}_{\text{PPO}}(\theta) = -\mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\varepsilon,\ 1+\varepsilon) \hat{A}_t \right) \right] + c_1 \mathcal{L}_{\text{VF}} - c_2 \mathcal{H}[\pi_\theta]$$

where $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$ is the probability ratio, $\hat{A}_t$ is the GAE advantage, $\mathcal{L}_{\text{VF}}$ is the value function loss, and $\mathcal{H}[\pi_\theta]$ is an entropy bonus.

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

### GRPO (Group Relative Policy Optimization)

**Script**: [`minillm/rlhf/grpo.py`](./minillm/rlhf/grpo.py)

**Objective**: Reinforcement learning via policy gradient **without a value network**. For each prompt, $G$ responses are sampled and scored. Advantages are computed via group-level normalization, reducing variance without requiring a critic.

**Step 1 — Group normalized advantage**:

$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \varepsilon}, \quad \mu_r = \frac{1}{G}\sum_{i=1}^G r_i, \quad \sigma_r = \text{std}(r_1, \ldots, r_G)$$

**Step 2 — Per-token policy loss with KL penalty** (reward is rule-based: format correctness of `<think>`/`<answer>` tags):

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{|o|} \sum_t \left[ \frac{\pi_\theta(o_t \mid x, o_{<t})}{\pi_{\theta_{\text{old}}}(o_t \mid x, o_{<t})} \cdot \hat{A} - \beta \cdot \underbrace{\left(e^{\log\pi_{\text{ref}} - \log\pi_\theta} - (\log\pi_{\text{ref}} - \log\pi_\theta) - 1\right)}_{\text{KL approximation}} \right]$$

where $\beta = 0.02$ and the KL term uses the unbiased approximation $e^d - d - 1$ with $d = \log\pi_{\text{ref}} - \log\pi_\theta$.

```bash
python -m minillm.rlhf.grpo \
    --data_path /root/autodl-tmp/data/rlaif-mini.jsonl \
    --save_dir /root/autodl-tmp/ckp/grpo/ \
    --epochs 1 \
    --batch_size 2 \
    --num_generations 8 \
    --beta 0.02 \
    --reasoning 1 \
    --use_moe 1
```

---

## Inference

```bash
python -m minillm.inference.server_api \
    --model_path /root/autodl-tmp/ckp/dis_cp/checkpoint_epoch_0_step_3899.pt \
    --tokenizer_path /root/minillm/minillm/tokenizer \
    --port 6006 \
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
