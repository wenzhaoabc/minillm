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

# Pretrain

**Single GPU**

```bash
# Command to run the pretraining on a single GPU
python train/pretrain.py --data_path /content/pretrain_hq.jsonl
```