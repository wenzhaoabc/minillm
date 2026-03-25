import os
import sys


import time
import argparse
from pathlib import Path
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from minillm.model.model_base import MiniLLMConfig
from minillm.model.model_io import load_minillm_model
from datasets import load_dataset
from minillm.dataset.lm_dataset import normalize_tool_chat_sample, post_processing_chat
from minillm.trainer.trainer_utils import Logger, LogMetrics, setup_seed, get_model_params, resolve_repo_path
warnings.filterwarnings('ignore')


def resolve_minillm_artifact_dir(load_from):
    explicit_load_path = Path(resolve_repo_path(load_from))
    if explicit_load_path.is_dir() and (explicit_load_path / 'config.json').exists():
        return explicit_load_path
    return None


def load_minillm_config(args, artifact_dir):
    if artifact_dir is not None:
        config_path = artifact_dir / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(f'MiniLLM config not found in artifact directory: {config_path}')
        lm_config = MiniLLMConfig.from_pretrained(str(artifact_dir))
        if args.inference_rope_scaling:
            lm_config.inference_rope_scaling = True
            lm_config.rope_scaling = {
                'beta_fast': 32,
                'beta_slow': 1,
                'factor': 16,
                'original_max_position_embeddings': 2048,
                'attention_factor': 1.0,
                'type': 'yarn',
            }
        Logger(f'Parsed MiniLLM config from {config_path}')
        return lm_config

def init_model(args):
    load_path = resolve_repo_path(args.load_from)
    artifact_dir = resolve_minillm_artifact_dir(args.load_from)
    if artifact_dir is not None:
        lm_config = load_minillm_config(args, artifact_dir)
        load_path = str(artifact_dir)
        Logger(f'Loading MiniLLM artifact from {load_path}')
        model, tokenizer = load_minillm_model(load_path, lm_config=lm_config, tokenizer_path=args.tokenizer_path, device=args.device, strict=True)
    else:
        Logger(f'Loading Transformers model from {load_path}')
        tokenizer = AutoTokenizer.from_pretrained(load_path)
        model = AutoModelForCausalLM.from_pretrained(load_path, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer


def load_eval_samples(data_path, split, sample_index, max_samples):
    samples = load_dataset('json', data_files=resolve_repo_path(data_path), split=split)
    if sample_index < 0 or sample_index >= len(samples):
        raise IndexError(f'sample_index={sample_index} out of range, dataset size={len(samples)}')
    end_index = min(len(samples), sample_index + max_samples)
    return [samples[index] for index in range(sample_index, end_index)]


def build_generation_case(sample, tokenizer):
    if 'conversations' in sample:
        messages, tools = normalize_tool_chat_sample(sample['conversations'])
        reference = None
        prompt_messages = messages
        if messages and messages[-1]['role'] == 'assistant':
            reference = messages[-1].get('content', '')
            prompt_messages = messages[:-1]
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
        )
        prompt = post_processing_chat(prompt, empty_think_ratio=1.0)
        return prompt, reference

    if 'chosen' in sample:
        messages, tools = normalize_tool_chat_sample(sample['chosen'])
        reference = None
        prompt_messages = messages
        if messages and messages[-1]['role'] == 'assistant':
            reference = messages[-1].get('content', '')
            prompt_messages = messages[:-1]
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
        )
        prompt = post_processing_chat(prompt, empty_think_ratio=1.0)
        return prompt, reference

    if 'text' in sample:
        text = str(sample['text'])
        return tokenizer.bos_token + text, None

    raise ValueError('Unsupported sample format. Expected one of: conversations / chosen / text.')

def main():
    parser = argparse.ArgumentParser(description="MiniLLM模型推理与对话")
    parser.add_argument('--load_from', required=True, type=str, help="模型目录路径；支持MiniLLM artifact目录或transformers模型目录")
    parser.add_argument('--tokenizer_path', default='minillm/tokenizer', type=str, help="仅在模型目录中不存在tokenizer时作为回退路径")
    parser.add_argument('--data_path', required=True, type=str, help="评测样本数据路径（json/jsonl）")
    parser.add_argument('--split', default='train', type=str, help="数据集split名")
    parser.add_argument('--sample_index', default=0, type=int, help="起始样本索引")
    parser.add_argument('--max_samples', default=1, type=int, help="最多评测多少条样本")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()

    model, tokenizer = init_model(args)
    Logger(f'Evaluation device: {args.device}')
    samples = load_eval_samples(args.data_path, args.split, args.sample_index, args.max_samples)
    Logger(f'Loaded {len(samples)} sample(s) from {resolve_repo_path(args.data_path)}')
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    for offset, sample in enumerate(samples):
        setup_seed(2026)
        prompt, reference = build_generation_case(sample, tokenizer)
        if reference:
            Logger(f'Sample[{args.sample_index + offset}] reference={reference}')
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(args.device)

        print(f'===== Sample {args.sample_index + offset} =====')
        print('Prompt:\n' + prompt)
        print('\nResponse:\n', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        if args.show_speed:
            LogMetrics('decode', prompt_tokens=len(inputs['input_ids'][0]), generated_tokens=gen_tokens, tokens_per_second=f'{gen_tokens / (time.time() - st):.2f}')
            print()
        else:
            print('\n')
        if reference:
            print('Reference:\n' + reference)
            print()

if __name__ == "__main__":
    main()