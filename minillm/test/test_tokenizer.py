import os
import sys

import argparse
import json

from datasets import load_dataset
from transformers import AutoTokenizer

from minillm.dataset.lm_dataset import (
    normalize_tool_chat_sample,
    post_processing_chat,
    pre_processing_chat,
)
from minillm.trainer.trainer_utils import Logger, resolve_repo_path


def load_sample(data_path, split, sample_index):
    samples = load_dataset("json", data_files=resolve_repo_path(data_path), split=split)
    if sample_index < 0 or sample_index >= len(samples):
        raise IndexError(
            f"sample_index={sample_index} out of range, dataset size={len(samples)}"
        )
    return samples[sample_index]


def render_conversation_sample(sample, tokenizer, apply_preprocess=False, add_generation_prompt=False):
    conversations = sample["conversations"]
    if apply_preprocess:
        conversations = pre_processing_chat(conversations, add_system_ratio=0.0)
    messages, tools = normalize_tool_chat_sample(conversations)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
    )
    rendered = post_processing_chat(rendered, empty_think_ratio=1.0)
    return {
        "mode": "conversation",
        "messages": messages,
        "tools": tools,
        "rendered": rendered,
    }


def render_dpo_sample(sample, tokenizer, add_generation_prompt=False):
    outputs = {}
    for key in ("chosen", "rejected"):
        messages, tools = normalize_tool_chat_sample(sample[key])
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )
        outputs[key] = {
            "messages": messages,
            "tools": tools,
            "rendered": post_processing_chat(rendered, empty_think_ratio=1.0),
        }
    return {"mode": "dpo", **outputs}


def render_pretrain_sample(sample, tokenizer):
    text = str(sample["text"])
    return {"mode": "pretrain", "rendered": text, "messages": None, "tools": None}


def inspect_sample(sample, tokenizer, apply_preprocess=False, add_generation_prompt=False):
    if "conversations" in sample:
        return render_conversation_sample(
            sample,
            tokenizer,
            apply_preprocess=apply_preprocess,
            add_generation_prompt=add_generation_prompt,
        )
    if "chosen" in sample and "rejected" in sample:
        return render_dpo_sample(
            sample,
            tokenizer,
            add_generation_prompt=add_generation_prompt,
        )
    if "text" in sample:
        return render_pretrain_sample(sample, tokenizer)
    raise ValueError(
        "Unsupported sample format. Expected one of: conversations / chosen+rejected / text."
    )


def print_render_result(result, tokenizer, max_chars):
    if result["mode"] == "dpo":
        for branch in ("chosen", "rejected"):
            branch_result = result[branch]
            Logger(f"[{branch}] tools={json.dumps(branch_result['tools'], ensure_ascii=False, indent=2) if branch_result['tools'] else 'None'}")
            Logger(f"[{branch}] messages=\n{json.dumps(branch_result['messages'], ensure_ascii=False, indent=2)}")
            rendered = branch_result["rendered"]
            token_ids = tokenizer(rendered, add_special_tokens=False).input_ids
            Logger(f"[{branch}] rendered_length={len(rendered)}, token_count={len(token_ids)}")
            print(rendered[:max_chars])
            if len(rendered) > max_chars:
                print("\n... [truncated] ...\n")
        return

    if result["messages"] is not None:
        Logger(
            f"tools={json.dumps(result['tools'], ensure_ascii=False, indent=2) if result['tools'] else 'None'}"
        )
        Logger(f"messages=\n{json.dumps(result['messages'], ensure_ascii=False, indent=2)}")

    rendered = result["rendered"]
    token_ids = tokenizer(rendered, add_special_tokens=False).input_ids
    Logger(f"rendered_length={len(rendered)}, token_count={len(token_ids)}")
    print(rendered[:max_chars])
    if len(rendered) > max_chars:
        print("\n... [truncated] ...\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect tokenizer chat template output for dataset samples")
    parser.add_argument("--data_path", required=True, type=str, help="JSON/JSONL dataset path")
    parser.add_argument("--tokenizer_path", default="minillm/tokenizer", type=str, help="Tokenizer directory")
    parser.add_argument("--split", default="train", type=str, help="Dataset split name")
    parser.add_argument("--sample_index", default=0, type=int, help="Sample index to inspect")
    parser.add_argument("--apply_preprocess", action="store_true", help="Apply the same pre_processing_chat step as SFTDataset")
    parser.add_argument("--add_generation_prompt", action="store_true", help="Pass add_generation_prompt=True to apply_chat_template")
    parser.add_argument("--max_chars", default=4000, type=int, help="Maximum rendered characters to print")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(resolve_repo_path(args.tokenizer_path))
    sample = load_sample(args.data_path, args.split, args.sample_index)
    Logger(f"Loaded sample_index={args.sample_index} from {resolve_repo_path(args.data_path)}")
    Logger(f"sample_keys={list(sample.keys())}")
    result = inspect_sample(
        sample,
        tokenizer,
        apply_preprocess=args.apply_preprocess,
        add_generation_prompt=args.add_generation_prompt,
    )
    print_render_result(result, tokenizer, args.max_chars)


if __name__ == "__main__":
    main()


# python minillm/test/test_tokenizer.py --data_path datasets/sft_t2t_mini.jsonl --sample_index 0