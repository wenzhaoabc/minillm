import argparse
import json
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence, Type

import torch
from transformers import HfArgumentParser


@dataclass
class ModelConfigArgs:
    dropout: float = 0.0
    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 512
    intermediate_size: Optional[int] = None
    max_position_embeddings: int = 32768
    num_attention_heads: int = 8
    num_hidden_layers: int = 8
    num_key_value_heads: int = 2
    vocab_size: int = 6400
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    inference_rope_scaling: bool = False
    flash_attn: bool = True
    flash_attn_impl: str = "auto"
    use_moe: bool = False
    num_experts_per_tok: int = 2
    n_routed_experts: int = 4
    n_shared_experts: int = 1
    scoring_func: str = "softmax"
    aux_loss_alpha: float = 0.01
    seq_aux: bool = True
    norm_topk_prob: bool = True


@dataclass
class TrainConfigArgs:
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 5e-4
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    num_workers: int = 8
    accumulation_steps: int = 1
    grad_clip: float = 1.0
    log_interval: int = 100
    save_interval: int = 1000
    max_seq_len: int = 340
    seed: int = 42
    use_compile: bool = False


@dataclass
class ScriptConfigArgs:
    data_path: str = ""
    tokenizer_path: str = "minillm/tokenizer"
    output_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    load_dir: Optional[str] = None
    load_from: Optional[str] = None
    from_weight: Optional[str] = None
    from_resume: bool = False
    use_wandb: bool = True
    wandb_project: str = "MiniLLM-Train"
    wandb_entity: str = "minillm"
    wandb_group: Optional[str] = None
    run_name: Optional[str] = None
    save_dir: str = "../out"
    save_weight: str = "pretrain"


@dataclass
class DPOConfigArgs:
    beta: float = 0.1


@dataclass
class PPOConfigArgs:
    critic_learning_rate: float = 8e-8
    max_gen_len: int = 1536
    clip_epsilon: float = 0.1
    vf_coef: float = 0.5
    kl_coef: float = 0.02
    reasoning: bool = True
    update_old_actor_freq: int = 4
    reward_model_path: str = "../../internlm2-1_8b-reward"


@dataclass
class GRPOConfigArgs:
    max_gen_len: int = 1536
    num_generations: int = 8
    beta: float = 0.02
    reasoning: bool = True
    reward_model_path: str = "../../internlm2-1_8b-reward"


@dataclass
class SPOConfigArgs:
    max_gen_len: int = 1536
    beta: float = 0.02
    reasoning: bool = True
    reward_model_path: str = "../../internlm2-1_8b-reward"


@dataclass
class LoRAConfigArgs:
    lora_name: str = "lora_identity"
    lora_rank: int = 8


def _load_json(path: Optional[str]):
    if not path:
        return {}
    with open(Path(path).expanduser(), "r", encoding="utf-8") as f:
        return json.load(f)


def _filter_for_dataclass(dc_type: Type, payload: dict):
    allowed = {field.name for field in fields(dc_type)}
    return {key: value for key, value in payload.items() if key in allowed}


def _parse_from_config_files(dataclass_types: Sequence[Type], argv: Sequence[str]):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str)
    bootstrap.add_argument("--model_config", type=str)
    bootstrap.add_argument("--train_config", type=str)
    bootstrap.add_argument("--script_config", type=str)
    bootstrap_args, _ = bootstrap.parse_known_args(argv)

    has_config = any(
        [
            bootstrap_args.config,
            bootstrap_args.model_config,
            bootstrap_args.train_config,
            bootstrap_args.script_config,
        ]
    )
    if not has_config:
        return None

    merged = {}
    unified = _load_json(bootstrap_args.config)
    if unified:
        for key in ("model_config", "train_config", "script_config"):
            merged.update(unified.get(key, {}))
        for key, value in unified.items():
            if key not in {"model_config", "train_config", "script_config"}:
                merged[key] = value

    merged.update(_load_json(bootstrap_args.model_config))
    merged.update(_load_json(bootstrap_args.train_config))
    merged.update(_load_json(bootstrap_args.script_config))

    return tuple(dc_type(**_filter_for_dataclass(dc_type, merged)) for dc_type in dataclass_types)


def parse_config_groups(*extra_dataclasses: Type, argv: Optional[Sequence[str]] = None):
    dataclass_types = (ModelConfigArgs, TrainConfigArgs, ScriptConfigArgs, *extra_dataclasses)
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parsed_from_files = _parse_from_config_files(dataclass_types, raw_argv)
    if parsed_from_files is not None:
        return parsed_from_files

    parser = HfArgumentParser(dataclass_types)
    return parser.parse_args_into_dataclasses(args=raw_argv)


def namespace_from_configs(*configs):
    merged = {}
    for config in configs:
        merged.update(asdict(config))
    return SimpleNamespace(**merged)