"""
训练工具函数集合
"""

import os
import sys


import random
import math
import json
import shutil
import inspect
import importlib
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from minillm.model.model_base import MiniLLMForCausalLM
from minillm.model.model_io import (
    load_minillm_model,
    load_model_state,
    save_model_artifacts,
    save_model_state,
    save_training_state,
    load_training_state,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path_like):
    if path_like is None:
        return None
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def build_lm_config_from_args(args, overrides=None):
    """Build MiniLLMConfig kwargs dynamically from argparse namespace/dict."""
    from minillm.model.model_base import MiniLLMConfig

    src = vars(args) if hasattr(args, "__dict__") else dict(args)
    sig = inspect.signature(MiniLLMConfig.__init__)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name in {"self", "kwargs"} or name not in src:
            continue
        value = src[name]
        if isinstance(param.default, bool) and isinstance(value, int):
            value = bool(value)
        kwargs[name] = value
    if overrides:
        kwargs.update(overrides)
    return MiniLLMConfig(**kwargs)


def build_artifact_name(lm_config, name):
    moe_suffix = "_moe" if lm_config.use_moe else ""
    return f"{name}_{lm_config.hidden_size}{moe_suffix}"


def resolve_output_dir(lm_config, output_dir=None, save_dir=None, save_weight=None):
    if output_dir:
        return resolve_repo_path(output_dir)
    if save_dir and save_weight:
        return os.path.join(
            resolve_repo_path(save_dir), build_artifact_name(lm_config, save_weight)
        )
    raise ValueError(
        "Either output_dir or both save_dir and save_weight must be provided"
    )


def resolve_checkpoint_dir(output_dir=None, checkpoint_dir=None):
    if checkpoint_dir:
        return resolve_repo_path(checkpoint_dir)
    if output_dir:
        return str(Path(resolve_repo_path(output_dir)) / "checkpoints")
    raise ValueError("Either checkpoint_dir or output_dir must be provided")


def build_checkpoint_name(epoch, step):
    return f"epoch_{epoch + 1:04d}_step_{step:08d}"


def _latest_checkpoint_record(checkpoint_root):
    return Path(checkpoint_root) / "latest_checkpoint.json"


def _write_latest_checkpoint_record(checkpoint_root, checkpoint_dir):
    checkpoint_root = Path(checkpoint_root)
    checkpoint_dir = Path(checkpoint_dir)
    record_path = _latest_checkpoint_record(checkpoint_root)
    payload = {
        "latest_checkpoint": checkpoint_dir.name,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    tmp_path = record_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(record_path)


def _list_checkpoint_dirs(checkpoint_root):
    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.exists():
        return []
    return sorted(
        [
            path
            for path in checkpoint_root.iterdir()
            if path.is_dir() and (path / "trainer_state.pt").exists()
        ],
        key=lambda path: path.name,
    )


def prune_checkpoints(checkpoint_root, save_total_limit):
    if save_total_limit is None or save_total_limit <= 0:
        return []

    checkpoint_dirs = _list_checkpoint_dirs(checkpoint_root)
    if len(checkpoint_dirs) <= save_total_limit:
        return []

    to_remove = checkpoint_dirs[:-save_total_limit]
    removed = []
    for checkpoint_dir in to_remove:
        shutil.rmtree(checkpoint_dir, ignore_errors=False)
        removed.append(str(checkpoint_dir))
    return removed


def resolve_latest_checkpoint(checkpoint_root):
    checkpoint_root = Path(checkpoint_root)
    record_path = _latest_checkpoint_record(checkpoint_root)
    if record_path.exists():
        with open(record_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        latest_dir = checkpoint_root / payload["latest_checkpoint"]
        if latest_dir.exists():
            return str(latest_dir)

    candidates = _list_checkpoint_dirs(checkpoint_root)
    if candidates:
        return str(candidates[-1])

    legacy_state = checkpoint_root / "trainer_state.pt"
    if legacy_state.exists():
        return str(checkpoint_root)
    return None


def resolve_load_dir(lm_config, load_dir=None, from_weight=None, save_dir=None):
    if load_dir:
        return resolve_repo_path(load_dir)
    if from_weight in {None, "none"}:
        return None

    candidate = Path(resolve_repo_path(from_weight))
    if candidate.exists():
        return str(candidate)

    if save_dir is None:
        raise FileNotFoundError(f"Unable to resolve load path from {from_weight}")
    return resolve_output_dir(lm_config, save_dir=save_dir, save_weight=from_weight)


def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, "n_routed_experts", getattr(config, "num_experts", 0))
    n_active = getattr(config, "num_experts_per_tok", 0)
    n_shared = getattr(config, "n_shared_experts", 0)
    expert = (
        sum(p.numel() for n, p in model.named_parameters() if "mlp.experts.0." in n)
        / 1e6
    )
    shared_expert = (
        sum(
            p.numel()
            for n, p in model.named_parameters()
            if "mlp.shared_experts.0." in n
        )
        / 1e6
    )
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total:
        Logger(f"Model Params: {total:.2f}M-A{active:.2f}M")
    else:
        Logger(f"Model Params: {total:.2f}M")


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def format_log_message(content, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"[{timestamp}] [{level}] {content}"


def Logger(content, level="INFO"):
    if is_main_process():
        print(format_log_message(content, level=level))


def LogMetrics(title, **metrics):
    parts = [f"{key}={value}" for key, value in metrics.items()]
    Logger(f"{title}: " + ", ".join(parts))


def save_run_metadata(save_dir, lm_config, args, extra=None, wandb=None):
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
        "lm_config": (
            lm_config.to_dict() if hasattr(lm_config, "to_dict") else dict(lm_config)
        ),
        "args": vars(args) if hasattr(args, "__dict__") else dict(args),
    }
    if extra:
        payload["extra"] = extra

    latest_path = os.path.join(save_dir, "latest_run_meta.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if wandb is not None and hasattr(wandb, "config"):
        try:
            wandb.config.update(
                {"lm_config": payload["lm_config"], "args": payload["args"]},
                allow_val_change=True,
            )
        except TypeError:
            # Some trackers do not support allow_val_change
            wandb.config.update(
                {"lm_config": payload["lm_config"], "args": payload["args"]}
            )

    return latest_path


def infer_wandb_group(args=None):
    explicit_group = getattr(args, "wandb_group", None) if args is not None else None
    if explicit_group:
        return explicit_group

    script_name = Path(sys.argv[0]).stem.lower()
    if script_name.startswith("train_"):
        return script_name.removeprefix("train_")
    return script_name or None


def init_wandb_run(args, ckp_data, run_name, lm_config=None):
    if not (getattr(args, "use_wandb", False) and is_main_process()):
        return None

    import wandb

    wandb_id = ckp_data.get("wandb_id") if ckp_data else None
    group = infer_wandb_group(args)
    resume = "allow" if wandb_id else None
    run = wandb.init(
        project=getattr(args, "wandb_project", "MiniLLM-Train"),
        entity=getattr(args, "wandb_entity", "minillm"),
        group=group,
        job_type=group,
        name=run_name,
        id=wandb_id,
        resume=resume,
    )
    if lm_config is not None:
        save_run_metadata(
            getattr(args, "save_dir", str(REPO_ROOT / "out")),
            lm_config,
            args,
            wandb=run,
        )
    return run


def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def get_lr_warmup(current_step, total_steps, lr, warmup_steps):
    "warmup_steps<1, presents the rotal of warmup_steps in total steps"
    if warmup_steps < 1:
        warmup_steps = warmup_steps * total_steps
    if current_step < warmup_steps:
        return lr * (current_step / warmup_steps)
    return get_lr(current_step - warmup_steps, total_steps - warmup_steps, lr)


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lm_checkpoint(
    lm_config,
    weight="full_sft",
    model=None,
    optimizer=None,
    epoch=0,
    step=0,
    wandb=None,
    save_dir=None,
    output_dir=None,
    save_model_artifact=True,
    model_dir=None,
    save_total_limit=None,
    **kwargs,
):
    checkpoint_root = resolve_checkpoint_dir(
        output_dir=output_dir, checkpoint_dir=save_dir
    )
    os.makedirs(checkpoint_root, exist_ok=True)

    if model is not None:
        tokenizer = kwargs.pop("tokenizer", None)
        checkpoint_format = kwargs.pop("checkpoint_format", "full")
        checkpoint_name = build_checkpoint_name(epoch, step)
        checkpoint_dir = os.path.join(checkpoint_root, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        latest_model_dir = model_dir or output_dir
        if save_model_artifact:
            if latest_model_dir is None:
                raise ValueError(
                    "output_dir or model_dir is required when save_model_artifact=True"
                )
            save_model_artifacts(model, latest_model_dir, tokenizer=tokenizer)

        model_path = None
        checkpoint_model_dir = checkpoint_dir
        if checkpoint_format == "full":
            model_path = save_model_state(
                model, Path(checkpoint_dir) / "model_state.pt"
            )
        elif checkpoint_format == "lora":
            from minillm.model.model_lora import save_lora

            save_lora(model, checkpoint_dir)
            model_path = str(Path(checkpoint_dir) / "adapter_model.bin")
        elif checkpoint_format != "none":
            raise ValueError(f"Unsupported checkpoint_format={checkpoint_format}")

        wandb_id = None
        if wandb:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                wandb_id = getattr(run, "id", None) if run else None
            else:
                wandb_id = getattr(wandb, "id", None)

        resume_data = {
            "model_dir": checkpoint_model_dir,
            "model_path": model_path,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "wandb_id": wandb_id,
            "lm_config": lm_config.to_dict() if hasattr(lm_config, "to_dict") else None,
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    raw_value = (
                        value.module
                        if isinstance(value, DistributedDataParallel)
                        else value
                    )
                    raw_value = getattr(raw_value, "_orig_mod", raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        save_training_state(checkpoint_dir, **resume_data)
        _write_latest_checkpoint_record(checkpoint_root, checkpoint_dir)
        removed_checkpoints = prune_checkpoints(checkpoint_root, save_total_limit)
        for removed_checkpoint in removed_checkpoints:
            Logger(f"Pruned old checkpoint: {removed_checkpoint}")
        del resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        latest_checkpoint_dir = resolve_latest_checkpoint(checkpoint_root)
        if latest_checkpoint_dir is None:
            return None
        ckp_data = load_training_state(latest_checkpoint_dir, map_location="cpu")
        if ckp_data is not None:
            ckp_data.setdefault("checkpoint_dir", latest_checkpoint_dir)
            saved_ws = ckp_data.get("world_size", 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
                Logger(
                    f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}'
                )
            return ckp_data
        return None


def init_model(
    lm_config,
    from_weight="pretrain",
    tokenizer_path=None,
    save_dir="./out",
    load_dir=None,
    device="cuda",
    strict=False,
):
    tokenizer_path = tokenizer_path or str(PACKAGE_ROOT / "tokenizer")
    tokenizer_path = resolve_repo_path(tokenizer_path)
    resolved_load_dir = resolve_load_dir(
        lm_config,
        load_dir=load_dir,
        from_weight=from_weight,
        save_dir=save_dir,
    )

    if resolved_load_dir is not None:
        model, tokenizer = load_minillm_model(
            resolved_load_dir,
            lm_config=lm_config,
            tokenizer_path=tokenizer_path,
            device=device,
            strict=strict,
        )
        Logger(f"Loaded weights from: {resolved_load_dir}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = MiniLLMForCausalLM(lm_config).to(device)

    get_model_params(model, lm_config)
    Logger(
        f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M"
    )
    return model, tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
