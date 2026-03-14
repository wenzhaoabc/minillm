import os
import random
from contextlib import nullcontext
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer  # type: ignore
import torch.distributed as dist

from minillm.model.model_base import MiniLLMModel


def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, "n_routed_experts", getattr(config, "num_experts", 0))
    n_active = getattr(config, "num_experts_per_tok", 0)
    n_shared = getattr(config, "n_shared_experts", 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if "mlp.experts.0." in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if "mlp.shared_experts.0." in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total:
        print(f"Model Params: {total:.2f}M-A{active:.2f}M")
    else:
        print(f"Model Params: {total:.2f}M")


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


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
    save_dir="../checkpoints",
    **kwargs,
):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = "_moe" if lm_config.use_moe else ""
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth"
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + ".tmp"
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                wandb_id = getattr(run, "id", None) if run else None
            else:
                wandb_id = getattr(wandb, "id", None)

        resume_data = {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "wandb_id": wandb_id,
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, "_orig_mod", raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location="cpu")
            saved_ws = ckp_data.get("world_size", 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
                print(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


DEFAULT_TOKENIZER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tokenizer"))


def init_model(lm_config, from_weight="pretrain", tokenizer_path=DEFAULT_TOKENIZER_PATH, save_dir="../out", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniLLMModel(lm_config)

    if from_weight != "none":
        moe_suffix = "_moe" if lm_config.use_moe else ""
        weight_path = f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    print(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M")
    return model.to(device), tokenizer


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


def validate_training_paths(*, data_path, tokenizer_path, out_dir, model_path=None, resume_from=None):
    missing_paths = []
    for name, path in {
        "data_path": data_path,
        "tokenizer_path": tokenizer_path,
        "model_path": model_path,
        "resume_from": resume_from,
    }.items():
        if path and not os.path.exists(path):
            missing_paths.append(f"{name}={path}")

    if missing_paths:
        raise FileNotFoundError("Missing required path(s): " + ", ".join(missing_paths))

    os.makedirs(out_dir, exist_ok=True)


def get_autocast_context(device: str, dtype: str):
    if device == "cpu":
        return nullcontext()

    amp_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if dtype not in amp_dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype}'. Expected one of: {', '.join(amp_dtype_map)}")

    device_type = "cuda" if str(device).startswith("cuda") else str(device)
    return torch.autocast(device_type=device_type, dtype=amp_dtype_map[dtype])


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_training_checkpoint(
    *,
    model,
    optimizer,
    scaler,
    epoch,
    step,
    out_dir,
    checkpoint_name,
    resume_name="latest_resume.pt",
    extra_state=None,
):
    os.makedirs(out_dir, exist_ok=True)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    state_dict = raw_model.state_dict()
    resume_model_state = {k: v.detach().cpu() for k, v in state_dict.items()}
    weights_to_save = {k: v.detach().half().cpu() for k, v in state_dict.items()}

    checkpoint_path = os.path.join(out_dir, checkpoint_name)
    torch.save(weights_to_save, checkpoint_path)

    resume_state = {
        "model": resume_model_state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        resume_state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    if extra_state:
        resume_state.update(extra_state)

    resume_path = os.path.join(out_dir, resume_name)
    torch.save(resume_state, resume_path)
    return checkpoint_path, resume_path


def load_training_checkpoint(*, model, optimizer, scaler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
        if optimizer is not None and checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            move_optimizer_state_to_device(optimizer, device)
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])

        if checkpoint.get("python_random_state") is not None:
            random.setstate(checkpoint["python_random_state"])
        if checkpoint.get("numpy_random_state") is not None:
            np.random.set_state(checkpoint["numpy_random_state"])
        if checkpoint.get("torch_random_state") is not None:
            torch.set_rng_state(checkpoint["torch_random_state"])
        if torch.cuda.is_available() and checkpoint.get("torch_cuda_random_state_all") is not None:
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_random_state_all"])

        return checkpoint.get("epoch", 0), checkpoint.get("step", -1)

    model.load_state_dict(checkpoint, strict=False)
    return 0, -1


def load_model_state_dict(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint
