from pathlib import Path
from typing import Optional

import torch
from torch.nn.parallel import DistributedDataParallel
from safetensors.torch import load_file as safe_load_file
from transformers import AutoTokenizer

from .config import MiniLLMConfig
from .model_base import MiniLLMForCausalLM


def ensure_auto_class_registration():
    MiniLLMConfig.register_for_auto_class()
    MiniLLMForCausalLM.register_for_auto_class("AutoModelForCausalLM")


def unwrap_model(model):
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(raw_model, "_orig_mod", raw_model)


def normalize_state_dict_keys(state_dict):
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def _convert_state_dict_dtype(state_dict, target_dtype: Optional[torch.dtype]):
    if target_dtype is None:
        return {key: value.detach().cpu() for key, value in state_dict.items()}

    converted = {}
    for key, value in state_dict.items():
        tensor = value.detach().cpu()
        if torch.is_floating_point(tensor):
            tensor = tensor.to(dtype=target_dtype)
        converted[key] = tensor
    return converted


def save_model_artifacts(
    model,
    output_dir,
    tokenizer=None,
    save_dtype: Optional[torch.dtype] = torch.float16,
    safe_serialization: bool = False,
):
    ensure_auto_class_registration()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_model = unwrap_model(model)
    state_dict = _convert_state_dict_dtype(raw_model.state_dict(), save_dtype)
    raw_model.save_pretrained(
        str(output_path),
        state_dict=state_dict,
        safe_serialization=safe_serialization,
    )
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_path))
    return str(output_path)


def save_model_state(model, output_path, save_dtype: Optional[torch.dtype] = torch.float16):
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = _convert_state_dict_dtype(unwrap_model(model).state_dict(), save_dtype)
    torch.save(state_dict, target_path)
    return str(target_path)


def load_model_state(output_path, map_location="cpu"):
    state_dict = torch.load(Path(output_path), map_location=map_location)
    return normalize_state_dict_keys(state_dict)


def save_training_state(output_dir, **payload):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    state_path = output_path / "trainer_state.pt"
    tmp_path = output_path / "trainer_state.pt.tmp"
    torch.save(payload, tmp_path)
    tmp_path.replace(state_path)
    return str(state_path)


def load_training_state(output_dir, map_location="cpu"):
    state_path = Path(output_dir) / "trainer_state.pt"
    if not state_path.exists():
        return None
    return torch.load(state_path, map_location=map_location)


def load_minillm_model(
    load_path,
    lm_config=None,
    tokenizer_path=None,
    device="cpu",
    strict=True,
):
    resolved = Path(load_path).expanduser()
    tokenizer_source = resolved if resolved.is_dir() else Path(tokenizer_path).expanduser()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), trust_remote_code=True)

    if resolved.is_dir():
        if lm_config is None:
            lm_config = MiniLLMConfig.from_pretrained(str(resolved))
        model = MiniLLMForCausalLM(lm_config)

        safe_weights = resolved / "model.safetensors"
        torch_weights = resolved / "pytorch_model.bin"
        if safe_weights.exists():
            state_dict = safe_load_file(str(safe_weights), device="cpu")
        elif torch_weights.exists():
            state_dict = torch.load(str(torch_weights), map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No supported weight file found in {resolved}, expected model.safetensors or pytorch_model.bin"
            )
        model.load_state_dict(normalize_state_dict_keys(state_dict), strict=strict)
    else:
        if lm_config is None:
            raise ValueError("lm_config is required when loading raw .pth weights")
        state_dict = torch.load(str(resolved), map_location=device)
        model = MiniLLMForCausalLM(lm_config)
        model.load_state_dict(normalize_state_dict_keys(state_dict), strict=strict)

    return model.to(device), tokenizer