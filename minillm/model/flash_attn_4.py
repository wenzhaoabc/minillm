import torch

try:
    from flash_attn.cute import flash_attn_func as flash_attn_4_func
except Exception:
    flash_attn_4_func = None


def is_flash_attn_4_available() -> bool:
    if flash_attn_4_func is None or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major in {9, 10, 11}


def _is_all_ones_attention_mask(attention_mask: torch.Tensor | None) -> bool:
    if attention_mask is None:
        return True
    return bool(torch.all(attention_mask == 1).item())


def can_use_flash_attn_4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    training: bool = False,
    dropout_p: float = 0.0,
) -> bool:
    if not is_flash_attn_4_available():
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    if q.dtype != k.dtype or q.dtype != v.dtype:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        return False
    if training and dropout_p > 0.0:
        return False
    return _is_all_ones_attention_mask(attention_mask)


def flash_attn_4_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    training: bool = False,
    dropout_p: float = 0.0,
    causal: bool = True,
) -> torch.Tensor:
    if not can_use_flash_attn_4(
        q,
        k,
        v,
        attention_mask=attention_mask,
        training=training,
        dropout_p=dropout_p,
    ):
        raise RuntimeError("FlashAttention-4 is not available for the current inputs")

    q_seq_first = q.transpose(1, 2).contiguous()
    k_seq_first = k.transpose(1, 2).contiguous()
    v_seq_first = v.transpose(1, 2).contiguous()
    output = flash_attn_4_func(
        q_seq_first,
        k_seq_first,
        v_seq_first,
        causal=causal,
    )
    return output.transpose(1, 2)