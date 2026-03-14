import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional dependency at import time
    triton = None
    tl = None


SUPPORTED_HEAD_DIMS = {16, 32, 64, 128}


def is_triton_flash_attention_available() -> bool:
    return triton is not None and tl is not None and torch.cuda.is_available()


def build_causal_mask(seq_q: int, seq_k: int, device: torch.device | str) -> torch.Tensor:
    q_idx = torch.arange(seq_q, device=device)[:, None]
    k_idx = torch.arange(seq_k, device=device)[None, :]
    return k_idx <= (q_idx + seq_k - seq_q)


def build_additive_attention_mask(
    attention_mask: torch.Tensor | None,
    seq_q: int,
    seq_k: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    causal = build_causal_mask(seq_q, seq_k, device=device)
    if attention_mask is None:
        keep = causal.view(1, 1, seq_q, seq_k)
        batch_size = 1
        mask_device = keep.device
    else:
        keep = causal.view(1, 1, seq_q, seq_k) & attention_mask[:, None, None, :].bool()
        batch_size = attention_mask.shape[0]
        mask_device = attention_mask.device
    mask = torch.zeros((batch_size, 1, seq_q, seq_k), device=mask_device, dtype=dtype)
    return mask.masked_fill(~keep, float("-inf"))


def _is_all_ones_attention_mask(attention_mask: torch.Tensor | None) -> bool:
    if attention_mask is None:
        return True
    return bool(torch.all(attention_mask == 1).item())


def can_use_triton_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    training: bool,
) -> bool:
    if training or not is_triton_flash_attention_available():
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    if q.dtype != k.dtype or q.dtype != v.dtype:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if q.shape[-1] not in SUPPORTED_HEAD_DIMS:
        return False
    return _is_all_ones_attention_mask(attention_mask)


if triton is not None:

    @triton.jit
    def flash_attn_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        stride_qb,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_om,
        stride_od,
        q_len,
        k_len,
        scale,
        head_dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_b = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, head_dim)

        q_ptrs = q_ptr + pid_b * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q_mask = (offs_m[:, None] < q_len) & (offs_d[None, :] < head_dim)
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, head_dim), tl.float32)
        causal_shift = k_len - q_len

        for start_n in range(0, k_len, BLOCK_N):
            current_n = start_n + offs_n
            k_ptrs = k_ptr + pid_b * stride_kb + current_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = v_ptr + pid_b * stride_vb + current_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

            kv_mask = (current_n[:, None] < k_len) & (offs_d[None, :] < head_dim)
            k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

            qk = tl.dot(q, tl.trans(k)) * scale
            causal = current_n[None, :] <= (offs_m[:, None] + causal_shift)
            valid = (offs_m[:, None] < q_len) & (current_n[None, :] < k_len) & causal
            qk = tl.where(valid, qk, -float("inf"))

            m_ij = tl.where(offs_m < q_len, tl.max(qk, axis=1), 0.0)
            p = tl.where(valid, tl.exp(qk - m_ij[:, None]), 0.0)
            l_ij = tl.sum(p, axis=1)

            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v) * beta[:, None]
            l_i = l_i * alpha + l_ij * beta
            m_i = m_new

        out = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
        o_ptrs = o_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        o_mask = (offs_m[:, None] < q_len) & (offs_d[None, :] < head_dim)
        tl.store(o_ptrs, out.to(q.dtype), mask=o_mask)


def triton_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    if not is_triton_flash_attention_available():
        raise RuntimeError("Triton flash attention is not available")

    batch, heads, q_len, head_dim = q.shape
    _, _, k_len, _ = k.shape
    q_flat = q.contiguous().view(batch * heads, q_len, head_dim)
    k_flat = k.contiguous().view(batch * heads, k_len, head_dim)
    v_flat = v.contiguous().view(batch * heads, k_len, head_dim)
    out_flat = torch.empty_like(q_flat)

    num_warps = 4 if head_dim <= 64 else 8
    grid = (triton.cdiv(q_len, block_m), batch * heads)
    flash_attn_fwd_kernel[grid](
        q_flat,
        k_flat,
        v_flat,
        out_flat,
        q_flat.stride(0),
        q_flat.stride(1),
        q_flat.stride(2),
        k_flat.stride(0),
        k_flat.stride(1),
        k_flat.stride(2),
        v_flat.stride(0),
        v_flat.stride(1),
        v_flat.stride(2),
        out_flat.stride(0),
        out_flat.stride(1),
        out_flat.stride(2),
        q_len,
        k_len,
        head_dim ** -0.5,
        head_dim=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=2,
    )
    return out_flat.view(batch, heads, q_len, head_dim)


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    training: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    if can_use_triton_flash_attention(q, k, v, attention_mask=attn_mask, training=training):
        return triton_flash_attention(q, k, v)

    seq_q, seq_k = q.shape[-2], k.shape[-2]
    additive_mask = build_additive_attention_mask(attn_mask, seq_q, seq_k, q.dtype, q.device)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=additive_mask,
        dropout_p=dropout_p if training else 0.0,
        is_causal=False,
    )