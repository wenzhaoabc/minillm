import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from model.config import LMConfig


class RMSNorm(nn.Module):
    """
    均方根归一化
    计算输入张量最后一个维度的均方值，然后用输入除以这个均方根值，不减去均值，即不进行中心化
    """

    def __init__(self, dim: int, epsilon: float):
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, dim]
        r_rms = torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True))  # 1/sqrt(mean(x^2))
        res = x.float() * (r_rms + self.epsilon) * self.weight
        return res.type_as(x)


def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """
    预计算旋转位置编码
    dim: 位置编码的维度
    end: 位置编码的长度
    theta: 旋转因子
    """
    # 生成旋转位置编码
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    # [end, dim//2]
    t = torch.arange(end, device=freqs.device)
    # [end, dim//2]
    freqs = torch.outer(t, freqs).float()
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)
    return pos_cis


def apply_rotary_pos_emb(xq: torch.Tensor, xk: torch.Tensor, pos_cis: torch.Tensor):
    """
    应用旋转位置编码
    poc_cis: [seq_len, dim//2],旋转位置编码,旋转因子
    """

    def unite_shape(pos_cis: torch.Tensor, x: torch.Tensor):
        "将pos_cis的形状调整为x的形状,便于广播操作"
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    # xq: [batch_size, seq_len, n_heads, head_dim]
    # xq_: [batch_size, seq_len, n_heads,head_dim//2]
    # 将最后一维的维度调整为2，每两列作为一个复数的实部和虚部
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # pos_cis: [seq_len, dim//2]
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_repeat: int) -> torch.Tensor:
    """
    将x的head维度重复n_repeat次
    """
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_repeat == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_repeat, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_repeat, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        assert config.n_heads % self.n_kv_heads == 0

        self.n_local_heads = config.n_heads  # 本地头的数量
        self.n_local_kv_heads = self.n_kv_heads  # 本地kv头的数量
        self.n_repeat = self.n_local_heads // self.n_local_kv_heads  # 每个kv头重复的次数
        self.head_dim = config.dim // config.n_heads  # 每个头的维度
        # [in_feature,out_feature] : [dim, dim]
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)  # Attention dropout
        self.resid_dropout = nn.Dropout(config.dropout)  # Residual dropout
        self.dropout = config.dropout
        self.flash = (
            hasattr(nn.functional, "scaled_dot_product_attention") and config.flash_attn
        )  # 是否使用Flash Attention
        # 上三角掩码矩阵，对角线以上元素为-inf,对角线及以下元素为0
        mask = torch.full((1, 1, config.max_seq_len, config.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        # 注册一个缓冲区，不会被优化器更新, persistent=False表示不会被保存到checkpoint中
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        pos_cis: torch.Tensor,
        past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        batch_size, seq_len, dim = x.size()
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq: torch.Tensor = xq.view(batch_size, seq_len, self.n_local_heads, self.head_dim)
        xk: torch.Tensor = xk.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
        xv: torch.Tensor = xv.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_pos_emb(xq, xk, pos_cis)
        # KV Cache
        if past_key_values is not None:
            xk = torch.cat([past_key_values[0], xk], dim=1)
            xv = torch.cat([past_key_values[1], xv], dim=1)
        past_key_values = (xk, xv) if use_cache else None

        # GQA
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_repeat).transpose(1, 2)  # [batch_size, n_kv_heads, seq_len, head_dim]
        xv = repeat_kv(xv, self.n_repeat).transpose(1, 2)  # [batch_size, n_kv_heads, seq_len, head_dim]

        # flash attention
        if self.flash and seq_len > 1:
            dropout_p = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=dropout_p, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores += self.mask[:, :, :seq_len, :seq_len]
            scores = F.softmax(scores.float(), dim=-1).type_as(x)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # [batch_size, seq_len, dim]
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_key_values


class FeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        if config.hidden_dim is None:
            # 隐藏层维度是输入维度的4倍，经验值，然后取2/3的整数倍，减小计算量
            hidden_dim = config.dim * 4
            hidden_dim = int(2 * hidden_dim / 3)
            # 将隐藏层维度调整为multiple_of的倍数，满足某些硬件的对齐要求
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 门控机制
        y = F.silu(self.w1(x)) * self.w3(x)
        return self.dropout(self.w2(y))


class MoEGate(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.top_k = config.n_experts_per_token
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.dim
        # weights: [n_routed_experts, gating_dim]
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化权重
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, dim)
        logits = F.linear(hidden_states, self.weight)  # [batch_size * seq_len, n_routed_experts]
        if self.scoring_func == "softmax":
            scores = F.softmax(logits, dim=-1)  # [batch_size * seq_len, n_routed_experts]
        else:
            raise NotImplementedError(f"scoring_func {self.scoring_func} not implemented")

        # 选择top_k个专家
        # topk_weights: [batch_size * seq_len, top_k]
        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1, sorted=False)

        if self.top_k and self.norm_topk_prob:
            denominator = torch.sum(topk_weights, dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator  # 归一化

        if self.training and self.alpha > 0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # 每个token被分到的前top_k个expert的索引
            topk_idx_for_aux_loss = topk_indices.view(batch_size, -1)  # [batch_size, top_k * seq_len]

            # 计算每个exprt的分数和归一化选择次数，衡量expert使用的均匀性和分配合理性
            if self.seq_aux:
                # [batch_size, seq_len, n_routed_experts] 每个token对所有expert的分数
                scores_for_seq_aux = scores_for_aux.view(batch_size, seq_len, -1)
                ce = torch.zeros(batch_size, self.n_routed_experts, device=hidden_states.device)
                # 将每个 token 被分配到的专家的计数累加到 ce 中
                ce.scatter_add_(
                    1, topk_idx_for_aux_loss, torch.ones(batch_size, seq_len * aux_topk, device=hidden_states.device)
                )
                # 计算每个专家的归一化选择次数
                ce.div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # 平均每个专家被选择的次数
                ce = mask_ce.float().mean(0)  # [n_routed_experts]
                # 平均每个expert的分数
                pi = scores_for_aux.mean(0)  # [n_routed_experts]
                # 每个专家的归一化分配次数
                fi = ce * self.n_routed_experts
                aux_loss = (pi * fi).sum() * self.alpha
        else:
            aux_loss = 0.0

        return topk_indices, topk_weights, aux_loss


class MoEFeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_routed_experts)])
        self.gate = MoEGate(config)
        if self.config.shared_experts is not None:
            self.shared_expert = FeedForward(config)

    def forward(self, x: torch.Tensor):
        identity = x  # 残差连接
        original_shape = x.shape
        batch_size, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_indices, topk_weights, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])  # [batch_size * seq_len, dim]
        flat_topk_indices = topk_indices.view(-1)  # [batch_size * seq_len * top_k]
        if self.training:
            x = x.repeat_interleave(
                self.config.n_experts_per_token, dim=0
            )  # [batch_size * seq_len * n_routed_experts, dim]
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_indices == i] = expert(x[flat_topk_indices == i]).to(y.dtype)
            # topk_weights: [batch_size * seq_len, top_k]
            # y.view(*topk_weights.shape, -1) : [batch_size * seq_len, n_routed_experts, dim]
            y = (y.view(*topk_weights.shape, -1) * topk_weights.unsqueeze(-1)).sum(dim=1)
            # original_shape: [batch_size, seq_len, dim]
            y = y.view(*original_shape)
        else:
            # 推理阶段，使用最优的专家
            y = self.moe_inference(x, flat_topk_indices, topk_weights.view(-1, 1)).view(*original_shape)

        if self.config.shared_experts is not None:
            y = y + self.shared_expert(identity)

        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_inference(self, x: torch.Tensor, flat_expert_indices: torch.Tensor, flat_expert_weights: torch.Tensor):
        # 仅用于推理阶段
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()  # 排序后的索引
        # 累积和，前n个expert处理了多少个token
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 映射回token的索引，每个token的对应的expert索引
        token_idxs = idxs // self.config.n_experts_per_token
        for i, end_inx in enumerate(tokens_per_expert):
            start_inx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_inx == end_inx:
                continue
            # 选择当前expert处理的token
            expert = self.experts[i]
            expert_token_idxs = token_idxs[start_inx:end_inx]
            expert_tokens = x[expert_token_idxs]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 将当前expert处理的token的输出加到缓存中
            # mul_ token输出乘以对应的权重
            expert_out.mul_(flat_expert_weights[idxs[start_inx:end_inx]])
            # scatter_add_ 将当前expert处理的token的输出拼接到expert_cache中
            expert_cache.scatter_add_(0, expert_token_idxs.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class TransformerLayer(nn.Module):
    def __init__(self, layer_id: int, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ff = FeedForward(config) if not config.use_moe else MoEFeedForward(config)
        self.ff_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        pos_cis: torch.Tensor,
        past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        # Attention
        h_attn, past_kv = self.attention(
            self.attention_norm(x), pos_cis, past_key_values=past_key_values, use_cache=use_cache
        )
        # Residual connection
        h = x + h_attn
        # Feed Forward
        h_ff = self.ff(self.ff_norm(h))
        # Residual connection
        out = h + h_ff
        return out, past_kv


class MiniLLM(PreTrainedModel):
    config_class = LMConfig

    def __init__(self, params: LMConfig | None = None):
        self.params = params or LMConfig()
        super().__init__(self.params)
        self.vocab_size = self.params.vocab_size
        self.n_layers = self.params.n_layers
        self.tok_emb = nn.Embedding(self.vocab_size, self.params.dim)
        self.dropout = nn.Dropout(self.params.dropout)
        self.layers = nn.ModuleList([TransformerLayer(i, self.params) for i in range(self.n_layers)])
        self.norm = RMSNorm(self.params.dim, self.params.norm_eps)
        self.output = nn.Linear(self.params.dim, self.vocab_size, bias=False)
        self.tok_emb.weight = self.output.weight  # 共享参数

        # 预计算的旋转位置编码
        self.pos_cis: torch.Tensor
        self.register_buffer(
            "pos_cis",
            precompute_pos_cis(self.params.dim // self.params.n_heads, theta=self.params.rope_theta),
            persistent=False,
        )
        self.OUT = CausalLMOutputWithPast()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        layers_past_kvs: list | None = None,
        use_cache: bool = False,
        **kwargs,
    ):
        # input_ids: [batch_size, seq_len]
        layers_past_kvs = layers_past_kvs or [None] * self.n_layers
        start_pos = kwargs.get("start_pos", 0)
        h: torch.Tensor = self.dropout(self.tok_emb(input_ids))
        pos_cis = self.pos_cis[start_pos : start_pos + h.size(1)]

        past_kvs = []
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(h, pos_cis, past_key_values=layers_past_kvs[l], use_cache=use_cache)
            past_kvs.append(past_kv)

        # [batch_size, seq_len, vocab_size] 模型预测的词汇表概率分布
        logits = self.output(self.norm(h))
        aux_loss: float = sum(l.ff.aux_loss for l in self.layers if isinstance(l.ff, MoEFeedForward))
        self.OUT.__setitem__("logits", logits)  # 预测的词汇表概率分布
        self.OUT.__setitem__("aux_loss", aux_loss)
        self.OUT.__setitem__("past_key_values", past_kvs)
        return self.OUT

    @torch.inference_mode()
    def my_generate(
        self,
        input_ids: torch.Tensor,
        eos_token_id: int = 2,
        max_new_tokens: int = 100,
        temperature: float = 0.75,
        top_p: float = 0.90,
        stream: bool = False,
        rp: float = 1.0,  # Repetition Penalty 重复惩罚，降低模型生成重复 token 的概率
        use_cache: bool = False,
        pad_token_id: int = 0,
        **kwargs,
    ):
        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **kwargs)

        generated = []

        for i in range(input_ids.size(0)):
            # non_pad: [1, seq_len_filtered]
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            out = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **kwargs)
            tokens_list = [tokens[:, -1:] for tokens in out]
            # 连接所有的tokens，得到一个完整的序列
            gen = torch.cat(tokens_list, dim=1) if tokens_list else non_pad
            full_seq = torch.cat([non_pad, gen], dim=1)
            generated.append(full_seq)
        max_len = max([g.size(1) for g in generated])
        generated = [
            torch.cat([g, torch.full((1, max_len - g.size(1)), pad_token_id, dtype=g.dtype, device=g.device)], dim=-1)
            for g in generated
        ]
        return torch.cat(generated, dim=0)

    def _stream(
        self,
        input_ids: torch.Tensor,
        eos_token_id: int = 2,
        max_new_tokens: int = 100,
        temperature: float = 0.75,
        top_p: float = 0.90,
        rp: float = 1.0,
        use_cache: bool = False,
        **kwargs,
    ):
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        while input_ids.shape[1] < max_new_tokens - 1:
            if first_seq or not use_cache:
                out = self(input_ids, layers_past_kvs=past_kvs, use_cache=use_cache, **kwargs)
                first_seq = False
            else:
                out = self(
                    input_ids[:, -1:],
                    layers_past_kvs=past_kvs,
                    use_cache=use_cache,
                    start_pos=input_ids.shape[1] - 1,
                    **kwargs,
                )

            # 取序列中最后一个词的概率分布
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
            # 降低已生成的token的分数，使模型倾向于生成新的token
            # 生成任务，batch_size=1
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            # temperature 控制生成文本的多样性，随机性，调整模型输出的概率分布
            logits /= temperature + 1e-9
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
                # 归一化概率分布
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                # 累积概率分布
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # 选择累加概率大于top_p的token
                sorted_indices_to_remove: torch.Tensor = cumulative_probs > top_p
                # 右移一位，确保概率最大的第一个token不会被移除
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float("Inf")
            # 从概率分布中采样一个token
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, input_ids_next), dim=-1)
            yield input_ids[:, start:]
            if input_ids_next.item() == eos_token_id:
                break
