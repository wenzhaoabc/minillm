"""
Reward model for RLHF (Reinforcement Learning from Human Feedback).

reference:
1. RLHF Reward Model Training: https://medium.com/towards-generative-ai/reward-model-training-2209d1befb5f
2.如何正确复现 Instruct GPT / RLHF? : https://zhuanlan.zhihu.com/p/622134699
3. Rewaed Model:  https://www.cnblogs.com/LittleHann/p/17457372.html
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


from minillm.model.config import MiniLLMConfig
from minillm.model.model_base import MiniLLMModel as BaseModel


class RewardModel(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.config = config
        self.model = BaseModel(config)
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        nn.init.normal_(self.v_head.weight, mean=0.0, std=0.01)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ):
        # h: [batch_size, seq_len, hidden_size]
        h, _, aux_loss = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        assert isinstance(h, torch.Tensor)
        # Get the last token hidden state for each sequence
        if attention_mask is None:
            # if no attention mask is provided, each sequence has the full length
            last_token_indices = input_ids.shape[1] - 1
            # Expand the last token indices to match the batch size
            if isinstance(last_token_indices, int) and h.shape[0] > 1:
                last_token_indices = torch.tensor([last_token_indices] * h.shape[0], device=h.device)
            elif isinstance(last_token_indices, int):
                last_token_indices = torch.tensor([last_token_indices], device=h.device)
        else:
            # if attention mask is provided, find the last token for each sequence
            sequence_lengths = attention_mask.sum(dim=1)
            last_token_indices = sequence_lengths - 1

        batch_indices = torch.arange(h.shape[0], device=h.device)
        last_token_hidden_states = h[batch_indices, last_token_indices]

        # Compute the reward score
        reward_scores = self.v_head(last_token_hidden_states).squeeze(-1)  # [batch_size]
        return reward_scores, aux_loss
