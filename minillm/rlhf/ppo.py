"""
PPO (Proximal Policy Optimization) implementation for reinforcement learning from human feedback (RLHF).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import copy
import os
import json
import time
from typing import Dict, List, Tuple
import numpy as np


class PPODataset(Dataset):
    def __init__(self, data_path, tokenizer, max_prompt_length=512):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.data = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get("prompt", item.get("instruction", ""))

        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(
            prompt, add_special_tokens=True, max_length=self.max_prompt_length, truncation=True
        )
        prompt_ids = torch.tensor(prompt_tokens, dtype=torch.long)

        return {"prompt_ids": prompt_ids, "prompt_text": prompt}


def collate_fn(batch):
    prompts_ids = [item["prompt_ids"] for item in batch]
    prompts_text = [item["prompt_text"] for item in batch]

    # 动态padding到批次内最大长度
    max_len = max(len(p) for p in prompts_ids)

    padded_prompts = []
    attention_masks = []

    for prompt_ids in prompts_ids:
        padding_length = max_len - len(prompt_ids)
        padded_prompt = torch.cat([prompt_ids, torch.zeros(padding_length, dtype=torch.long)])
        attention_mask = torch.cat([torch.ones(len(prompt_ids)), torch.zeros(padding_length)])

        padded_prompts.append(padded_prompt)
        attention_masks.append(attention_mask)

    return {
        "prompt_ids": torch.stack(padded_prompts),
        "attention_masks": torch.stack(attention_masks),
        "prompt_texts": prompts_text,
    }


class SharedBackboneModel(nn.Module):
    """共享骨干网络的优化模型"""

    def __init__(self, base_model):
        super().__init__()
        self.shared_backbone = base_model  # 共享的backbone

        # 不同的输出头
        hidden_size = base_model.config.hidden_size if hasattr(base_model, "config") else base_model.hidden_size

        self.lm_head = base_model.lm_head  # LLM head
        self.value_head = nn.Linear(hidden_size, 1)  # 价值头
        self.reward_head = nn.Linear(hidden_size, 1)  # 奖励头

        # 初始化新的头
        nn.init.normal_(self.value_head.weight, std=0.01)
        nn.init.zeros_(self.value_head.bias)
        nn.init.normal_(self.reward_head.weight, std=0.01)
        nn.init.zeros_(self.reward_head.bias)

    def forward(self, input_ids, attention_mask=None, output_type="lm"):
        """
        output_type: 'lm', 'value', 'reward'
        """
        outputs = self.shared_backbone(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if output_type == "lm":
            return self.lm_head(hidden_states)
        elif output_type == "value":
            last_hidden = hidden_states[:, -1, :]
            return self.value_head(last_hidden).squeeze(-1)
        elif output_type == "reward":
            last_hidden = hidden_states[:, -1, :]
            return self.reward_head(last_hidden).squeeze(-1)
        else:
            raise ValueError(f"Unknown output_type: {output_type}")


class OptimizedPolicyModel(nn.Module):
    """优化后的策略模型"""

    def __init__(self, shared_model):
        super().__init__()
        self.shared_model = shared_model

    def forward(self, input_ids, attention_mask=None):
        return self.shared_model(input_ids, attention_mask, output_type="lm")

    def generate(
        self, input_ids, attention_mask=None, max_new_tokens=50, do_sample=True, temperature=1.0, pad_token_id=0
    ):
        """生成文本序列"""
        self.eval()
        with torch.no_grad():
            batch_size = input_ids.shape[0]
            generated_ids = input_ids.clone()

            for _ in range(max_new_tokens):
                logits = self.forward(generated_ids, attention_mask)
                next_token_logits = logits[:, -1, :] / temperature

                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                # 更新attention_mask
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((batch_size, 1), device=attention_mask.device)], dim=-1
                    )

                if (next_token == pad_token_id).all():
                    break

        return generated_ids, attention_mask


class OptimizedValueModel(nn.Module):

    def __init__(self, shared_model):
        super().__init__()
        self.shared_model = shared_model

    def forward(self, input_ids, attention_mask=None):
        return self.shared_model(input_ids, attention_mask, output_type="value")


class OptimizedRewardModel(nn.Module):

    def __init__(self, shared_model):
        super().__init__()
        self.shared_model = shared_model

    def forward(self, input_ids, attention_mask=None):
        return self.shared_model(input_ids, attention_mask, output_type="reward")


class PPOTrainer:
    """PPO训练器"""

    def __init__(
        self,
        policy_model,
        value_model,
        reference_model,
        reward_model,
        policy_optimizer,
        value_optimizer,
        args,
        writer=None,
    ):
        self.policy_model = policy_model
        self.value_model = value_model
        self.reference_model = reference_model  # 冻结
        self.reward_model = reward_model  # 冻结

        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.args = args
        self.writer = writer
        self.global_step = 0

        # 冻结参考模型和奖励模型
        for param in self.reference_model.parameters():
            param.requires_grad = False
        for param in self.reward_model.parameters():
            param.requires_grad = False

    def generate_episode(self, prompt_ids, attention_mask=None):
        """生成一个episode"""
        batch_size = prompt_ids.shape[0]
        device = prompt_ids.device
        max_new_tokens = self.args.max_new_tokens

        # 生成文本和收集信息
        generated_ids, final_attention_mask = self.policy_model.generate(
            prompt_ids,
            attention_mask,
            max_new_tokens,
            do_sample=True,
            temperature=self.args.temperature,
            pad_token_id=self.args.pad_token_id,
        )

        # 计算每个位置的log概率和价值
        self.policy_model.train()
        self.value_model.train()

        prompt_len = prompt_ids.shape[1]
        response_len = generated_ids.shape[1] - prompt_len

        if response_len <= 0:
            return None  # 没有生成新的token

        # 计算log概率
        with torch.no_grad():
            policy_logits = self.policy_model(generated_ids, final_attention_mask)
            response_logits = policy_logits[:, prompt_len - 1 : -1, :]  # 新生成部分的logits
            response_tokens = generated_ids[:, prompt_len:]  # 新生成的tokens

            log_probs = F.log_softmax(response_logits, dim=-1)
            selected_log_probs = torch.gather(log_probs, -1, response_tokens.unsqueeze(-1)).squeeze(-1)

        # 计算价值
        values_list = []
        for i in range(response_len):
            seq_so_far = generated_ids[:, : prompt_len + i + 1]
            mask_so_far = final_attention_mask[:, : prompt_len + i + 1] if final_attention_mask is not None else None
            with torch.no_grad():
                value = self.value_model(seq_so_far, mask_so_far)
                values_list.append(value)

        values = torch.stack(values_list, dim=1)  # [batch_size, response_len]

        # 计算奖励（只在最后一步）
        with torch.no_grad():
            episode_reward = self.reward_model(generated_ids, final_attention_mask)
            kl_penalty = self.compute_kl_penalty(prompt_ids, generated_ids, final_attention_mask)
            final_reward = episode_reward - kl_penalty

        # 计算优势和returns
        advantages, returns = self.compute_advantages_sparse_reward(final_reward, values)

        return {
            "prompt_ids": prompt_ids,
            "generated_ids": generated_ids,
            "attention_mask": final_attention_mask,
            "log_probs": selected_log_probs,
            "values": values,
            "advantages": advantages,
            "returns": returns,
            "episode_reward": episode_reward,
            "kl_penalty": kl_penalty,
            "final_reward": final_reward,
        }

    def compute_advantages_sparse_reward(self, episode_reward, values):
        """计算稀疏奖励的优势函数"""
        batch_size, seq_len = values.shape
        device = values.device

        # 构建稀疏奖励（只在最后一步）
        rewards = torch.zeros(batch_size, seq_len, device=device)
        rewards[:, -1] = episode_reward

        # 计算returns和advantages
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # 最后一步
        returns[:, -1] = rewards[:, -1]
        advantages[:, -1] = returns[:, -1] - values[:, -1]

        # 从后往前计算GAE
        for t in reversed(range(seq_len - 1)):
            td_error = rewards[:, t] + self.args.gamma * values[:, t + 1] - values[:, t]
            advantages[:, t] = td_error + self.args.gamma * self.args.gae_lambda * advantages[:, t + 1]
            returns[:, t] = rewards[:, t] + self.args.gamma * returns[:, t + 1]

        return advantages, returns

    def compute_kl_penalty(self, prompt_ids, generated_ids, attention_mask=None):
        """计算KL散度惩罚"""
        prompt_len = prompt_ids.shape[1]
        response_len = generated_ids.shape[1] - prompt_len

        if response_len <= 0:
            return torch.zeros(generated_ids.shape[0], device=generated_ids.device)

        # Policy和Reference模型的logits
        policy_logits = self.policy_model(generated_ids, attention_mask)
        with torch.no_grad():
            ref_logits = self.reference_model(generated_ids, attention_mask)

        # 只计算新生成部分的KL
        policy_response_logits = policy_logits[:, prompt_len - 1 : -1, :]
        ref_response_logits = ref_logits[:, prompt_len - 1 : -1, :]
        response_tokens = generated_ids[:, prompt_len:]

        # 计算log概率
        policy_log_probs = F.log_softmax(policy_response_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_response_logits, dim=-1)

        policy_selected = torch.gather(policy_log_probs, -1, response_tokens.unsqueeze(-1)).squeeze(-1)
        ref_selected = torch.gather(ref_log_probs, -1, response_tokens.unsqueeze(-1)).squeeze(-1)

        # KL散度
        kl_div = (policy_selected - ref_selected).sum(dim=-1)
        return kl_div * self.args.kl_coeff

    def ppo_update(self, episodes):
        """执行PPO更新"""
        policy_losses = []
        value_losses = []

        for episode in episodes:
            if episode is None:
                continue

            # 重新计算当前策略的log概率
            current_policy_logits = self.policy_model(episode["generated_ids"], episode["attention_mask"])
            prompt_len = episode["prompt_ids"].shape[1]
            current_response_logits = current_policy_logits[:, prompt_len - 1 : -1, :]
            response_tokens = episode["generated_ids"][:, prompt_len:]

            current_log_probs = F.log_softmax(current_response_logits, dim=-1)
            current_selected_log_probs = torch.gather(current_log_probs, -1, response_tokens.unsqueeze(-1)).squeeze(-1)

            # 计算比率
            old_log_probs = episode["log_probs"].detach()
            ratio = torch.exp(current_selected_log_probs - old_log_probs)

            # PPO loss
            advantages = episode["advantages"].detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 标准化

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.args.clip_ratio, 1 + self.args.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            current_values = self.value_model(
                episode["generated_ids"][:, : prompt_len + episode["values"].shape[1]],
                (
                    episode["attention_mask"][:, : prompt_len + episode["values"].shape[1]]
                    if episode["attention_mask"] is not None
                    else None
                ),
            )
            current_values = current_values.view(episode["values"].shape)

            returns = episode["returns"].detach()
            value_loss = F.mse_loss(current_values, returns)

            policy_losses.append(policy_loss)
            value_losses.append(value_loss)

        if not policy_losses:
            return

        # 更新策略网络
        avg_policy_loss = torch.stack(policy_losses).mean()
        self.policy_optimizer.zero_grad()
        avg_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.args.max_grad_norm)
        self.policy_optimizer.step()

        # 更新价值网络
        avg_value_loss = torch.stack(value_losses).mean()
        self.value_optimizer.zero_grad()
        avg_value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.args.max_grad_norm)
        self.value_optimizer.step()

        if self.writer:
            self.writer.add_scalar("Loss/Policy", avg_policy_loss.item(), self.global_step)
            self.writer.add_scalar("Loss/Value", avg_value_loss.item(), self.global_step)

        return avg_policy_loss.item(), avg_value_loss.item()

    def train_epoch(self, dataloader, epoch):
        total_episodes = 0
        total_reward = 0
        total_kl_penalty = 0
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()

            prompt_ids = batch["prompt_ids"].to(self.args.device)
            attention_masks = batch["attention_masks"].to(self.args.device)

            # 生成episodes
            episodes = []
            for i in range(prompt_ids.shape[0]):
                episode = self.generate_episode(prompt_ids[i : i + 1], attention_masks[i : i + 1])
                if episode is not None:
                    episodes.append(episode)

            if not episodes:
                continue

            # PPO更新
            policy_loss, value_loss = self.ppo_update(episodes)

            for episode in episodes:
                total_episodes += 1
                total_reward += episode["episode_reward"].mean().item()
                total_kl_penalty += episode["kl_penalty"].mean().item()

            batch_time = time.time() - batch_start_time

            # log
            if batch_idx % self.args.log_interval == 0:
                avg_reward = total_reward / max(total_episodes, 1)
                avg_kl = total_kl_penalty / max(total_episodes, 1)

                print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}")
                print(f"  Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
                print(f"  Avg Reward: {avg_reward:.4f}, Avg KL: {avg_kl:.4f}")
                print(f"  Batch Time: {batch_time:.2f}s, Episodes: {len(episodes)}")

                if self.writer:
                    self.writer.add_scalar("Metrics/Average_Reward", avg_reward, self.global_step)
                    self.writer.add_scalar("Metrics/Average_KL_Penalty", avg_kl, self.global_step)
                    self.writer.add_scalar("Metrics/Episodes_Per_Batch", len(episodes), self.global_step)

            self.global_step += 1

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s, Total episodes: {total_episodes}")


def initialize_optimized_models(args):
    from minillm.model.model_base import MiniLLMModel as BaseModel

    # 1. 加载基础模型
    sft_model = BaseModel(args)
    if args.model_path:
        sft_state_dict = torch.load(args.model_path, map_location="cpu")
        sft_model.load_state_dict(sft_state_dict)

    # 2. 创建共享backbone模型
    shared_model = SharedBackboneModel(copy.deepcopy(sft_model))

    # 3. 初始化Policy Model和Value Model（共享backbone）
    policy_model = OptimizedPolicyModel(shared_model)
    value_model = OptimizedValueModel(shared_model)

    # 4. 初始化Reference Model（独立的backbone，用于KL计算）
    reference_shared_model = SharedBackboneModel(copy.deepcopy(sft_model))
    reference_model = OptimizedPolicyModel(reference_shared_model)

    # 5. 初始化Reward Model
    if args.share_reward_reference:
        reward_model = OptimizedRewardModel(reference_shared_model)
    else:
        reward_shared_model = SharedBackboneModel(copy.deepcopy(sft_model))
        reward_model = OptimizedRewardModel(reward_shared_model)

    # 6. 加载预训练的Reward Model权重
    if args.reward_model_path:
        reward_state_dict = torch.load(args.reward_model_path, map_location="cpu")
        # 只加载reward_head的权重
        reward_head_state = {
            k.replace("reward_head.", ""): v for k, v in reward_state_dict.items() if "reward_head" in k
        }
        if reward_head_state:
            reward_model.shared_model.reward_head.load_state_dict(reward_head_state)

    return policy_model, value_model, reference_model, reward_model


def setup_ppo_training(args):
    # 初始化模型
    policy_model, value_model, reference_model, reward_model = initialize_optimized_models(args)

    device = torch.device(args.device)
    policy_model.to(device)
    value_model.to(device)
    reference_model.to(device)
    reward_model.to(device)

    # 优化器
    policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.policy_lr, eps=1e-5)
    value_optimizer = torch.optim.Adam(value_model.parameters(), lr=args.value_lr, eps=1e-5)

    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tensorboard_logs"))

    # 创建PPO训练器
    trainer = PPOTrainer(
        policy_model=policy_model,
        value_model=value_model,
        reference_model=reference_model,
        reward_model=reward_model,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        args=args,
        writer=writer,
    )

    return trainer, writer


def main(args):
    trainer, writer = setup_ppo_training(args)

    # 加载tokenizer
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # 创建数据集和数据加载器
    dataset = PPODataset(args.data_path, tokenizer, args.max_prompt_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print(f"Dataset loaded: {len(dataset)} examples")
    print(f"Training for {args.epochs} epochs with batch size {args.batch_size}")

    # 确保输出目录存在
    os.makedirs(args.out_dir, exist_ok=True)

    # 训练循环
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Starting Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")

        trainer.train_epoch(dataloader, epoch)

        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.out_dir, f"ppo_checkpoint_epoch_{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "policy_model_state_dict": trainer.policy_model.state_dict(),
                    "value_model_state_dict": trainer.value_model.state_dict(),
                    "policy_optimizer_state_dict": trainer.policy_optimizer.state_dict(),
                    "value_optimizer_state_dict": trainer.value_optimizer.state_dict(),
                    "args": args,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved: {checkpoint_path}")

    # 关闭TensorBoard writer
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PPO Training for RLHF")

    # 模型路径
    parser.add_argument("--model_path", type=str, required=True, help="Path to SFT model")
    parser.add_argument("--reward_model_path", type=str, required=True, help="Path to reward model")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")

    # 数据相关
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--out_dir", type=str, default="./ppo_outputs", help="Output directory")

    # 模型参数
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Max prompt length")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max new tokens to generate")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--policy_lr", type=float, default=1e-6, help="Policy learning rate")
    parser.add_argument("--value_lr", type=float, default=1e-5, help="Value learning rate")

    # PPO参数
    parser.add_argument("--kl_coeff", type=float, default=0.02, help="KL penalty coefficient")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")

    # 生成参数
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    parser.add_argument("--pad_token_id", type=int, default=0, help="Pad token ID")

    # 优化参数
    parser.add_argument(
        "--share_reward_reference", action="store_true", help="Share backbone between reward and reference models"
    )

    # 其他参数
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=1, help="Save interval (epochs)")

    args = parser.parse_args()

    print("PPO Training Configuration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    main(args)
