import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from minillm.model.model_base import MiniLLMForCausalLM
from minillm.model.model_io import load_model_state
from dataset.lm_dataset import RLAIFDataset
from trainer.configs import PPOConfigArgs, parse_config_groups, namespace_from_configs
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, build_lm_config_from_args, init_wandb_run, resolve_checkpoint_dir, resolve_repo_path, save_run_metadata, resolve_output_dir

warnings.filterwarnings('ignore')


# 自定义的Critic模型，继承自MiniLLMLM
class CriticModel(MiniLLMForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        # 替换lm_head为输出单一价值的线性层
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 使用基础模型获取隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        # 使用value_head获取价值估计
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """整合所有奖励函数计算总奖励"""
    def reasoning_model_reward(rewards):
        # 1. 格式奖励（仅针对训练推理模型时使用）
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        # 2. 标记奖励（防止严格奖励稀疏，仅针对训练推理模型时使用）
        def mark_num(text):
            reward = 0
            if text.count("<think>") == 1:
                reward += 0.25
            if text.count("</think>") == 1:
                reward += 0.25
            if text.count("<answer>") == 1:
                reward += 0.25
            if text.count("</answer>") == 1:
                reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    rewards = torch.zeros(len(responses), device=args.device)

    # 格式奖励
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    # 使用reward model计算整个response的奖励
    with torch.no_grad():
        reward_model_scores = []
        for prompt, response in zip(prompts, responses):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            tmp_chat = messages + [{"role": "assistant", "content": response}]
            score = reward_model.get_score(reward_tokenizer, tmp_chat)

            scale = 3.0
            score = max(min(score, scale), -scale)

            # 当args.reasoning=1时，额外计算<answer>内容的奖励
            if args.reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    # 对answer内容单独计算reward
                    tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    score = score * 0.4 + answer_score * 0.6
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def ppo_train_epoch(epoch, loader, iters, old_actor_model, ref_model, actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step=0, wandb=None):
    actor_model.train()
    critic_model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # list[str], length B
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, 
                       max_length=args.max_seq_len, padding_side="left").to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        prompt_length = enc.input_ids.shape[1]

        with torch.no_grad():
            # DDP 模型需要使用 .module 访问 generate 方法
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)  # [B, P+R]

        responses_text = [tokenizer.decode(gen_out[i, prompt_length:], skip_special_tokens=True) for i in range(len(prompts))]
        rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer)  # [B]

        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  # [B, P+R]
        last_indices = (full_mask * torch.arange(full_mask.size(1), device=gen_out.device)).argmax(dim=1)
        values = values_seq[torch.arange(values_seq.size(0), device=values_seq.device), last_indices]  # [B]
        advantages = rewards - values.detach()  # [B]

        with autocast_ctx:
            res = actor_model(input_ids=gen_out, attention_mask=full_mask)
            logits = res.logits  # [B, P+R, V]
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
        
        labels = gen_out[:, 1:].clone()  # [B, P+R-1]
        logp_tokens = F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
        seq_len = gen_out.size(1) - 1
        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= prompt_length - 1
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))  # [B, P+R-1]
        actor_logp = (logp_tokens * final_mask).sum(dim=1)  # [B]

        with torch.no_grad():
            old_logits = old_actor_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            old_logp_tokens = F.log_softmax(old_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            old_logp = (old_logp_tokens * final_mask).sum(dim=1)  # [B]
            
            ref_logits = ref_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            ref_logp_tokens = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)  # [B]

        kl = (actor_logp - old_logp).mean()  # scalar
        kl_ref = (actor_logp - ref_logp).mean()  # scalar
        ratio = torch.exp(actor_logp - old_logp)  # [B]
        surr1 = ratio * advantages  # [B]
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages  # [B]
        policy_loss = -torch.min(surr1, surr2).mean()  # scalar
        value_loss = F.mse_loss(values, rewards)  # scalar
        loss = (policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref + aux_loss) / args.accumulation_steps  # scalar
        loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        if is_main_process():
            response_ids = gen_out[:, enc.input_ids.shape[1]:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(has_eos, eos_indices + 1, torch.tensor(response_ids.shape[1], device=is_eos.device))
            avg_len = lengths.float().mean()

            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            current_aux_loss = aux_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "actor_loss": actor_loss_val,
                    "critic_loss": critic_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": reward_val,
                    "kl": kl_val,
                    "kl_ref": kl_ref_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                })

            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Actor Loss: {actor_loss_val:.4f}, Critic Loss: {critic_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, "
                   f"Reward: {reward_val:.4f}, KL: {kl_val:.4f}, KL_ref: {kl_ref_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        if (step + 1) % args.update_old_actor_freq == 0:
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            state_dict = raw_actor.state_dict()
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
            old_actor_model.to(args.device)

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            actor_model.eval()
            # 使用 lm_checkpoint 保存完整状态（包括 critic）
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir=args.checkpoint_dir, output_dir=args.output_dir,
                         tokenizer=tokenizer,
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()

        del enc, gen_out, responses_text, rewards, full_mask, values_seq, values, advantages
        del logits, labels, logp_tokens, final_mask, actor_logp, old_logits, old_logp, ref_logits, ref_logp
        del kl, kl_ref, ratio, surr1, surr2, policy_loss, value_loss, loss


if __name__ == "__main__":
    model_args, train_args, script_args, ppo_args = parse_config_groups(PPOConfigArgs)
    if not script_args.data_path:
        script_args.data_path = "../dataset/rlaif-mini.jsonl"
    if script_args.save_weight == 'pretrain':
        script_args.save_weight = 'ppo_actor'
    if train_args.batch_size == 32:
        train_args.batch_size = 2
    if train_args.learning_rate == 5e-4:
        train_args.learning_rate = 8e-8
    if train_args.max_seq_len == 340:
        train_args.max_seq_len = 66
    if train_args.log_interval == 100:
        train_args.log_interval = 1
    if train_args.save_interval == 1000:
        train_args.save_interval = 10
    args = namespace_from_configs(model_args, train_args, script_args, ppo_args)

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    args.save_dir = resolve_repo_path(args.save_dir)
    args.data_path = resolve_repo_path(args.data_path)
    args.reward_model_path = resolve_repo_path(args.reward_model_path)
    lm_config = build_lm_config_from_args(args)
    args.output_dir = resolve_output_dir(lm_config, output_dir=args.output_dir, save_dir=args.save_dir, save_weight=args.save_weight)
    args.checkpoint_dir = resolve_checkpoint_dir(output_dir=args.output_dir, checkpoint_dir=args.checkpoint_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.checkpoint_dir, output_dir=args.output_dir) if args.from_resume else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb_run_name = args.run_name or f"MiniLLM-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
    wandb = init_wandb_run(args, ckp_data, wandb_run_name, lm_config)
    save_run_metadata(args.output_dir, lm_config, args, extra={"stage": "ppo"}, wandb=wandb)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight or ("reason" if args.reasoning else "full_sft")
    actor_init_load_dir = args.load_dir or args.load_from
    # Actor模型
    actor_model, tokenizer = init_model(lm_config, base_weight, tokenizer_path=args.tokenizer_path, save_dir=args.save_dir, load_dir=actor_init_load_dir, device=args.device)
    if args.use_compile:
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')
    # Old Actor模型
    old_actor_model, _ = init_model(lm_config, base_weight, tokenizer_path=args.tokenizer_path, save_dir=args.save_dir, load_dir=actor_init_load_dir, device=args.device)
    old_actor_model = old_actor_model.eval().requires_grad_(False)
    # Reference模型
    ref_model, _ = init_model(lm_config, base_weight, tokenizer_path=args.tokenizer_path, save_dir=args.save_dir, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Critic模型
    critic_bootstrap_model, _ = init_model(lm_config, base_weight, tokenizer_path=args.tokenizer_path, save_dir=args.save_dir, load_dir=actor_init_load_dir, device=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(critic_bootstrap_model.state_dict(), strict=False)
    critic_model = critic_model.to(args.device)
    del critic_bootstrap_model
    # Reward模型
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(load_model_state(ckp_data['model_path'], map_location='cpu'))
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        old_actor_model.load_state_dict(actor_model.state_dict())
        old_actor_model.to(args.device)
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized() and dist.get_world_size() > 1:
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(
            actor_model,
            device_ids=[local_rank],
            find_unused_parameters=bool(getattr(lm_config, "use_moe", False)),
        )
        critic_model = DistributedDataParallel(
            critic_model,
            device_ids=[local_rank],
            find_unused_parameters=bool(getattr(lm_config, "use_moe", False)),
        )
        old_actor_model.to(args.device)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + skip, old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step, wandb)
        else:
            ppo_train_epoch(epoch, loader, len(loader), old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()