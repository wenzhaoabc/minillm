import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import gc
import warnings
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from minillm.model.model_base import MiniLLMForCausalLM
from minillm.model.model_io import load_model_state
from dataset.lm_dataset import RLAIFDataset
from trainer.configs import SPOConfigArgs, parse_config_groups, namespace_from_configs
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, build_lm_config_from_args, init_wandb_run, resolve_checkpoint_dir, resolve_repo_path, save_run_metadata, resolve_output_dir

warnings.filterwarnings('ignore')


class AutoAdaptiveValueTracker:
    """SPO自适应价值追踪器"""
    def __init__(self, rho_mode='kl', rho_const=0.9, D_half=0.06, clip_lower=0.5, clip_upper=0.96):
        self.rho_mode = rho_mode
        self.rho_const = rho_const
        self.D_half = D_half
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper
        N_init = 1.0 / (1.0 - self.clip_lower)
        self.alpha = 0.5 * N_init
        self.beta = 0.5 * N_init
        self.old_mean_logprob = None

    def get_baselines(self, batch_size):
        baseline = self.alpha / (self.alpha + self.beta)
        return torch.full((batch_size,), baseline, dtype=torch.float32)

    def compute_rho(self, cur_mean_logprob):
        if self.rho_mode == 'constant':
            return self.rho_const
        if self.old_mean_logprob is None:
            return self.rho_const
        kl = abs(self.old_mean_logprob - cur_mean_logprob)
        rho = 2 ** (-kl / self.D_half)
        return max(min(rho, self.clip_upper), self.clip_lower)

    def update(self, rewards, cur_logprobs=None, response_masks=None):
        if cur_logprobs is not None and response_masks is not None:
            mean_logprob = ((cur_logprobs * response_masks).sum() / response_masks.sum()).item()
            rho = self.compute_rho(mean_logprob)
            self.old_mean_logprob = mean_logprob
        else:
            rho = self.rho_const

        scale = 3.0
        normalized_rewards = (rewards + scale) / (2 * scale)
        avg_normalized_reward = normalized_rewards.mean().item()
        self.alpha = rho * self.alpha + avg_normalized_reward
        self.beta = rho * self.beta + (1 - avg_normalized_reward)
        return rho


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """整合所有奖励函数计算总奖励"""
    def reasoning_model_reward(rewards):
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern or match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        def mark_num(text):
            reward = 0
            if text.count("<think>") == 1: reward += 0.25
            if text.count("</think>") == 1: reward += 0.25
            if text.count("<answer>") == 1: reward += 0.25
            if text.count("</answer>") == 1: reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    rewards = torch.zeros(len(responses), device=args.device)
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    with torch.no_grad():
        reward_model_scores = []
        scale = 3.0

        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            tmp_chat = messages + [{"role": "assistant", "content": response}]
            score = reward_model.get_score(reward_tokenizer, tmp_chat)
            score = max(min(score, scale), -scale)

            if args.reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    score = score * 0.4 + answer_score * 0.6

            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def spo_train_epoch(epoch, loader, iters, ref_model, reward_model, reward_tokenizer, value_tracker, start_step=0, wandb=None):
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # list[str], length B
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        with torch.no_grad():
            # DDP 模型需要使用 .module 访问 generate 方法
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            outputs = model_for_gen.generate(
                **prompt_inputs, max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)  # [B, P+R]

        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]  # [B, R]

        def get_per_token_logps(mdl, input_ids, n_keep):
            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
            per_token_logps = []
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
                ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
                per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1))
            return torch.stack(per_token_logps)

        with autocast_ctx:
            per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))  # [B, R]
            res = model(outputs) if lm_config.use_moe else None
            aux_loss = res.aux_loss if res is not None else torch.tensor(0.0, device=args.device)
        
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))  # [B, R]

        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)  # list[str], length B
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device)  # [B]

        baselines = value_tracker.get_baselines(len(prompts)).to(args.device)  # [B]

        scale = 3.0
        # Un-normalize baselines to be in the same scale as raw rewards [-3, 3]
        unnormalized_baselines = baselines * (2 * scale) - scale  # [B]
        advantages = rewards - unnormalized_baselines  # [B]

        # 直接使用 baseline 提供的优势估计，只做裁剪防止梯度爆炸。不再做 batch 内归一化，因为 baseline 已经提供了跨 batch 的稳定基线
        advantages = advantages.clamp(-5.0, 5.0)

        is_eos = completion_ids == tokenizer.eos_token_id  # [B, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)  # [B]
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()  # [B, R]

        kl_div = ref_per_token_logps - per_token_logps  # [B, R]
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B, R]
        per_token_loss = -per_token_logps * advantages.unsqueeze(1) + args.beta * per_token_kl  # [B, R]
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
        loss.backward()

        response_masks = completion_mask.float()  # [B, R]
        rho = value_tracker.update(rewards, per_token_logps.detach(), response_masks)

        if (step + 1) % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_val = ((per_token_kl * completion_mask).sum() / (completion_mask.sum() + 1e-8)).item()
            avg_baseline_val = baselines.mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Actor Loss: {policy_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, Reward: {avg_reward_val:.4f}, '
                   f'Baseline: {avg_baseline_val:.4f}, KL: {kl_val:.4f}, Rho: {rho:.4f}, '
                   f'Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": avg_reward_val,
                    "kl": kl_val,
                    "rho": float(rho),
                    "baseline": avg_baseline_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr
                })

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir=args.checkpoint_dir, output_dir=args.output_dir, scheduler=scheduler, tokenizer=tokenizer)
            model.train()

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, advantages, completion_mask, baselines, response_masks


if __name__ == "__main__":
    model_args, train_args, script_args, spo_args = parse_config_groups(SPOConfigArgs)
    if not script_args.data_path:
        script_args.data_path = "../dataset/rlaif-mini.jsonl"
    if script_args.save_weight == 'pretrain':
        script_args.save_weight = 'spo'
    if train_args.batch_size == 32:
        train_args.batch_size = 2
    if train_args.learning_rate == 5e-4:
        train_args.learning_rate = 1e-7
    if train_args.accumulation_steps == 1:
        train_args.accumulation_steps = 4
    if train_args.max_seq_len == 340:
        train_args.max_seq_len = 66
    if train_args.log_interval == 100:
        train_args.log_interval = 1
    if train_args.save_interval == 1000:
        train_args.save_interval = 10
    args = namespace_from_configs(model_args, train_args, script_args, spo_args)

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    args.save_dir = resolve_repo_path(args.save_dir)
    args.data_path = resolve_repo_path(args.data_path)
    args.reward_model_path = resolve_repo_path(args.reward_model_path)
    lm_config = build_lm_config_from_args(args, overrides={"max_position_embeddings": args.max_seq_len + args.max_gen_len})
    args.output_dir = resolve_output_dir(lm_config, output_dir=args.output_dir, save_dir=args.save_dir, save_weight=args.save_weight)
    args.checkpoint_dir = resolve_checkpoint_dir(output_dir=args.output_dir, checkpoint_dir=args.checkpoint_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.checkpoint_dir, output_dir=args.output_dir) if args.from_resume else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb_run_name = args.run_name or f"MiniLLM-SPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
    wandb = init_wandb_run(args, ckp_data, wandb_run_name, lm_config)
    save_run_metadata(args.output_dir, lm_config, args, extra={"stage": "spo"}, wandb=wandb)
    
    # ========== 5. 初始化模型（Policy, Ref, Reward）和Value Tracker、数据 ==========
    base_weight = args.from_weight or ("reason" if args.reasoning else "full_sft")
    # Policy模型
    model, tokenizer = init_model(lm_config, base_weight, tokenizer_path=args.tokenizer_path, save_dir=args.save_dir, load_dir=args.load_dir or args.load_from, device=args.device)
    if args.use_compile:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # Reference模型
    ref_model, _ = init_model(lm_config, base_weight, tokenizer_path=args.tokenizer_path, save_dir=args.save_dir, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Reward模型
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    # Value Tracker
    value_tracker = AutoAdaptiveValueTracker(rho_mode='kl', rho_const=0.9, D_half=0.06, clip_lower=0.5, clip_upper=0.96)
    
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(load_model_state(ckp_data['model_path'], map_location='cpu'))
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized() and dist.get_world_size() > 1:
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=bool(getattr(lm_config, "use_moe", False)),
        )
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            spo_train_epoch(epoch, loader, len(loader) + skip, ref_model, reward_model, reward_tokenizer, value_tracker, start_step, wandb)
        else:
            spo_train_epoch(epoch, loader, len(loader), ref_model, reward_model, reward_tokenizer, value_tracker, 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()