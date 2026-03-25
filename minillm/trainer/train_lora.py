import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from dataset.lm_dataset import SFTDataset
from model.model_lora import save_lora, load_lora, apply_lora
from trainer.configs import LoRAConfigArgs, parse_config_groups, namespace_from_configs
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler, build_lm_config_from_args, init_wandb_run, resolve_checkpoint_dir, resolve_repo_path, save_run_metadata, resolve_output_dir

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            save_lora(model, args.output_dir)
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir=args.checkpoint_dir, output_dir=args.output_dir, save_model_artifact=False, checkpoint_format='lora', save_total_limit=args.save_total_limit)
            model.train()

        del input_ids, labels, res, loss


if __name__ == "__main__":
    model_args, train_args, script_args, lora_args = parse_config_groups(LoRAConfigArgs)
    if not script_args.data_path:
        script_args.data_path = "../dataset/lora_identity.jsonl"
    if script_args.save_dir == "../out":
        script_args.save_dir = "../out/lora"
    if script_args.from_weight is None:
        script_args.from_weight = "full_sft"
    if train_args.epochs == 1:
        train_args.epochs = 50
    if train_args.learning_rate == 5e-4:
        train_args.learning_rate = 1e-4
    if train_args.log_interval == 100:
        train_args.log_interval = 10
    args = namespace_from_configs(model_args, train_args, script_args, lora_args)
    if args.save_weight == 'pretrain':
        args.save_weight = args.lora_name

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    args.save_dir = resolve_repo_path(args.save_dir)
    args.data_path = resolve_repo_path(args.data_path)
    lm_config = build_lm_config_from_args(args)
    args.output_dir = resolve_output_dir(lm_config, output_dir=args.output_dir, save_dir=args.save_dir, save_weight=args.save_weight)
    args.checkpoint_dir = resolve_checkpoint_dir(output_dir=args.output_dir, checkpoint_dir=args.checkpoint_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir=args.checkpoint_dir, output_dir=args.output_dir) if args.from_resume else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb_run_name = args.run_name or f"MiniLLM-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
    wandb = init_wandb_run(args, ckp_data, wandb_run_name, lm_config)
    save_run_metadata(args.output_dir, lm_config, args, extra={"stage": "lora", "lora_name": args.lora_name}, wandb=wandb)
    
    # ========== 5. 定义模型、应用LoRA、冻结非LoRA参数 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, tokenizer_path=args.tokenizer_path, save_dir=args.save_dir, load_dir=args.load_dir or args.load_from, device=args.device)
    if args.use_compile:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    apply_lora(model, rank=args.lora_rank)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
    
    # 冻结非LoRA参数，收集LoRA参数
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
    
    # ========== 6. 定义数据和优化器 ==========
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    
    # ========== 7. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        load_lora(model, ckp_data['model_dir'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 8. DDP包模型 ==========
    if dist.is_initialized() and dist.get_world_size() > 1:
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=bool(getattr(lm_config, "use_moe", False)),
        )
    
    # ========== 9. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)
    
    # ========== 10. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()