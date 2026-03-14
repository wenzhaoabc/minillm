import os
import time
import math
import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers.models.auto.tokenization_auto import AutoTokenizer

from minillm.model.model_base import MiniLLMForCausalLM as MiniLLM
from minillm.model.config import MiniLLMConfig as MiniLLMConfig
from minillm.utils.mllog import get_logger
from minillm.utils.train_util import get_autocast_context, load_training_checkpoint, save_training_checkpoint, validate_training_paths
from minillm.train.dataset import PretrainDataset


def init_model():
    model = MiniLLM(config=lm_config).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    log.info(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    return model, tokenizer


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def save_checkpoint(step, checkpoint_name=None):
    checkpoint_name = checkpoint_name or f"checkpoint_epoch_{epoch}_step_{step}.pt"
    checkpoint_path, resume_path = save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=epoch,
        step=step,
        out_dir=args.out_dir,
        checkpoint_name=checkpoint_name,
    )
    log.info(f"Checkpoint saved to {checkpoint_path}")
    log.info(f"Resume state saved to {resume_path}")


def train_epoch(start_step=0):
    # Objective: L_PT = -1/N * sum_t log P_θ(x_t | x_{1:t-1})
    # reduction="none" so we can apply the loss_mask manually
    loss_function = torch.nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_dataloader):
        if step < start_step:
            continue
        # X: input token ids [B, T-1], Y: target token ids [B, T-1], loss_mask: [B, T-1]
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # Cosine LR decay: lr(t) = lr_min + 0.5 * lr_max * (1 + cos(π*t/T))
        lr = get_lr(epoch * iter_per_epoch + step, iter_per_epoch * args.epochs, args.learning_rate)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with ctx:
            res = model(X)
            # Per-token cross-entropy loss: CE(logits, targets)
            loss = loss_function(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(Y.size())
            # Mask out padding tokens; average only over valid positions
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            # Add MoE auxiliary load-balancing loss (zero for dense models)
            loss += res.aux_loss
            # Scale down for gradient accumulation
            loss = loss / args.accumulation_steps

        # Scale loss for mixed-precision, then backpropagate
        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            # Restore gradients to original scale before clipping/stepping
            scaler.unscale_(optimizer)
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # Update model parameters
            scaler.step(optimizer)
            # Adjust the loss scale factor for next iteration
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            log.log_training_progress(
                epoch=epoch,
                batch=step,
                total_batches=iter_per_epoch,
                loss=loss.item() * args.accumulation_steps,
                metrics={"aux_loss": res.aux_loss.item(), "step_time_seconds": spend_time},
                lr=lr,
                global_step=epoch * iter_per_epoch + step,
            )

        if (step + 1) % args.save_interval == 0 and (not args.ddp or dist.get_rank() == 0):
            model.eval()
            save_checkpoint(step)
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=288)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    parser.add_argument("--tokenizer_path", type=str, default="../tokenizer")
    # Model parameters
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--use_tensorboard", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="minillm")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    lm_config = MiniLLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len,
        use_moe=args.use_moe,
    )

    validate_training_paths(
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        out_dir=args.out_dir,
        resume_from=args.resume_from,
    )
    if args.ddp:
        dist.init_process_group(backend="nccl")
        args.device = f"cuda:{args.local_rank}"
        torch.cuda.set_device(args.device)
        RANK = dist.get_rank()
        LOCAL_RANK = args.local_rank
        WORLD_SIZE = dist.get_world_size()

    log = get_logger(
        log_dir=os.path.join(args.out_dir, "logs"),
        experiment_name="pretrain",
        console_level="info",
        file_level="debug",
        use_tensorboard=bool(args.use_tensorboard),
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        is_main_process=(not args.ddp or dist.get_rank() == 0),
    )
    config_kv = {k: v for k, v in args._get_kwargs()}
    log.log_config(config_kv)
    if args.ddp:
        log.info(f"Distributed training initialized, RANK: {RANK}, LOCAL_RANK: {LOCAL_RANK}, WORLD_SIZE: {WORLD_SIZE}")

    ctx = get_autocast_context(args.device, args.dtype)

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = init_model()
    train_dataset = PretrainDataset(args.data_path, tokenizer, args.max_seq_len)
    train_sampler = DistributedSampler(train_dataset) if args.ddp else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler,
    )

    # GradScaler enables mixed-precision training by scaling loss to avoid underflow in float16
    scaler = torch.amp.grad_scaler.GradScaler(enabled=(args.device != "cpu" and args.dtype in ["bfloat16", "float16"]))
    # AdamW optimizer — combines Adam with decoupled weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True
        )

    start_epoch = 0
    start_step = 0
    if args.resume_from:
        start_epoch, last_step = load_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            checkpoint_path=args.resume_from,
            device=args.device,
        )
        start_step = last_step + 1
        log.info(f"Resuming training from epoch={start_epoch}, step={start_step}")

    iter_per_epoch = len(train_dataloader)
    if start_step >= iter_per_epoch:
        start_epoch += 1
        start_step = 0
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_start_step = start_step if epoch == start_epoch else 0
        train_epoch(epoch_start_step)
        if (not args.ddp or dist.get_rank() == 0) and iter_per_epoch > 0:
            model.eval()
            save_checkpoint(iter_per_epoch - 1, checkpoint_name=f"checkpoint_epoch_{epoch}_final.pt")
            model.train()
    log.close()
