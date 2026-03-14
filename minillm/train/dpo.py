import argparse
import os
import time
import math
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from transformers.models.auto.tokenization_auto import AutoTokenizer

from minillm.model.model_base import MiniLLMForCausalLM as MiniLLM
from minillm.model.config import MiniLLMConfig as LMConfig
from minillm.train.dataset import DPODataset
from minillm.utils.mllog import get_logger
from minillm.utils.train_util import (
    get_autocast_context,
    load_model_state_dict,
    load_training_checkpoint,
    save_training_checkpoint,
    validate_training_paths,
)


def init_model():
    # model
    model = MiniLLM(config=lm_config)
    state_dict = load_model_state_dict(args.model_path, device=args.device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(args.device)
    # ref model
    ref_model = MiniLLM(config=lm_config)
    ref_model.load_state_dict(state_dict, strict=False)
    ref_model.eval()
    ref_model.requires_grad_(False)
    ref_model = ref_model.to(args.device)
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    log.info(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    return model, ref_model, tokenizer


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def logits_to_probs(logits, labels):
    # logits : (batch_size, seq_len, vocab_size)
    # labels : (batch_size, seq_len)
    # Returns per-token log probabilities: log P(label_t | context)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, 2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs


def dpo_loss(ref_probs: torch.Tensor, probs: torch.Tensor, mask: torch.Tensor, beta: float = 0.1):
    # Implements the DPO objective:
    # L_DPO = -E[log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
    #
    # ref_probs : (batch_size, seq_len) — log probs from the frozen reference model
    # probs     : (batch_size, seq_len) — log probs from the trainable policy
    # mask      : (batch_size, seq_len) — 1 for response tokens, 0 for prompt/padding
    # beta      : KL penalty coefficient (higher β = stay closer to ref policy)

    # Average log-prob over response length for each sample
    seq_lengths = mask.sum(dim=1, keepdim=True)  # (batch_size, 1)
    ref_probs = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze()  # (batch_size,)
    probs = (probs * mask).sum(dim=1) / seq_lengths.squeeze()  # (batch_size,)

    # First half of batch: chosen (y_w); second half: rejected (y_l)
    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[: batch_size // 2]    # log π_ref(y_w|x)
    rejected_ref_probs = ref_probs[batch_size // 2 :]  # log π_ref(y_l|x)
    chosen_probs = probs[: batch_size // 2]            # log π_θ(y_w|x)
    rejected_probs = probs[batch_size // 2 :]          # log π_θ(y_l|x)

    # Implicit reward difference: β * (log π_θ/π_ref for chosen - log π_θ/π_ref for rejected)
    logits = (chosen_probs - rejected_probs) - (chosen_ref_probs - rejected_ref_probs)
    loss = -torch.nn.functional.logsigmoid(logits * beta)
    return loss.mean()


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
        extra_state={"beta": args.beta},
    )
    log.info(f"Checkpoint saved to {checkpoint_path}")
    log.info(f"Resume state saved to {resume_path}")


def train_epoch(start_step=0):
    start_time = time.time()
    for step, batch in enumerate(train_dataloader):
        if step < start_step:
            continue
        # Load chosen (preferred) and rejected sequences separately
        x_chosen = batch["x_chosen"].to(args.device)
        y_chosen = batch["y_chosen"].to(args.device)
        mask_chosen = batch["mask_chosen"].to(args.device)
        #
        x_rejected = batch["x_rejected"].to(args.device)
        y_rejected = batch["y_rejected"].to(args.device)
        mask_rejected = batch["mask_rejected"].to(args.device)

        # Stack chosen and rejected into a single forward pass for efficiency
        # First half = chosen, second half = rejected (assumed by dpo_loss)
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        lr = get_lr(epoch * iter_per_epoch + step, iter_per_epoch * args.epochs, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with ctx:
            # Get reference model log-probs (no gradient — ref model is frozen)
            with torch.no_grad():
                ref_res = ref_model(x)
                ref_logits = ref_res.logits
            ref_probs = logits_to_probs(ref_logits, y)
            ref_probs = ref_probs * mask
            # Get policy model log-probs (gradient flows here)
            res = model(x)
            logits = res.logits
            probs = logits_to_probs(logits, y)
            probs = probs * mask
            # DPO loss: encourages policy to prefer chosen over rejected relative to ref
            loss = dpo_loss(ref_probs, probs, mask, beta=args.beta)
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            log.log_training_progress(
                epoch=epoch,
                batch=step,
                total_batches=iter_per_epoch * args.epochs,
                loss=loss.item() * args.accumulation_steps,
                metrics={"step_time_seconds": spend_time},
                lr=lr,
                global_step=epoch * iter_per_epoch + step,
            )

        if (step + 1) % args.save_interval == 0 and (not args.ddp or dist.get_rank() == 0):
            model.eval()
            save_checkpoint(step)
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Training")
    # Path to the model checkpoint
    parser.add_argument("--model_path", type=str, default="path/to/your/model", help="Path to the model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default="path/to/your/tokenizer", help="Path to the tokenizer")
    parser.add_argument("--data_path", type=str, default="path/to/your/data", help="Path to the training data")
    parser.add_argument("--out_dir", type=str, default="path/to/output", help="Directory to save the model")
    #
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-8, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for training")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
    parser.add_argument("--ddp", action="store_true", help="Use distributed data parallel training")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--seed", type=int, default=288, help="Random seed for initialization")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training"
    )
    #
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=500, help="Model save interval")
    #
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size of the model")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="Number of hidden layers in the model")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--use_moe", action="store_true", help="Use mixture of experts")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a resume checkpoint")
    parser.add_argument("--use_tensorboard", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="minillm")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()
    lm_config = LMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len,
        use_moe=args.use_moe,
    )
    validate_training_paths(
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        out_dir=args.out_dir,
        model_path=args.model_path,
        resume_from=args.resume_from,
    )
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    ctx = get_autocast_context(args.device, args.dtype)

    if args.ddp:
        dist.init_process_group(backend="nccl")
        args.device = f"cuda:{args.local_rank}"
        torch.cuda.set_device(args.device)
        RANK = dist.get_rank()
        LOCAL_RANK = args.local_rank
        WORLD_SIZE = dist.get_world_size()

    log = get_logger(
        log_dir=os.path.join(args.out_dir, "logs"),
        experiment_name="dpo",
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

    model, ref_model, tokenizer = init_model()
    train_dataset = DPODataset(data_path=args.data_path, tokenizer=tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_dataset) if args.ddp else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    scaler = torch.amp.grad_scaler.GradScaler(enabled=(args.device != "cpu" and args.dtype in ["bfloat16", "float16"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
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

    if args.ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

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
