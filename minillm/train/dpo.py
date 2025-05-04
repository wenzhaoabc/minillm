import argparse
import os
import time
import math
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from transformers.models.auto.tokenization_auto import AutoTokenizer

from minillm.model.model_base import MiniMindForCausalLM as MiniLLM
from minillm.model.config import MiniLLMConfig as LMConfig
from minillm.train.dataset import DPODataset
from minillm.utils.mllog import get_logger


def init_model():
    # model
    model = MiniLLM(config=lm_config)
    state_dict = torch.load(args.model_path, map_location=args.device)
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
    # probs : (batch_size, seq_len)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, 2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs


def dpo_loss(ref_probs: torch.Tensor, probs: torch.Tensor, mask: torch.Tensor, beta: float = 0.1):
    # ref_probs : (batch_size, seq_len)
    # probs : (batch_size, seq_len)
    # mask : (batch_size, seq_len)
    seq_lengths = mask.sum(dim=1, keepdim=True)  # (batch_size, 1)
    ref_probs = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze()  # (batch_size,)
    probs = (probs * mask).sum(dim=1) / seq_lengths.squeeze()  # (batch_size,)
    #
    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[: batch_size // 2]  # (batch_size // 2,)
    rejected_ref_probs = ref_probs[batch_size // 2 :]  # (batch_size // 2,)
    chosen_probs = probs[: batch_size // 2]  # (batch_size // 2,)
    rejected_probs = probs[batch_size // 2 :]  # (batch_size // 2,)

    # Compute the DPO loss
    logits = (chosen_probs - rejected_probs) - (chosen_ref_probs - rejected_ref_probs)
    loss = -torch.nn.functional.logsigmoid(logits * beta)
    return loss.mean()


def train_epoch():
    start_time = time.time()
    for step, batch in enumerate(train_dataloader):
        #
        x_chosen = batch["x_chosen"].to(args.device)
        y_chosen = batch["y_chosen"].to(args.device)
        mask_chosen = batch["mask_chosen"].to(args.device)
        #
        x_rejected = batch["x_rejected"].to(args.device)
        y_rejected = batch["y_rejected"].to(args.device)
        mask_rejected = batch["mask_rejected"].to(args.device)
        #
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        lr = get_lr(epoch * iter_per_epoch + step, iter_per_epoch * args.epochs, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with ctx:
            with torch.no_grad():
                ref_res = ref_model(x)
                ref_logits = ref_res.logits
            ref_probs = logits_to_probs(ref_logits, y)
            ref_probs = ref_probs * mask
            #
            res = model(x)
            logits = res.logits
            probs = logits_to_probs(logits, y)
            probs = probs * mask
            #
            loss = dpo_loss(ref_probs, probs, mask, beta=0.1)
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
            log.info(
                f"Epoch: {epoch}, Step: {step}/{iter_per_epoch}, Loss: {(loss.item() * args.accumulation_steps):.8f}, "
                f"LR: {lr:.8f}, Time: {spend_time:.2f}s"
            )
            log.log_training_progress(
                epoch=epoch,
                batch=step,
                total_batches=iter_per_epoch * args.epochs,
                loss=loss.item() * args.accumulation_steps,
                metrics={"lr": lr, "loss": loss.item() * args.accumulation_steps},
                lr=lr,
            )

        if (step + 1) % args.save_interval == 0 and (not args.ddp or dist.get_rank() == 0):
            model.eval()
            checkpoint_path = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch}_step_{step}.pt")
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, checkpoint_path)
            log.info(f"Checkpoint saved to {checkpoint_path}")
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
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for training")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
    parser.add_argument("--ddp", action="store_true", help="Use distributed data parallel training")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training"
    )
    #
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=500, help="Model save interval")
    #
    parser.add_argument("--hiden_size", type=int, default=512, help="Hidden size of the model")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="Number of hidden layers in the model")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--use_moe", action="store_true", help="Use mixture of experts")

    args = parser.parse_args()
    lm_config = LMConfig(
        hidden_size=args.hiden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len,
        use_moe=args.use_moe,
    )
    log = get_logger(
        experiment_name="dpo",
        console_level="info",
        file_level="debug",
        use_tensorboard=True,
    )
    config_kv = {k: v for k, v in args._get_kwargs()}
    log.log_config(config_kv)

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    ctx = nullcontext() if args.device == "cpu" else torch.autocast("cuda", dtype=torch.bfloat16)

    if args.ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(f"cuda:{args.local_rank}")
        RANK = dist.get_rank()
        LOCAL_RANK = args.local_rank
        WORLD_SIZE = dist.get_world_size()
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

    scaler = torch.amp.grad_scaler.GradScaler(device=args.device, enabled=(args.dtype in ["bfloat16", "float16"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if args.ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    iter_per_epoch = len(train_dataloader)
    for epoch in range(args.epochs):
        train_epoch()
