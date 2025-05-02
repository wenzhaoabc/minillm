import os
import time
import math
import argparse
from contextlib import nullcontext
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers.models.auto.tokenization_auto import AutoTokenizer

from minillm.model.config import MiniLLMConfig as LMConfig
from minillm.model.model_v1 import MiniLLM
from minillm.utils.mllog import MLLogger
from minillm.train.dataset import SFTDataset


def init_model():
    model = MiniLLM(params=lm_config)
    state_dict = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    log.info(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    return model, tokenizer


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch():
    loss_function = torch.nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_dataloader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch + iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param in optimizer.param_groups:
            param["lr"] = lr

    with ctx:
        res = model(X)
        loss = loss_function(
            res.logits.view(-1, res.logits.size(-1)),
            Y.view(-1),
        ).view(Y.size())
        loss = (loss * loss_mask).sum() / loss_mask.sum()
        loss += res.aux_loss
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
            f"Epoch: {epoch}, Step: {step}/{iter_per_epoch}, Loss: {(loss.item() * args.accumulation_steps):.4f}, "
            f"LR: {lr:.6f}, Time: {spend_time:.2f}s"
        )
        log.log_training_progress(
            epoch=epoch,
            batch=step,
            total_batches=iter_per_epoch,
            loss=loss.item() * args.accumulation_steps,
            metrics={"lr": lr, "loss": loss.item() * args.accumulation_steps, "aux_loss": res.aux_loss.item()},
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
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl")
    parser.add_argument("--model_path", type=str, default="../out/pretrain/checkpoint_epoch_1_step_1000.pt")
    parser.add_argument("--tokenizer_path", type=str, default="../tokenizer")
    # Model parameters
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--use_moe", default=False, type=bool)

    args = parser.parse_args()
    lm_config = LMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len,
        use_moe=args.use_moe,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    log = MLLogger(
        experiment_name="sft",
        console_level="info",
        file_level="debug",
        use_tensorboard=True,
    )
    config_kv = {k: v for k, v in args._get_kwargs()}
    log.log_config(config_kv)

    ctx = nullcontext() if args.device == "cpu" else torch.autocast("cuda", dtype=torch.bfloat16)
    torch.manual_seed(288)
    if args.device == "cuda":
        torch.cuda.manual_seed(288)

    if args.ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(f"cuda:{args.local_rank}")
        RANK = dist.get_rank()
        LOCAL_RANK = args.local_rank
        WORLD_SIZE = dist.get_world_size()
        log.info(f"Distributed training initialized, RANK: {RANK}, LOCAL_RANK: {LOCAL_RANK}, WORLD_SIZE: {WORLD_SIZE}")

    model, tokenizer = init_model()
    train_dataset = SFTDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
    )
    train_sampler = DistributedSampler(train_dataset) if args.ddp else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=False,
    )
    log.info(f"Total samples: {len(train_dataset)}")
    scaler = torch.amp.grad_scaler.GradScaler(enabled=(args.dtype in ["bfloat16", "float16"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True
        )

    iter_per_epoch = len(train_dataloader)
    for epoch in range(args.epochs):
        train_epoch()
