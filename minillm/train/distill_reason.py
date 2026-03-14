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
from minillm.model.model_base import MiniLLMForCausalLM as MiniLLM
from minillm.utils.mllog import get_logger
from minillm.utils.train_util import (
    get_autocast_context,
    load_model_state_dict,
    load_training_checkpoint,
    save_training_checkpoint,
    validate_training_paths,
)
from minillm.train.dataset import SFTDataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def init_model():
    model = MiniLLM(lm_config)
    state_dict = load_model_state_dict(args.model_path, device=args.device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(args.device)
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
    # format token ids
    start_of_think_ids = tokenizer("<think>").input_ids
    end_of_think_ids = tokenizer("</think>").input_ids
    start_of_answer_ids = tokenizer("<answer>").input_ids
    end_of_answer_ids = tokenizer("</answer>").input_ids

    loss_function = torch.nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_dataloader):
        if step < start_step:
            continue
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(
            epoch * iter_per_epoch + step,
            args.epochs * iter_per_epoch,
            args.learning_rate,
        )
        for param in optimizer.param_groups:
            param["lr"] = lr

        with ctx:
            res = model(X)
            loss = loss_function(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1),
            ).view(Y.size())

            format_token_ids = torch.isin(
                Y.view(-1),
                torch.tensor(start_of_think_ids + end_of_think_ids + start_of_answer_ids + end_of_answer_ids).to(
                    Y.device
                ),
            )

            loss_mask = loss_mask.view(-1)
            loss_mask[format_token_ids] = 10  # high weight for format tokens
            loss_mask = loss_mask.view(Y.size())

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
            log.log_training_progress(
                epoch=epoch,
                batch=step,
                total_batches=iter_per_epoch,
                loss=loss.item() * args.accumulation_steps,
                metrics={
                    "aux_loss": res.aux_loss.item(),
                    "step_time_seconds": spend_time,
                },
                lr=lr,
                global_step=epoch * iter_per_epoch + step,
            )

        if (step + 1) % args.save_interval == 0 and (not args.ddp or dist.get_rank() == 0):
            model.eval()
            save_checkpoint(step)
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Distill Reasoning")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl")
    parser.add_argument("--model_path", type=str, default="ckp.pt")
    parser.add_argument("--tokenizer_path", type=str, default="../tokenizer")
    # Model parameters
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=1024, type=int)
    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
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
    if args.ddp:
        dist.init_process_group(backend="nccl")
        args.device = f"cuda:{args.local_rank}"
        torch.cuda.set_device(args.device)
        RANK = dist.get_rank()
        LOCAL_RANK = args.local_rank
        WORLD_SIZE = dist.get_world_size()

    log = get_logger(
        log_dir=os.path.join(args.out_dir, "logs"),
        experiment_name="distill_reason",
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
    torch.manual_seed(288)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(288)

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
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=False,
    )
    log.info(f"Total samples: {len(train_dataset)}")
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
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True
        )

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
