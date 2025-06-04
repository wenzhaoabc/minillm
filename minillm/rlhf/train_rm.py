import os
import time
import math
import argparse
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers.models.auto.tokenization_auto import AutoTokenizer

from minillm.model.config import MiniLLMConfig as LMConfig
from minillm.utils.mllog import MLLogger
from minillm.rlhf.reward_model import RewardModel
from minillm.rlhf.ds_rm import RMDataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def init_model():
    model = RewardModel(lm_config)
    if args.model_path:
        state_dict = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)
    model = model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    log.info(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    return model, tokenizer


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch():
    start_time = time.time()
    for step, inputs in enumerate(train_dataloader):
        chosen_reward, _ = model(
            inputs["input_ids_chosen"].to(args.device),
            attention_mask=inputs["attention_mask_chosen"].to(args.device),
        )
        rejected_reward, _ = model(
            inputs["input_ids_rejected"].to(args.device),
            attention_mask=inputs["attention_mask_rejected"].to(args.device),
        )
        loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
        # Coefficient to incentivize the reward model to output mean-zero rewards
        # https://huggingface.co/papers/2312.09244
        loss += args.center_rewards_coefficient * torch.mean((chosen_reward + rejected_reward) ** 2)
        loss = loss / args.accumulation_steps

        lr = get_lr(epoch + iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param in optimizer.param_groups:
            param["lr"] = lr

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            # 计算奖励值统计信息
            chosen_mean = torch.mean(chosen_reward).item()
            rejected_mean = torch.mean(rejected_reward).item()
            chosen_std = torch.std(chosen_reward).item()
            rejected_std = torch.std(rejected_reward).item()
            reward_diff = chosen_mean - rejected_mean

            log.info(
                f"Epoch: {epoch}, Step: {step}/{iter_per_epoch}, Loss: {(loss.item() * args.accumulation_steps):.8f}, "
                f"LR: {lr:.8f}, Time: {spend_time:.2f}s"
                f"Chosen: {chosen_mean:.4f}±{chosen_std:.4f}, "
                f"Rejected: {rejected_mean:.4f}±{rejected_std:.4f}, "
                f"Diff: {reward_diff:.4f}"
            )
            log.log_training_progress(
                epoch=epoch,
                batch=step,
                total_batches=iter_per_epoch,
                loss=loss.item() * args.accumulation_steps,
                metrics={
                    "lr": lr,
                    "loss": loss.item() * args.accumulation_steps,
                    "chosen_mean": chosen_mean,
                    "rejected_mean": rejected_mean,
                    "chosen_std": chosen_std,
                    "rejected_std": rejected_std,
                    "reward_diff": reward_diff,
                },
                lr=lr,
            )

        if (step + 1) % args.save_interval == 0 and (not args.ddp or dist.get_rank() == 0):
            model.eval()
            checkpoint_path = os.path.join(args.out_dir, f"rm_cp_e{epoch}_s{step}.pt")
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, checkpoint_path)
            log.info(f"Checkpoint saved to {checkpoint_path}")
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reward Model Training")
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
    parser.add_argument("--center_rewards_coefficient", type=float, default=0.01)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl")
    parser.add_argument("--model_path", type=str, default="../out/pretrain/checkpoint_epoch_1_step_1000.pt")
    parser.add_argument("--tokenizer_path", type=str, default="../tokenizer")
    # Model parameters
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=1024, type=int)
    parser.add_argument("--use_moe", action="store_true")

    args = parser.parse_args()
    lm_config = LMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len,
        use_moe=args.use_moe,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    log = MLLogger(
        experiment_name="rm",
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
    train_dataset = RMDataset(
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
    scaler = torch.amp.grad_scaler.GradScaler(device=args.device, enabled=(args.dtype in ["bfloat16", "float16"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if args.ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True
        )

    iter_per_epoch = len(train_dataloader)
    for epoch in range(args.epochs):
        train_epoch()
