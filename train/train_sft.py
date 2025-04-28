import os
import argparse
from contextlib import nullcontext
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from model.config import LMConfig
from model.model_v1 import MiniLLM
from utils.mllog import MLLogger
from train.dataset import SFTDataset


def init_model(llm_config: LMConfig):
    tokenizer = AutoTokenizer.from_pretrained("./model/tokenizer")
    model = MiniLLM(llm_config)
    moe_path = "moe_" if args.use_moe else None
    checkpoint = f"./out/pretrain_{llm_config.dim}{moe_path}.pth"
    state_dict = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    log.log_model_summary(
        f"LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万"
    )
    return model, tokenizer


def init_distributed_mode():
    """Initialize distributed mode."""
    if not ddp:
        return
    global ddp_rank, ddp_local_rank, ddp_world_size, DEVICE
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ.get("RANK", -1))
    ddp_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    ddp_world_size = int(os.environ.get("WORLD_SIZE", -1))
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dim", default=512, type=int)
    parser.add_argument("--n_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--use_moe", default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./dataset/sft_mini_512.jsonl")

    args = parser.parse_args()

    llm_config = LMConfig(
        dim=args.dim,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        use_moe=args.use_moe,
    )

    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    tokens_per_iter = args.batch_size * llm_config.max_seq_len
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    ctx = nullcontext() if device_type == "cpu" else torch.autocast("cuda", dtype=torch.bfloat16)

    ddp = int(os.environ.get("RANK", -1)) != -1  # Check if distributed data parallel is enabled
    ddp_local_rank, DEVICE = 0, "cuda:0"
    base_seed = 288
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        ddp_rank = dist.get_rank()

    if args.use_tb and (ddp or ddp_local_rank == 0):
        from utils.mllog import MLLogger

        use_tensorboard = True
    else:
        use_tensorboard = False

    log = MLLogger(
        log_dir=args.save_dir,
        experiment_name="sft",
        console_level="info",
        file_level="debug",
        use_tensorboard=use_tensorboard,
    )
    model, tokenizer = init_model(llm_config)
    train_ds = SFTDataset(
        args.data_path,
        tokenizer=tokenizer,
        max_length=llm_config.max_seq_len,
    )
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    scaler = torch.GradScaler(enabled=(device_type == "cuda"), device=device_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch)
