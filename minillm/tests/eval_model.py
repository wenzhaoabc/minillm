import argparse
import random
import warnings
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from minillm.model.config import MiniLLMConfig as LMConfig
from minillm.model.model_base import MiniLLMForCausalLM as MiniLLM
from minillm.utils.train_util import load_model_state_dict


warnings.filterwarnings("ignore")


def init_model(args, lm_config):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )
    if args.load == 0:
        model = MiniLLM(lm_config)

        model.load_state_dict(
            load_model_state_dict(args.model_path, device=args.device), strict=False
        )

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, trust_remote_code=True
        )
    print(
        f"MiniMind模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)"
    )
    return model.eval().to(args.device), tokenizer


def get_prompt_datas(args):
    if args.model_mode == 0:
        # pretrain模型的接龙能力（无法对话）
        prompt_datas = [
            "马克思主义基本原理",
            "人类大脑的主要功能",
            "万有引力原理是",
            "世界上最高的山峰是",
            "二氧化碳在空气中",
            "地球上最大的动物有",
            "杭州市的美食有",
        ]
    else:
        if args.lora_name == "None":
            # 通用对话问题
            prompt_datas = [
                "请介绍一下自己。",
                "你更擅长哪一个学科？",
                "鲁迅的《狂人日记》是如何批判封建礼教的？",
                "我咳嗽已经持续了两周，需要去医院检查吗？",
                "详细的介绍光速的物理概念。",
                "推荐一些杭州的特色美食吧。",
                "请为我讲解“大语言模型”这个概念。",
                "如何理解ChatGPT？",
                "Introduce the history of the United States, please.",
            ]
        else:
            # 特定领域问题
            lora_prompt_datas = {
                "lora_identity": [
                    "你是ChatGPT吧。",
                    "你叫什么名字？",
                    "你和openai是什么关系？",
                ],
                "lora_medical": [
                    "我最近经常感到头晕，可能是什么原因？",
                    "我咳嗽已经持续了两周，需要去医院检查吗？",
                    "服用抗生素时需要注意哪些事项？",
                    "体检报告中显示胆固醇偏高，我该怎么办？",
                    "孕妇在饮食上需要注意什么？",
                    "老年人如何预防骨质疏松？",
                    "我最近总是感到焦虑，应该怎么缓解？",
                    "如果有人突然晕倒，应该如何急救？",
                ],
            }
            prompt_datas = lora_prompt_datas[args.lora_name]

    return prompt_datas


# 设置可复现的随机种子
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args, lm_config):
    model, tokenizer = init_model(args, lm_config)

    prompts = get_prompt_datas(args)
    test_mode = int(input("[0] 自动测试\n[1] 手动输入\n"))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    messages = []
    for idx, prompt in enumerate(
        prompts if test_mode == 0 else iter(lambda: input("👶: "), "")
    ):
        setup_seed(random.randint(0, 2048))
        # setup_seed(2025)  # 如需固定每次输出则换成【固定】的随机种子
        if test_mode == 0:
            print(f"👶: {prompt}")

        messages = messages[-args.history_cnt :] if args.history_cnt else []
        messages.append({"role": "user", "content": prompt})

        new_prompt = (
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if args.model_mode != 0
            else (tokenizer.bos_token + prompt)
        )

        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(
            args.device
        )

        print("🤖️: ", end="")
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_seq_len,
            num_return_sequences=1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            top_p=args.top_p,
            temperature=args.temperature,
        )

        response = tokenizer.decode(
            generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        messages.append({"role": "assistant", "content": response})
        print("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with MiniMind")
    parser.add_argument("--lora_name", default="None", type=str)
    parser.add_argument("--out_dir", default="out", type=str)
    parser.add_argument("--temperature", default=0.85, type=float)
    parser.add_argument("--top_p", default=0.85, type=float)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str
    )
    # 此处max_seq_len（最大输出长度）并不意味模型具有对应的长文本的性能，仅防止QA出现被截断的问题
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=1024, type=int)
    parser.add_argument("--use_moe", action="store_true")
    # 携带历史对话上下文条数
    parser.add_argument("--history_cnt", default=0, type=int)
    parser.add_argument("--model_path", default="./out/pretrain", type=str)
    parser.add_argument("--tokenizer_path", type=str, default="minillm/tokenizer")
    parser.add_argument(
        "--load", default=0, type=int, help="0: 原生torch权重，1: transformers加载"
    )
    parser.add_argument(
        "--model_mode",
        default=1,
        type=int,
        help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型，4: RLAIF-Chat模型",
    )
    args = parser.parse_args()

    lm_config = LMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len,
        use_moe=args.use_moe,
    )

    main(args, lm_config)
