import argparse
import json
import os
import sys
from pathlib import Path

__package__ = "infer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
import warnings
import uvicorn

from threading import Thread
from queue import Queue
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from minillm.model.model_base import MiniLLMConfig, MiniLLMForCausalLM
from minillm.model.model_io import load_minillm_model
from model.model_lora import apply_lora, load_lora
from trainer.trainer_utils import Logger, resolve_output_dir, resolve_repo_path

warnings.filterwarnings('ignore')

app = FastAPI()


def resolve_lora_dir(args, lm_config):
    if args.lora_dir:
        return resolve_repo_path(args.lora_dir)
    if args.lora_weight == 'None':
        return None

    legacy_path = Path(resolve_repo_path(f'{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth'))
    if legacy_path.exists():
        return str(legacy_path)

    return resolve_output_dir(lm_config, save_dir=args.save_dir, save_weight=args.lora_weight)


def init_model(args):
    if args.load_from == 'model':
        lm_config = MiniLLMConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_position_embeddings=args.max_seq_len,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling,
        )
        load_path = resolve_output_dir(lm_config, output_dir=args.output_dir, save_dir=args.save_dir, save_weight=args.weight)
        Logger(f'Loading MiniLLM artifact from {load_path}')
        model, tokenizer = load_minillm_model(load_path, lm_config=lm_config, tokenizer_path=args.tokenizer_path, device=device, strict=True)
        lora_dir = resolve_lora_dir(args, lm_config)
        if lora_dir is not None:
            apply_lora(model)
            Logger(f'Loading LoRA artifact from {lora_dir}')
            load_lora(model, lora_dir)
    else:
        Logger(f'Loading Transformers model from {args.load_from}')
        tokenizer = AutoTokenizer.from_pretrained(args.load_from)
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    Logger(f'MiniLLM模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    return model.eval().to(device), tokenizer


class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    stream: bool = False
    tools: list = []


class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


def generate_stream_response(messages, temperature, top_p, max_tokens):
    try:
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)[-max_tokens:]
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        def _generate():
            model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )

        Thread(target=_generate).start()

        while True:
            text = queue.get()
            if text is None:
                yield json.dumps({
                    "choices": [{
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }, ensure_ascii=False)
                break

            yield json.dumps({
                "choices": [{"delta": {"content": text}}]
            }, ensure_ascii=False)

    except Exception as e:
        yield json.dumps({"error": str(e)})


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        if request.stream:
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens
                )),
                media_type="text/event-stream"
            )
        else:
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True
            )[-request.max_tokens:]
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + request.max_tokens,
                    do_sample=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=request.top_p,
                    temperature=request.temperature
                )
                answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minillm",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop"
                    }
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server for MiniLLM")
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--output_dir', default=None, type=str, help="标准化模型目录；设置后优先直接从该目录加载")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, dpo, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--tokenizer_path', default='minillm/tokenizer', type=str, help="原生MiniLLM权重加载时的tokenizer目录")
    parser.add_argument('--lora_dir', default=None, type=str, help="LoRA adapter目录；设置后优先直接从该目录加载")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--max_seq_len', default=8192, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    device = args.device
    model, tokenizer = init_model(args)
    Logger(f'Starting OpenAI-compatible server on 0.0.0.0:8998 with device={device}')
    uvicorn.run(app, host="0.0.0.0", port=8998)
