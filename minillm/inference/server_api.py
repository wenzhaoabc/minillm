import argparse
import json
import os
import time
import uvicorn
from queue import Queue
from threading import Thread

import torch
from transformers import AutoTokenizer, TextStreamer
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

from minillm.model.model_base import MiniMindForCausalLM as MiniLLM
from minillm.model.config import MiniLLMConfig
from minillm.utils.mllog import get_logger

app = FastAPI()


class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
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
        input_ids = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        def _generate():
            model.generate(
                input_ids=input_ids["input_ids"],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                attention_mask=input_ids["attention_mask"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )

        Thread(target=_generate).start()

        while True:
            text = queue.get()
            if text is None:
                yield json.dumps(
                    {"choices": [{"delta": {"content": ""}}, {"finish_reason": "stop"}]}, ensure_ascii=False
                )
            yield json.dumps({"choices": [{"delta": {"content": text}}, {"finish_reason": None}]}, ensure_ascii=False)

    except Exception as e:
        log.info(f"Error in applying chat template: {e}")
        yield json.dumps({"error": str(e)})


@app.post("/v1/chat/completions")
async def generate_response(request: ChatRequest):
    try:
        if request.stream:
            return StreamingResponse(
                (
                    f"data: {chunk}\n\n"
                    for chunk in generate_stream_response(
                        messages=request.messages,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        max_tokens=request.max_tokens,
                    )
                ),
                media_type="text/event-stream",
            )
        else:
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
            )[-request.max_tokens :]
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_length=request.max_tokens + inputs["input_ids"].shape[1],
                    do_sample=True,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_text = tokenizer.batch_decode(
                generated_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            return {
                "id": f"chatcmpl-{os.urandom(16).hex()}",
                "object": "chat.completion",
                "created": time.time(),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
            }

    except Exception as e:
        pass


def init_model():
    model = MiniLLM(ml_cofig)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model = model.eval().to(device)

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MiniLLM Inference Server")
    parser.add_argument("--model_path", type=str, help="Path to the model directory")
    parser.add_argument("--tokenizer_path", type=str, help="Path to the tokenizer directory")
    parser.add_argument("--port", type=int, default=8000, help="Port for the server")

    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--use_moe", action="store_true", help="Use MoE model")
    args = parser.parse_args()

    log = get_logger(experiment_name="infer")

    ml_cofig = MiniLLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        use_moe=args.use_moe,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = init_model()
    log.info(f"Model loaded on {device}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
    )
