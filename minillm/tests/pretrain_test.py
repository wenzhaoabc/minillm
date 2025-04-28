import torch
from minillm.model.model_v1 import MiniLLM
from minillm.model.config import MiniLLMConfig as LMConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM


# 加载模型配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 加载预训练权重
checkpoint_path = "path/to/your/checkpoint.pt"
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
model.eval().to(DEVICE)  # 将模型移动到GPU上


# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/minillm/tokenizer")

# 输入文本
input_text = "测试输入文本"
input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)["input_ids"]
input_ids = input_ids.to(DEVICE)  # 将输入数据移动到GPU上

# 推理
with torch.no_grad():
    output = model(input_ids=input_ids)

# 获取预测结果
logits = output.logits
predicted_ids = torch.argmax(logits, dim=-1)
predicted_ids = predicted_ids.to(DEVICE)
predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

print("输入文本:", input_text)
print("生成文本:", predicted_text)
