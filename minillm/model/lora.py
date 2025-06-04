# Lora
import torch
import torch.nn as nn

from minillm.model.model_base import MiniLLMForCausalLM as MiniLLM
from minillm.model.config import MiniLLMConfig


class LoraLayer(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵 A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵 B
        # 矩阵A 高斯初始化
        self.A.weight.data.normal_(0, 0.01)
        # 矩阵B 全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        # 计算低秩矩阵的输出
        return self.B(self.A(x))


def apply_lora(model, rank):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            # 获取输入输出特征数
            in_features = module.in_features
            out_features = module.out_features
            # 创建低秩矩阵层
            lora = LoraLayer(in_features, out_features, rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            def lora_forward(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            # 替换原有线性层
            module.forward = lora_forward


def save_lora(model: nn.Module, path: str):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {f"{name}.lora.{k}": v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)

    torch.save(state_dict, path)


def load_lora(model: nn.Module, path: str):
    state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {k.replace(f"{name}.lora.", ""): v for k, v in state_dict.items() if f"{name}.lora." in k}
            module.lora.load_state_dict(lora_state)


""" llm = MiniLLM(MiniLLMConfig())
apply_lora(llm, rank=4)

for name, module in llm.named_modules():
    if isinstance(module, nn.Linear):
        print(name, " -> ", module)
        print(name, " -> ", module.weight.shape, " -> ", module.in_features, ",", module.out_features)
        print() """
