# MiniLLM Training Examples

这组示例面向单机训练场景，目标是提供一套统一、可读、可改的训练入口。

## 设计原则

- 所有示例默认从仓库根目录运行。
- 模型结构统一复用 `examples/configs/common/model_config.small.json`。
- 每个阶段只修改自己的 `train_config.json` 和 `script_config.json`。
- 优先演示显式 `output_dir` 和 `load_dir`，避免依赖旧的权重名前缀约定。

## 推荐流程

1. 训练 tokenizer。
2. 运行预训练 `pretrain`。
3. 运行全参数监督微调 `full_sft`。
4. 根据目标选择 `dpo` 或 `lora`。
5. 如果需要 RL 类后训练，再运行 `ppo`、`grpo` 或 `spo`。

## 目录说明

- `examples/configs/common`：共享模型结构配置。
- `examples/configs/<stage>`：阶段专属训练配置和运行配置。
- `examples/scripts`：带注释的启动脚本。

## 共享模型配置

当前统一使用一个小尺寸 dense 模型，适合作为从零开始练习工程链路的基线：

- `hidden_size = 512`
- `num_hidden_layers = 8`
- `num_attention_heads = 8`
- `num_key_value_heads = 2`
- `intermediate_size = 1408`
- `vocab_size = 6400`
- `use_moe = false`

这套配置和仓库里的 `Small-26M` 路线一致，显存压力较低，也更适合先把训练流程跑通。

## 使用方式

先检查并修改每个阶段配置里的数据路径，再运行对应脚本，例如：

```bash
bash examples/scripts/run_pretrain.sh
bash examples/scripts/run_full_sft.sh
bash examples/scripts/run_dpo.sh
```

多卡时可以覆盖 `NPROC_PER_NODE`：

```bash
NPROC_PER_NODE=4 bash examples/scripts/run_pretrain.sh
```

## 需要你自行调整的地方

- `datasets/*.jsonl` 只是约定路径，需要替换成你自己的数据文件。
- `REWARD_MODEL_PATH` 需要指向你本地可用的 reward model 目录。
- `batch_size`、`accumulation_steps`、`max_seq_len` 需要按显存实际调整。

## 说明

这里没有把 `train_reason.py` 放进统一示例链路，因为它还没有迁移到新的 config-group 与 artifact 规范。等该脚本完成迁移后，再把它接入这套 examples 会更干净。