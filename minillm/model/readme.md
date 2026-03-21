# 基于Triton使用Flash Attention


Triton 是一种用于并行编程的语言和编译器。它旨在提供一个基于 Python 的编程环境，用于高效地编写能够在现代 GPU 硬件上以最大吞吐量运行的自定义 DNN 计算内核。Flash Attention是一种高效的注意力机制计算方式，通过online softmax尽量减少 HBM 读写、提高 Tensor Core 利用率、降低显存占用。本文档介绍如何使用triton实现flash-attn的计算。

以简化为目的，该项目专注于实现forward阶段的flash-attn，backward阶段还需要考虑梯度等，实现较为复杂，该文档暂时不考虑。具体到GPU架构上，该文档只考虑博主在使用RTX 5090机器，即SM100+，核心实现参考官方`flash-attn-4`的实现。

## triton


