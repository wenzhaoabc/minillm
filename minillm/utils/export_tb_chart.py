import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def export_dual_plots(log_path, output_name="loss_comparison.png"):
    # 1. 加载数据
    ea = event_accumulator.EventAccumulator(log_path)
    ea.Reload()

    # 定义要提取的两个标签
    tags = ['train/loss', 'train/aux_loss']
    data = {}

    for tag in tags:
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            data[tag] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events]
            }
        else:
            print(f"Warning: Tag '{tag}' not found in logs.")

    # 2. 创建画布：2行1列
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    plt.subplots_adjust(hspace=0.3) # 调整子图间的间距

    # 绘制上方图表 (Main Loss)
    if 'train/loss' in data:
        ax1.plot(data['train/loss']['steps'], data['train/loss']['values'], 
                 color='#1f77b4', label='Main Loss', linewidth=1.5)
        ax1.set_title('Training Main Loss', fontsize=14)
        ax1.set_ylabel('Loss Value')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

    # 绘制下方图表 (Aux Loss)
    if 'train/aux_loss' in data:
        ax2.plot(data['train/aux_loss']['steps'], data['train/aux_loss']['values'], 
                 color='#ff7f0e', label='Aux Loss', linewidth=1.5)
        ax2.set_title('Training Auxiliary Loss', fontsize=14)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Loss Value')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    # 3. 保存与展示
    plt.tight_layout()
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_name}")
    plt.show()

# 使用时请替换为你的 events 文件路径
export_dual_plots('/root/autodl-tmp/minillm/out/sft/logs/sft/tensorboard/events.out.tfevents.1773222791.autodl-container-c0e9kcmfc7-a19362b0.51204.0')