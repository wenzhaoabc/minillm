from utils.mllog import get_logger, log_execution_time
import time
import random
import traceback


def main():
    # 1. 初始化日志记录器
    # 简单用法：使用默认参数
    logger = get_logger()

    # 高级用法：自定义设置
    # logger = get_logger(
    #     log_dir="./experiment_logs",
    #     experiment_name="text_classification_bert",
    #     console_level="info",
    #     file_level="debug",
    #     use_tensorboard=True
    # )

    try:
        # 2. 记录配置参数
        config = {
            "model_name": "bert-base-uncased",
            "max_seq_length": 128,
            "batch_size": 32,
            "learning_rate": 3e-5,
            "epochs": 3,
            "dataset": "imdb",
            "seed": 42,
        }
        logger.log_config(config)

        # 3. 记录超参数
        hyperparams = {
            "learning_rate": 3e-5,
            "weight_decay": 0.01,
            "adam_epsilon": 1e-8,
            "warmup_steps": 100,
            "dropout_rate": 0.1,
        }
        logger.log_hyperparams(hyperparams)

        # 4. 记录模型结构摘要
        model_summary = """BERT Classifier(
          (bert): BertModel(...)
          (dropout): Dropout(p=0.1)
          (classifier): Linear(in_features=768, out_features=2)
        )"""
        logger.log_model_summary(model_summary)

        # 5. 模拟训练过程
        num_epochs = 3
        batches_per_epoch = 50

        for epoch in range(num_epochs):
            # 模拟训练循环
            for batch in range(batches_per_epoch):
                # 模拟训练损失和指标
                loss = 1.0 - 0.2 * epoch - 0.01 * batch + random.random() * 0.1
                accuracy = 0.7 + 0.1 * epoch + 0.003 * batch + random.random() * 0.05

                # 记录训练进度
                logger.log_training_progress(
                    epoch=epoch,
                    batch=batch,
                    total_batches=batches_per_epoch,
                    loss=loss,
                    metrics={"accuracy": accuracy},
                    lr=hyperparams["learning_rate"] * (0.9**epoch),
                )

                # 不要在实际代码中使用这个，这里只是为了模拟训练速度
                time.sleep(0.01)

            # 模拟验证结果
            val_metrics = {
                "accuracy": 0.75 + 0.08 * epoch + random.random() * 0.03,
                "precision": 0.73 + 0.09 * epoch + random.random() * 0.04,
                "recall": 0.71 + 0.07 * epoch + random.random() * 0.05,
                "f1": 0.72 + 0.08 * epoch + random.random() * 0.03,
            }

            # 记录验证结果
            logger.log_validation_results(epoch, val_metrics)

            # 模拟整体训练指标
            train_metrics = {
                "loss": 0.9 - 0.3 * epoch + random.random() * 0.1,
                "accuracy": 0.75 + 0.1 * epoch + random.random() * 0.05,
            }
            logger.log_metrics(train_metrics, step=epoch)

        # 6. 使用装饰器记录函数执行时间
        @log_execution_time(logger, "compute_embeddings")
        def compute_embeddings():
            # 模拟计算嵌入向量的耗时操作
            time.sleep(2)
            return "embeddings computed"

        result = compute_embeddings()
        logger.logger.info(f"Result: {result}")

    except Exception as e:
        # 7. 记录异常
        logger.log_exception(e)
    finally:
        # 8. 关闭日志记录器
        logger.close()


if __name__ == "__main__":
    main()
