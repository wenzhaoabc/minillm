import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, Union


class MLLogger:
    """Unified ML logger for console/file logs plus optional TensorBoard and W&B."""

    LOG_LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    def __init__(
        self,
        log_dir: str = "./logs",
        experiment_name: Optional[str] = None,
        console_level: str = "info",
        file_level: str = "debug",
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_run_id: Optional[str] = None,
        wandb_resume: Optional[str] = None,
        wandb_mode: Optional[str] = None,
        is_main_process: bool = True,
    ):
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"

        self.experiment_name = experiment_name
        self.is_main_process = is_main_process
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        if self.logger.handlers:
            self.logger.handlers.clear()

        effective_console_level = "error" if not self.is_main_process else console_level
        effective_file_level = "error" if not self.is_main_process else file_level

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.LOG_LEVELS.get(effective_console_level.lower(), logging.INFO))
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        log_file = os.path.join(self.log_dir, f"{experiment_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.LOG_LEVELS.get(effective_file_level.lower(), logging.DEBUG))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.tensorboard_writer = None
        self.wandb_module = None
        self.wandb_run = None
        self.metrics_history: Dict[str, list[dict[str, Union[int, float, None]]]] = {}
        self._emitted_metric_events: set[tuple[str, Optional[int], str]] = set()

        if use_tensorboard and self.is_main_process:
            try:
                from torch.utils.tensorboard.writer import SummaryWriter

                tensorboard_dir = os.path.join(self.log_dir, "tensorboard")
                self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)
                self.logger.info(f"TensorBoard logs will be saved to {tensorboard_dir}")
            except ImportError:
                self.logger.warning("Could not import TensorBoard. TensorBoard logging disabled.")

        if use_wandb and self.is_main_process:
            try:
                import wandb

                init_kwargs: Dict[str, Any] = {
                    "project": wandb_project or "minillm",
                    "name": wandb_run_name or experiment_name,
                    "dir": self.log_dir,
                }
                if wandb_run_id:
                    init_kwargs["id"] = wandb_run_id
                if wandb_resume:
                    init_kwargs["resume"] = wandb_resume
                if wandb_mode:
                    init_kwargs["mode"] = wandb_mode

                self.wandb_module = wandb
                self.wandb_run = wandb.init(**init_kwargs)
                self.logger.info(f"W&B run initialized: {self.wandb_run.name}")
            except ImportError:
                self.logger.warning("Could not import wandb. W&B logging disabled.")

        self.logger.info(f"Initialized MLLogger for experiment: {experiment_name}")
        self.logger.info(f"Log directory: {self.log_dir}")

    def _format_scalar(self, value: Union[int, float]) -> str:
        if isinstance(value, int):
            return str(value)
        return f"{value:.6f}"

    def _to_scalar(self, value: Any) -> Optional[Union[int, float]]:
        if value is None:
            return None
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return value
        return None

    def _normalize_metrics(self, metrics: Optional[Dict[str, Any]]) -> Dict[str, Union[int, float]]:
        normalized: Dict[str, Union[int, float]] = {}
        for name, value in (metrics or {}).items():
            scalar = self._to_scalar(value)
            if scalar is not None:
                normalized[name] = scalar
        return normalized

    def _metric_key(self, stage: str, name: str) -> str:
        return name if "/" in name else f"{stage}/{name}"

    def _remember_metrics(self, stage: str, metrics: Dict[str, Union[int, float]], step: Optional[int]) -> None:
        for name, value in metrics.items():
            key = self._metric_key(stage, name)
            self.metrics_history.setdefault(key, []).append({"step": step, "value": value})

    def _register_metric_event(self, stage: str, metrics: Dict[str, Union[int, float]], step: Optional[int]) -> bool:
        payload = {self._metric_key(stage, name): value for name, value in metrics.items()}
        event_key = (stage, step, json.dumps(payload, sort_keys=True))
        if event_key in self._emitted_metric_events:
            return False
        self._emitted_metric_events.add(event_key)
        return True

    def _emit_metric_sinks(self, stage: str, metrics: Dict[str, Union[int, float]], step: Optional[int]) -> None:
        if step is None or not metrics:
            return

        payload = {self._metric_key(stage, name): value for name, value in metrics.items()}

        if self.tensorboard_writer:
            for name, value in payload.items():
                self.tensorboard_writer.add_scalar(name, value, step)

        if self.wandb_run is not None and self.wandb_module is not None:
            self.wandb_module.log(payload, step=step)

    def log_config(self, config: Dict[str, Any]) -> None:
        config_str = json.dumps(config, indent=2)
        self.logger.info(f"Experiment configuration:\n{config_str}")

        config_file = os.path.join(self.log_dir, "config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        if self.wandb_run is not None:
            self.wandb_run.config.update(config, allow_val_change=True)

        self.logger.info(f"Configuration saved to {config_file}")

    def log_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
        hyperparams_str = json.dumps(hyperparams, indent=2)
        self.logger.info(f"Hyperparameters:\n{hyperparams_str}")

        hyperparams_file = os.path.join(self.log_dir, "hyperparams.json")
        with open(hyperparams_file, "w", encoding="utf-8") as f:
            json.dump(hyperparams, f, indent=2)

        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_hparams(hyperparams, {})
            except RuntimeError:
                self.logger.warning("Failed to log hyperparameters to TensorBoard")

        if self.wandb_run is not None:
            self.wandb_run.config.update(hyperparams, allow_val_change=True)

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        stage: str = "metrics",
        log_to_console: bool = True,
    ) -> None:
        normalized = self._normalize_metrics(metrics)
        if not normalized:
            return

        if not self._register_metric_event(stage, normalized, step):
            return

        self._remember_metrics(stage, normalized, step)
        self._emit_metric_sinks(stage, normalized, step)

        if log_to_console:
            metric_str = ", ".join(f"{name}={self._format_scalar(value)}" for name, value in normalized.items())
            step_str = f", step={step}" if step is not None else ""
            self.logger.info(f"[{stage}]{step_str} {metric_str}")

    def log_training_progress(
        self,
        epoch: int,
        batch: int,
        total_batches: int,
        loss: float,
        metrics: Optional[Dict[str, Any]] = None,
        lr: Optional[float] = None,
        stage: str = "train",
        global_step: Optional[int] = None,
    ) -> None:
        global_step = epoch * total_batches + batch if global_step is None else global_step

        payload = self._normalize_metrics(metrics)
        payload.pop("loss", None)
        payload.pop("lr", None)
        payload = {"loss": self._to_scalar(loss) or loss, **payload}
        if lr is not None:
            payload["lr"] = lr

        if not self._register_metric_event(stage, payload, global_step):
            return

        message_parts = [
            f"[{stage}]",
            f"epoch={epoch}",
            f"batch={batch}/{total_batches}",
            f"step={global_step}",
            f"loss={self._format_scalar(payload['loss'])}",
        ]
        if lr is not None:
            message_parts.append(f"lr={self._format_scalar(lr)}")

        for name, value in payload.items():
            if name in {"loss", "lr"}:
                continue
            message_parts.append(f"{name}={self._format_scalar(value)}")

        self.logger.info(", ".join(message_parts))
        self._remember_metrics(stage, payload, global_step)
        self._emit_metric_sinks(stage, payload, global_step)

    def log_validation_results(self, epoch: int, metrics: Dict[str, Any], stage: str = "validation") -> None:
        normalized = self._normalize_metrics(metrics)
        if not normalized:
            return
        if not self._register_metric_event(stage, normalized, epoch):
            return

        metric_str = ", ".join(f"{name}={self._format_scalar(value)}" for name, value in normalized.items())
        self.logger.info(f"[{stage}] epoch={epoch}, {metric_str}")
        self._remember_metrics(stage, normalized, epoch)
        self._emit_metric_sinks(stage, normalized, epoch)

    def log_model_summary(self, model_summary: str) -> None:
        self.logger.info(f"Model Architecture:\n{model_summary}")

        model_file = os.path.join(self.log_dir, "model_architecture.txt")
        with open(model_file, "w", encoding="utf-8") as f:
            f.write(model_summary)

    def log_exception(self, exception: Exception) -> None:
        self.logger.error(f"Exception occurred: {str(exception)}")
        self.logger.error(traceback.format_exc())

    def save_metrics_history(self) -> None:
        metrics_file = os.path.join(self.log_dir, "metrics_history.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, indent=2)
        self.logger.info(f"Metrics history saved to {metrics_file}")

    def get_run(self):
        return self.wandb_run

    def close(self) -> None:
        self.save_metrics_history()

        if self.tensorboard_writer:
            self.tensorboard_writer.close()

        if self.wandb_run is not None and self.wandb_module is not None:
            self.wandb_module.finish()

        self.logger.info(f"MLLogger for experiment {self.experiment_name} closed")

        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)


def get_logger(
    log_dir: str = "./logs",
    experiment_name: Optional[str] = None,
    console_level: str = "info",
    file_level: str = "debug",
    use_tensorboard: bool = False,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_run_id: Optional[str] = None,
    wandb_resume: Optional[str] = None,
    wandb_mode: Optional[str] = None,
    is_main_process: bool = True,
) -> MLLogger:
    return MLLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        console_level=console_level,
        file_level=file_level,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_run_id=wandb_run_id,
        wandb_resume=wandb_resume,
        wandb_mode=wandb_mode,
        is_main_process=is_main_process,
    )


def log_execution_time(logger, func_name=None):
    def decorator(func):
        nonlocal func_name
        if func_name is None:
            func_name = func.__name__

        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.logger.info(f"Function '{func_name}' executed in {duration:.2f} seconds")
            return result

        return wrapper

    return decorator
