import logging
import os
import time
from datetime import datetime
import json
import sys
import traceback
from typing import Dict, Any, Optional, Union, List


class MLLogger:
    """
    A comprehensive logging utility for machine learning applications.
    This logger provides methods to log various ML-specific events and metrics.
    """

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
    ):
        """
        Initialize the MLLogger.

        Args:
            log_dir: Directory to store log files
            experiment_name: Name of the current experiment
            console_level: Logging level for console output
            file_level: Logging level for file output
            use_tensorboard: Whether to log to TensorBoard
        """
        # Create experiment name based on timestamp if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"

        self.experiment_name = experiment_name
        self.log_dir = os.path.join(log_dir, experiment_name)

        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.DEBUG)  # Capture all levels
        self.logger.propagate = False

        # Clear existing handlers if any
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.LOG_LEVELS.get(console_level.lower(), logging.INFO))
        console_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler
        log_file = os.path.join(self.log_dir, f"{experiment_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.LOG_LEVELS.get(file_level.lower(), logging.DEBUG))
        file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

        # TensorBoard support
        self.tensorboard_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard.writer import SummaryWriter

                tensorboard_dir = os.path.join(self.log_dir, "tensorboard")
                self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)
                self.logger.info(f"TensorBoard logs will be saved to {tensorboard_dir}")
            except ImportError:
                self.logger.warning("Could not import TensorBoard. TensorBoard logging disabled.")

        # Log basic info
        self.logger.info(f"Initialized MLLogger for experiment: {experiment_name}")
        self.logger.info(f"Log directory: {self.log_dir}")

        # Metrics tracking
        self.metrics_history = {}

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration parameters for the experiment."""
        self.logger.info("Experiment configuration:")
        config_str = json.dumps(config, indent=2)
        self.logger.info(f"\n{config_str}")

        # Also save to file
        config_file = os.path.join(self.log_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Configuration saved to {config_file}")

    def log_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
        """Log hyperparameters for the experiment."""
        self.logger.info("Hyperparameters:")
        hyperparams_str = json.dumps(hyperparams, indent=2)
        self.logger.info(f"\n{hyperparams_str}")

        # Also save to file
        hyperparams_file = os.path.join(self.log_dir, "hyperparams.json")
        with open(hyperparams_file, "w") as f:
            json.dump(hyperparams, f, indent=2)

        # Log to TensorBoard if available
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_hparams(hyperparams, {})
            except:
                self.logger.warning("Failed to log hyperparameters to TensorBoard")

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """Log metrics from training or evaluation."""
        step_info = f" (Step: {step})" if step is not None else ""
        self.logger.info(f"Metrics{step_info}:")

        for name, value in metrics.items():
            self.logger.info(f"  {name}: {value}")

            # Update metrics history
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append((step, value))

            # Log to TensorBoard if available
            if self.tensorboard_writer and step is not None:
                self.tensorboard_writer.add_scalar(name, value, step)

    def log_training_progress(
        self,
        epoch: int,
        batch: int,
        total_batches: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
    ) -> None:
        """Log training progress."""
        progress = f"Epoch: {epoch}, Batch: {batch}/{total_batches}, Loss: {loss:.4f}"

        if lr is not None:
            progress += f", LR: {lr:.8f}"

        if metrics:
            metrics_str = ", ".join([f"{name.upper()}: {value:.8f}" for name, value in metrics.items()])
            progress += f", {metrics_str}"

        self.logger.info(progress)

        # Log to TensorBoard if available
        if self.tensorboard_writer:
            step = epoch * total_batches + batch
            self.tensorboard_writer.add_scalar("training/loss", loss, step)

            if lr is not None:
                self.tensorboard_writer.add_scalar("training/lr", lr, step)

            if metrics:
                for name, value in metrics.items():
                    self.tensorboard_writer.add_scalar(f"training/{name}", value, step)

    def log_validation_results(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log validation results."""
        self.logger.info(f"Validation Results - Epoch: {epoch}")

        for name, value in metrics.items():
            self.logger.info(f"  {name}: {value:.4f}")

            # Log to TensorBoard if available
            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar(f"validation/{name}", value, epoch)

        # Update metrics history
        for name, value in metrics.items():
            key = f"val_{name}"
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append((epoch, value))

    def log_model_summary(self, model_summary: str) -> None:
        """Log model architecture summary."""
        self.logger.info(f"Model Architecture:\n{model_summary}")

        # Also save to file
        model_file = os.path.join(self.log_dir, "model_architecture.txt")
        with open(model_file, "w") as f:
            f.write(model_summary)

    def log_exception(self, exception: Exception) -> None:
        """Log an exception."""
        self.logger.error(f"Exception occurred: {str(exception)}")
        self.logger.error(traceback.format_exc())

    def save_metrics_history(self) -> None:
        """Save the metrics history to a file."""
        metrics_file = os.path.join(self.log_dir, "metrics_history.json")
        with open(metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        self.logger.info(f"Metrics history saved to {metrics_file}")

    def close(self) -> None:
        """Close the logger and save final data."""
        self.save_metrics_history()

        if self.tensorboard_writer:
            self.tensorboard_writer.close()

        self.logger.info(f"MLLogger for experiment {self.experiment_name} closed")

        # Close all handlers
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)


def get_logger(
    log_dir: str = "./logs",
    experiment_name: Optional[str] = None,
    console_level: str = "info",
    file_level: str = "debug",
    use_tensorboard: bool = False,
) -> MLLogger:
    """Factory function to create and configure a logger."""
    return MLLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        console_level=console_level,
        file_level=file_level,
        use_tensorboard=use_tensorboard,
    )


def log_execution_time(logger, func_name=None):
    """Decorator to log the execution time of a function."""

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
