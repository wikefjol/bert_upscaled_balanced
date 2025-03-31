import os
import re
import json
import matplotlib.pyplot as plt
import logging
import inspect
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Callable, Any

LOG_LEVELS = {
    "DL": 8,
    "DM": 9,
    "DH": 10,
    "I": logging.INFO,
    "W": logging.WARNING,
    "E": logging.ERROR,
    "C": logging.CRITICAL,
}

class CustomRotatingFileHandler(RotatingFileHandler):
    """
    A custom rotating file handler to name backups as `filename_<number>.log`.
    """
    def _rotate_filename(self, filename, count):
        """Helper method to construct rotated filenames."""
        base, ext = os.path.splitext(filename)
        return f"{base}_{count}{ext}"

    def doRollover(self):
        """
        Perform a rollover, renaming log files with the custom convention.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self._rotate_filename(self.baseFilename, i)
                dfn = self._rotate_filename(self.baseFilename, i + 1)
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            dfn = self._rotate_filename(self.baseFilename, 1)
            if os.path.exists(self.baseFilename):
                if os.path.exists(dfn):
                    os.remove(dfn)
                os.rename(self.baseFilename, dfn)

        if not self.delay:
            self.stream = self._open()

def safe_repr(value: Any, max_length: int = 800) -> str:
    """
    Safely represent a value for logging, truncating or summarizing large objects.
    """
    # If the object has a 'shape' attribute (e.g., DataFrame, NumPy array, tensor):
    if hasattr(value, "shape"):
        shape = getattr(value, "shape", None)
        return f"<{type(value).__name__} shape={shape}>"
    
    # Summarize large lists, dicts, tuples
    if isinstance(value, (list, dict, tuple)) and len(value) > max_length:
        return f"<{type(value).__name__} len={len(value)}>"

    # Truncate long strings
    if isinstance(value, str) and len(value) > max_length:
        return value[:max_length] + "... [truncated]"
    
    # Default: fallback to normal repr
    return repr(value)


def with_logging(level: int, max_length: int = 800) -> Callable[..., Any]:
    """
    Decorator to log the start and end of a function, including its module and class.
    Avoids logging large objects by summarizing or truncating them.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = logging.getLogger("system_logger")

            # Identify the module and function name
            module_name = func.__module__.upper()
            qual_name = func.__qualname__

            # Get function signature and bind arguments
            signature = inspect.signature(func)
            bound_arguments = signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            
            # Summarize arguments (ignoring self)
            parameters = ", ".join(
                f"{k}={safe_repr(v, max_length)}"
                for k, v in bound_arguments.arguments.items()
                if k != 'self'
            )

            # Build and log the "input" message
            padding_width = 60  # Keep padding fixed for consistent alignment
            log_message = f"[{module_name}] '{qual_name}'".ljust(padding_width) + f"input: {parameters}"
            logger.log(level, log_message)

            # Call the function
            result = func(*args, **kwargs)

            # Summarize and log the "output" message
            if result is not None:
                result_summary = safe_repr(result, max_length)
                result_message = f"[{module_name}] '{qual_name}'".ljust(padding_width) + f"output: {result_summary}"
                logger.log(level, result_message)

            return result
        return wrapper
    return decorator

def setup_logging(log_dir, system_level=logging.INFO, training_level=logging.INFO):
    """
    Set up system, pretraining, and finetuning loggers with specified levels and directory.
    """
    # Custom debug levels
    DEBUG_LOW = 8
    DEBUG_MEDIUM = 9
    DEBUG_HIGH = 10
    logging.addLevelName(DEBUG_LOW, "DEBUG --*")
    logging.addLevelName(DEBUG_MEDIUM, "DEBUG -*-")
    logging.addLevelName(DEBUG_HIGH, "DEBUG *--")

    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Clear existing handlers for system_logger
    system_logger = logging.getLogger("system_logger")
    while system_logger.handlers:
        system_logger.handlers.pop()

    # Clear existing handlers for pretraining_logger
    pretraining_logger = logging.getLogger("pretraining_logger")
    while pretraining_logger.handlers:
        pretraining_logger.handlers.pop()

    # Clear existing handlers for finetuning_logger
    finetuning_logger = logging.getLogger("finetuning_logger")
    while finetuning_logger.handlers:
        finetuning_logger.handlers.pop()


    # Set up system logger
    system_logger.setLevel(system_level)
    system_log_file = os.path.join(log_dir, "system.log")
    # Open file in write mode to overwrite existing content
    with open(system_log_file, 'w'):
        pass
    system_handler = CustomRotatingFileHandler(system_log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    system_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    system_handler.setFormatter(system_formatter)
    system_logger.addHandler(system_handler)

    # # Set up pretraining logger
    # pretraining_logger.setLevel(training_level)
    # pretraining_log_file = os.path.join(log_dir, "pretraining.log")
    # # Open file in write mode to overwrite existing content
    # with open(pretraining_log_file, 'w'):
    #     pass
    # pretraining_handler = CustomRotatingFileHandler(pretraining_log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    # pretraining_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    # pretraining_handler.setFormatter(pretraining_formatter)
    # pretraining_logger.addHandler(pretraining_handler)

    # # Set up finetuning logger
    # finetuning_logger.setLevel(training_level)
    # finetuning_log_file = os.path.join(log_dir, "finetuning.log")
    # # Open file in write mode to overwrite existing content
    # with open(finetuning_log_file, 'w'):
    #     pass
    # finetuning_handler = CustomRotatingFileHandler(finetuning_log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    # finetuning_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    # finetuning_handler.setFormatter(finetuning_formatter)
    # finetuning_logger.addHandler(finetuning_handler)

    return system_logger#, pretraining_logger, finetuning_logger


def plot_training_performance(
    metrics_file_name,
    run_name
):
    """
    Reads a JSONL file of training/validation metrics and produces a two-panel plot:
    Left: Train and Val Loss with Learning Rate
    Right: Train and Val Accuracy with Learning Rate (0-1 on Y-axis)

    Args:
        metrics_file_name (str): Name of the JSONL file containing training metrics.
        run_name (str): Name of the current run for the header and file naming.
    """
    log_dir = os.path.join("runs", run_name, "logs")  # Dynamic log directory
    output_file_name = os.path.splitext(metrics_file_name)[0] + ".png"  # Derived output file name
    file_path = os.path.join(log_dir, metrics_file_name)
    if not os.path.exists(file_path):
        print(f"Metrics file '{file_path}' not found.")
        return

    # Ensure unique output file name
    output_file_path = os.path.join(log_dir, f"{run_name}_{output_file_name}")

    # Lists to store metric values
    epochs = []
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []

    # Read each line of the JSONL file
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse the JSON line into a dict
            entry = json.loads(line)

            # Common keys for both pretraining and fine-tuning
            epoch = entry["epoch"]
            epochs.append(epoch)

            # Handle different formats for losses
            train_loss = entry.get("train_avg_loss") or entry.get("train_loss")
            val_loss = entry.get("val_avg_loss") or entry.get("val_loss")
            lr = entry.get("learning_rate", 0.0)  # Learning rate

            # Handle different formats for accuracy
            train_acc = entry.get("train_mlm_accuracy") or entry.get("train_accuracy") or 0.0
            val_acc = entry.get("val_mlm_accuracy") or entry.get("val_accuracy") or 0.0

            # Append the numeric values if they exist
            train_losses.append(train_loss if train_loss is not None else 0.0)
            val_losses.append(val_loss if val_loss is not None else 0.0)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            learning_rates.append(lr)

    # If no data found, just return
    if not epochs:
        print(f"No valid entries found in {file_path}.")
        return

    # Plot the results
    plt.figure(figsize=(14, 6))
    plt.suptitle(f"Training Performance for {run_name}", fontsize=16)

    # Subplot for loss with learning rate
    ax1 = plt.subplot(1, 2, 1)
    line1, = ax1.plot(epochs, train_losses, label="Train Loss")
    line2, = ax1.plot(epochs, val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.grid(True)

    # Add learning rate to the loss plot
    ax1_lr = ax1.twinx()
    line3, = ax1_lr.plot(epochs, learning_rates, label="Learning Rate", color="green", linestyle="--")
    ax1_lr.set_ylabel("Learning Rate")

    # Combine legends
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")

    # Subplot for accuracy with learning rate
    ax2 = plt.subplot(1, 2, 2)
    line4, = ax2.plot(epochs, train_accuracies, label="Train Accuracy")
    line5, = ax2.plot(epochs, val_accuracies, label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(-0.1, 1.1)  # Force accuracy to be 0-1
    ax2.set_title("Training and Validation Accuracy")
    ax2.grid(True)

    # Add learning rate to the accuracy plot
    ax2_lr = ax2.twinx()
    line6, = ax2_lr.plot(epochs, learning_rates, label="Learning Rate", color="green", linestyle="--")
    ax2_lr.set_ylabel("Learning Rate")

    # Combine legends
    lines = [line4, line5, line6]
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc="upper left")

    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file_path)
    plt.close()
    print(f"Training performance plot saved to `{output_file_path}`.")


def plot_finetuning_comparison(runs_dir= "runs"):
    """
    Scans each subfolder under 'runs/' for 'logs/finetuning.jsonl'.
    Parses the validation accuracy for each run and plots all on one chart.
    Saves the resulting comparison plot to 'runs/finetuning_comparison.png'.
    """
    all_run_names = []
    all_epochs = []
    all_val_accs = []

    # Traverse 'runs/' looking for subfolders with 'logs/finetuning.jsonl'
    for run_name in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, run_name)
        if not os.path.isdir(run_path):
            continue  # Skip files, only process directories
        
        logs_dir = os.path.join(run_path, "logs")
        if not os.path.isdir(logs_dir):
            continue

        finetuning_file = os.path.join(logs_dir, "finetuning.jsonl")
        if not os.path.exists(finetuning_file):
            continue  # Skip runs that don't have a finetuning file

        # Parse epoch and val_accuracy from the JSONL
        epochs = []
        val_accs = []
        with open(finetuning_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                epoch = entry["epoch"]
                val_accuracy = entry.get("val_accuracy", 0.0)

                epochs.append(epoch)
                val_accs.append(val_accuracy)

        # If valid data found, store for plotting
        if epochs:
            all_run_names.append(run_name)
            all_epochs.append(epochs)
            all_val_accs.append(val_accs)

    # If no runs found with finetuning data, just exit
    if not all_run_names:
        print("No finetuning.jsonl files found in the runs directory.")
        return

    # Plot all runs on a single figure
    plt.figure(figsize=(8, 6))
    for run_name, epochs, val_accs in zip(all_run_names, all_epochs, all_val_accs):
        plt.plot(epochs, val_accs, marker="o", label=run_name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.ylim(-0.1, 1.1)  # Force accuracy from 0 to 1
    plt.title("Finetuning Validation Accuracy Across Runs")
    plt.legend()
    plt.grid(True)

    comparison_plot_path = os.path.join(runs_dir, "finetuning_comparison.png")
    plt.tight_layout()
    plt.savefig(comparison_plot_path)
    plt.close()

    print(f"Comparison plot saved to '{comparison_plot_path}'")

def plot_pretraining_comparison(runs_dir):
    """
    Scans each subfolder under 'runs/' for 'logs/pretraining.jsonl'.
    Parses the validation MLM accuracy for each run and plots all on one chart.
    Saves the resulting comparison plot to 'runs/pretraining_comparison.png'.
    """
    all_run_names = []
    all_epochs = []
    all_val_mlm_accs = []

    # Traverse 'runs/' looking for subfolders with 'logs/pretraining.jsonl'
    for run_name in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, run_name)
        if not os.path.isdir(run_path):
            continue  # Skip files, only process directories

        logs_dir = os.path.join(run_path, "logs")
        if not os.path.isdir(logs_dir):
            continue

        pretraining_file = os.path.join(logs_dir, "pretraining.jsonl")
        if not os.path.exists(pretraining_file):
            continue  # Skip runs that don't have a pretraining file

        # Parse epoch and val_mlm_accuracy from the JSONL
        epochs = []
        val_mlm_accs = []
        with open(pretraining_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                epoch = entry["epoch"]
                val_mlm_accuracy = entry.get("val_mlm_accuracy", 0.0)

                epochs.append(epoch)
                val_mlm_accs.append(val_mlm_accuracy)

        # If valid data found, store for plotting
        if epochs:
            all_run_names.append(run_name)
            all_epochs.append(epochs)
            all_val_mlm_accs.append(val_mlm_accs)

    # If no runs found with pretraining data, just exit
    if not all_run_names:
        print("No pretraining.jsonl files found in the runs directory.")
        return

    # Plot all runs on a single figure
    plt.figure(figsize=(8, 6))
    for run_name, epochs, val_mlm_accs in zip(all_run_names, all_epochs, all_val_mlm_accs):
        plt.plot(epochs, val_mlm_accs, marker="o", label=run_name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation MLM Accuracy")
    plt.ylim(0, 1)  # Force accuracy from 0 to 1
    plt.title("Pretraining Validation MLM Accuracy Across Runs")
    plt.legend()
    plt.grid(True)

    comparison_plot_path = os.path.join(runs_dir, "pretraining_comparison.png")
    plt.tight_layout()
    plt.savefig(comparison_plot_path)
    plt.close()

    print(f"Comparison plot saved to '{comparison_plot_path}'")