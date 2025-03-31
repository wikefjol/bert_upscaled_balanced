import os
import json
import time
import logging
import torch
import gc
from abc import ABC, abstractmethod
import wandb
from tqdm import tqdm

from src.utils.champions import (
    build_champion_key,
    save_champion,
    update_champion_metadata
)

class BaseTrainer(ABC):
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        num_epochs,
        patience,
        metrics_jsonl_path,
        optimizer,
        scheduler=None,
        use_amp=False,
        earliest_stop_epoch=10,
        val_interval=1,
        min_delta=1e-4,
        **kwargs
    ):
        """
        BaseTrainer handles:
        - Core training loop
        - Early stopping
        - Champion saving (best model checkpoints)
        - AMP (automatic mixed precision)
        - Logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.patience = patience
        self.metrics_jsonl_path = metrics_jsonl_path
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.val_interval = val_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.global_step = 0
        self.logger = logging.getLogger("BaseTrainer")
        self.min_delta = min_delta
        self.no_improvement_count = 0

        # For champion/tracking
        self.champs_dir = kwargs.get("champs_dir", "")
        self.earliest_champion_epoch = kwargs.get("earliest_champion_epoch", 0)
        self.earliest_stop_epoch = earliest_stop_epoch
        self.config = kwargs.get("config", {})

        # Use model.mode == 'pretrain' or 'finetune' to decide champion metric
        # Pretrain => val_loss (lower is better)
        # Finetune => val_accuracy (higher is better)
        if self.model.mode == "pretrain":
            self.champion_metric_name = "val_loss"
            self.best_metric = float("inf")
            self.compare_fn = lambda new, old: (old - new) > self.min_delta
        else:  # self.model.mode == "classify"
            self.champion_metric_name = "val_accuracy"
            self.best_metric = float("-inf")
            self.compare_fn = lambda new, old: (new - old) > self.min_delta

        # Build champion_key once, e.g. "pretrain_k2_overlap_10layers_8heads_512hidden_2048intermediate"
        self.champion_key = build_champion_key(self.model.mode, self.config)
        # Path to JSON where we store champion metadata
        self.metadata_path = os.path.join(self.champs_dir, "champions.json")

        # Mixed precision scaler
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    @abstractmethod
    def _run_epoch(self, epoch_nr):
        """
        Implement the training loop for one epoch.
        Return a dict with at least {"loss": float, "accuracy": float}.
        """
        pass

    @abstractmethod
    def _validate_epoch(self, epoch_nr):
        """
        Implement the validation loop for one epoch.
        Return a dict with at least {"loss": float, "accuracy": float}.
        """
        pass

    def train(self):
        """
        Main training loop:
          - training epochs
          - optional validation checks
          - champion logic
          - early stopping
        """
        # 'mlm' or 'cls' prefix for logging
        task_prefix = "mlm" if self.model.mode == "pretrain" else "cls"

        for epoch in range(self.num_epochs):
            train_start = time.time()
            train_metrics = self._run_epoch(epoch)
            train_end = time.time()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Validation every self.val_interval epochs
            if (epoch + 1) % self.val_interval == 0:
                # VRAM usage (CUDA only)
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    allocated = torch.cuda.memory_allocated(device)
                    total = torch.cuda.get_device_properties(device).total_memory
                    vram_percent = allocated / total * 100
                else:
                    vram_percent = 0.0

                val_start = time.time()
                val_metrics = self._validate_epoch(epoch)
                val_end = time.time()

                val_loss = val_metrics.get("loss", float("inf"))
                val_accuracy = val_metrics.get("accuracy", 0.0)
                grad_norm = val_metrics.get("grad_norm", -1)

                # Print header every 5 validation cycles
                if (epoch % 5) == 0:
                    print("")
                    header = (
                        f"{'Epoch':<6} | {'Train Loss':<10} | {'Train Acc':<10} | "
                        f"{'Val Loss':<10} | {'Val Acc':<10} | {'LR':<8} | "
                        f"{'Total Time (s)':<14} | {'VRAM':<10}"
                    )
                    print(header)
                    print("-" * len(header))

                row = (
                    f"{epoch+1:<6} | {train_metrics['loss']:<10.4f} | {train_metrics['accuracy']:<10.4f} | "
                    f"{val_loss:<10.4f} | {val_accuracy:<10.4f} | {current_lr:<8.6f} | "
                    f"{(val_end - train_start):<14.2f} | {vram_percent:<10.2f}%"
                )
                print(row)

                # Log to Weights & Biases
                wandb.log({
                    "epoch": epoch + 1,
                    f"train_{task_prefix}_loss": train_metrics["loss"],
                    f"train_{task_prefix}_accuracy": train_metrics["accuracy"],
                    f"val_{task_prefix}_loss": val_loss,
                    f"val_{task_prefix}_accuracy": val_accuracy,
                    "learning_rate": current_lr,
                    "grad_norm": grad_norm
                })

                # Determine champion metric for this epoch
                if self.model.mode == "pretrain":
                    # use val_loss
                    current_metric = val_loss
                else:
                    # finetune => val_accuracy
                    current_metric = val_accuracy

                # Check improvement if we've reached earliest_champion_epoch
                if (epoch + 1) >= self.earliest_champion_epoch:
                    if self.compare_fn(current_metric, self.best_metric):
                        # We have a new best
                        self.best_metric = current_metric
                        self.no_improvement_count = 0

                        # Save champion
                        checkpoint_path, config_path = save_champion(
                            self.model,
                            self.config,
                            epoch + 1,
                            self.champion_key,
                            self.champs_dir
                        )
                        # Update champion metadata
                        update_champion_metadata(
                            metadata_path=self.metadata_path,
                            champion_key=self.champion_key,
                            new_val_metric=current_metric,
                            new_checkpoint_path=checkpoint_path,
                            new_config_path=config_path,
                            epoch=epoch + 1,
                            champion_metric_name=self.champion_metric_name,
                            mode=self.model.mode
                        )
                        self.logger.info(f"Champion updated for {self.champion_key} at epoch {epoch+1}")
                    else:
                        # No improvement
                        self.no_improvement_count += 1

            else:
                # If we skip validation, just log training metrics
                grad_norm = -2
                wandb.log({
                    "epoch": epoch + 1,
                    f"train_{task_prefix}_loss": train_metrics["loss"],
                    f"train_{task_prefix}_accuracy": train_metrics["accuracy"],
                    "learning_rate": current_lr,
                    "grad_norm": grad_norm
                })
                self.logger.info(
                    f"Epoch {epoch+1}: Training metrics logged; validation skipped."
                )

            # Logger summary
            if (epoch + 1) % self.val_interval == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs}: "
                    f"Train: {train_metrics}, Val: {val_metrics}, LR: {current_lr:.6f}"
                )
            else:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs}: Train: {train_metrics}, LR: {current_lr:.6f}"
                )

            torch.cuda.empty_cache()
            gc.collect()

            # --- Early Stopping Condition ---
            # Only check on validation epochs + after earliest_stop_epoch
            if (
                (epoch + 1) % self.val_interval == 0
                and (epoch + 1) >= self.earliest_stop_epoch
                and self.no_improvement_count >= self.patience
            ):
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
