# base_trainer.py:

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

        # Decide champion metric based on mode
        if self.model.mode == "pretrain":
            self.champion_metric_name = "val_loss"
            self.best_metric = float("inf")
            self.compare_fn = lambda new, old: (old - new) > self.min_delta
        else:  # e.g. self.model.mode == "classify"
            self.champion_metric_name = "val_accuracy"
            self.best_metric = float("-inf")
            self.compare_fn = lambda new, old: (new - old) > self.min_delta

        # Build champion_key once, e.g. "pretrain_k2_overlap_..."
        self.champion_key = build_champion_key(self.model.mode, self.config)
        # Path to JSON where we store champion metadata
        self.metadata_path = os.path.join(self.champs_dir, "champions.json")

        # Mixed precision
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
        Optionally more keys, e.g. {"macro_f1": float, "balanced_accuracy": float, ...}.
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
        task_prefix = "mlm" if self.model.mode == "pretrain" else "cls"

        for epoch in range(self.num_epochs):
            train_start = time.time()
            train_metrics = self._run_epoch(epoch)
            train_end = time.time()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Validation every self.val_interval epochs
            if (epoch + 1) % self.val_interval == 0:
                # VRAM usage
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
                val_loss_dist = val_metrics.get("dist_loss", float("inf"))
                val_accuracy_dist = val_metrics.get("dist_accuracy", 0.0)

                val_macro_f1 = val_metrics.get("macro_f1", None)
                val_weighted_f1 = val_metrics.get("weighted_f1", None)
                val_bal_acc = val_metrics.get("balanced_accuracy", None)

                val_macro_f1_dist = val_metrics.get("dist_macro_f1", None)
                val_weighted_f1_dist = val_metrics.get("dist_weighted_f1", None)
                val_bal_acc_dist = val_metrics.get("dist_balanced_accuracy", None)


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

                # Construct row for main metrics
                row = (
                    f"{epoch+1:<6} | {train_metrics['loss']:<10.4f} | {train_metrics['accuracy']:<10.4f} | "
                    f"{val_loss:<10.4f} | {val_accuracy:<10.4f} | {current_lr:<8.6f} | "
                    f"{(val_end - train_start):<14.2f} | {vram_percent:<10.2f}%"
                )
                row += f" | distLoss: {val_loss_dist:.4f} | distAcc: {val_accuracy_dist:.4f}"
                

                # Optionally append extra classification metrics on the same line
                if val_macro_f1 is not None:
                    row += f" | macroF1: {val_macro_f1:.4f}"
                if val_weighted_f1 is not None:
                    row += f" | weightedF1: {val_weighted_f1:.4f}"
                if val_bal_acc is not None:
                    row += f" | balAcc: {val_bal_acc:.4f}"
                if val_macro_f1_dist is not None:
                    row += f" | distMacroF1: {val_macro_f1_dist:.4f}"
                if val_weighted_f1_dist is not None:
                    row += f" | distWeightedF1: {val_weighted_f1_dist:.4f}"
                if val_bal_acc_dist is not None:
                    row += f" | distBalAcc: {val_bal_acc_dist:.4f}"

                print(row)

                # Prepare WandB log dictionary

                grad_norm = val_metrics.get("grad_norm", -1)
                wb_dict = {
                    f"train_{task_prefix}_loss": train_metrics["loss"],
                    f"train_{task_prefix}_accuracy": train_metrics["accuracy"],
                    f"val_{task_prefix}_loss": val_loss,
                    f"val_{task_prefix}_accuracy": val_accuracy,
                    "learning_rate": current_lr,
                }
                wb_dict[f"val_dist_{task_prefix}_loss"] = val_loss_dist
                wb_dict[f"val_dist_{task_prefix}_accuracy"] = val_accuracy_dist
                # Add classification metrics if present
                if val_macro_f1 is not None:
                    wb_dict[f"val_{task_prefix}_macro_f1"] = val_macro_f1
                if val_weighted_f1 is not None:
                    wb_dict[f"val_{task_prefix}_weighted_f1"] = val_weighted_f1
                if val_bal_acc is not None:
                    wb_dict[f"val_{task_prefix}_balanced_acc"] = val_bal_acc
                if val_macro_f1_dist is not None:
                    wb_dict[f"val_dist_{task_prefix}_macro_f1"] = val_macro_f1_dist

                # Log to Weights & Biases
                wandb.log(wb_dict)

                # Determine champion metric
                if self.model.mode == "pretrain":
                    current_metric = val_loss
                else:
                    current_metric = val_accuracy

                # Champion logic
                if (epoch + 1) >= self.earliest_champion_epoch:
                    if self.compare_fn(current_metric, self.best_metric):
                        self.best_metric = current_metric
                        self.no_improvement_count = 0
                        checkpoint_path, config_path = save_champion(
                            self.model,
                            self.config,
                            epoch + 1,
                            self.champion_key,
                            self.champs_dir
                        )
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
                        self.no_improvement_count += 1

            else:
                # If we skip validation, just log training metrics to WandB
                grad_norm = -2
                wandb.log({
                    f"train_{task_prefix}_loss": train_metrics["loss"],
                    f"train_{task_prefix}_accuracy": train_metrics["accuracy"],
                    "learning_rate": current_lr,
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

            # Early stopping
            if (
                (epoch + 1) % self.val_interval == 0
                and (epoch + 1) >= self.earliest_stop_epoch
                and self.no_improvement_count >= self.patience
            ):
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
