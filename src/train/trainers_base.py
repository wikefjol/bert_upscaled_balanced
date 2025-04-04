# src/train/trainers_base.py

import os
import json
import time
import logging
import torch
import gc
from abc import ABC, abstractmethod
import wandb
from tqdm import tqdm
from datetime import datetime
import csv

from src.utils.champions import build_champion_key, save_champion, update_champion_metadata

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
          - Logging
          - Mixed precision
          - Distinct saving logic for pretraining vs. finetuning
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
        self.logger = logging.getLogger("BaseTrainer")

        self.min_delta = min_delta
        self.no_improvement_count = 0
        self.earliest_stop_epoch = earliest_stop_epoch
        self.config = kwargs.get("config", {})

        self.champs_dir = kwargs.get("champs_dir", "")
        self.earliest_champion_epoch = kwargs.get("earliest_champion_epoch", 0)

        # Pretrain champion metric: minimize val_loss
        # Finetune "best model": maximize val_accuracy
        if self.model.mode == "pretrain":
            self.champion_metric_name = "val_loss"
            self.best_metric = float("inf")
            self.compare_fn = lambda new, old: (old - new) > self.min_delta
        else:
            self.champion_metric_name = "val_accuracy"
            self.best_metric = float("-inf")
            self.compare_fn = lambda new, old: (new - old) > self.min_delta

        # Build champion key (only relevant for pretraining)
        self.champion_key = build_champion_key(self.model.mode, self.config)
        self.metadata_path = os.path.join(self.champs_dir, "champions.json")

        # Mixed precision scaler
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # For finetuning: we create one run folder per run. Overwrite best model each time it improves.
        self.run_folder = None
        self.finetune_dir = kwargs.get("finetune_dir", "./finetuned_models")

    @abstractmethod
    def _run_epoch(self, epoch_nr):
        """Implement the training loop for one epoch."""
        pass

    @abstractmethod
    def _validate_epoch(self, epoch_nr):
        """Implement the validation loop for one epoch."""
        pass

    def train(self):
        """
        Main training loop:
          - For 'pretrain', uses champion logic (compare val_loss, update champions.json).
          - For 'finetune', each epoch we compare val_accuracy vs. self.best_metric, 
            overwriting best_model.pt in a dedicated run folder if it improves.
        """
        task_prefix = "mlm" if self.model.mode == "pretrain" else "cls"

        for epoch in range(self.num_epochs):
            epoch_start = time.time()

            # --- Run epoch (training) ---
            train_metrics = self._run_epoch(epoch)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # If we have a scheduler, it may also be stepping epoch-wise
            # (If it steps per-batch, it's handled in _run_epoch or the loop.)
            # Typically not needed here unless you step once per epoch.
            #
            # if self.scheduler is not None:
            #     self.scheduler.step()

            # --- Optional Validation ---
            if (epoch + 1) % self.val_interval == 0:
                val_metrics = self._validate_epoch(epoch)
                val_loss = val_metrics.get("loss", float("inf"))
                val_accuracy = val_metrics.get("accuracy", 0.0)

                if self.model.mode == "pretrain":
                    # -------- Pretraining: Champion Logic --------
                    # Print minimal logging or mirror your existing approach
                    if (epoch % 5) == 0:  # Print a header every 5 epochs
                        hdr = ("Epoch  | Train Loss | Train Acc  | Val Loss   | Val Acc    | LR      ")
                        print(hdr)
                        print("-" * len(hdr))

                    print(
                        f"{epoch+1:5d}  | {train_metrics['loss']:.4f}   | {train_metrics['accuracy']:.4f}  "
                        f"| {val_loss:.4f}   | {val_accuracy:.4f}   | {current_lr:.6f}"
                    )

                    wandb.log({
                        f"train_{task_prefix}_loss": train_metrics["loss"],
                        f"train_{task_prefix}_accuracy": train_metrics["accuracy"],
                        f"val_{task_prefix}_loss": val_loss,
                        f"val_{task_prefix}_accuracy": val_accuracy,
                        "learning_rate": current_lr
                    })

                    # Pretraining champion = minimize val_loss
                    current_metric = val_loss
                    if (epoch + 1) >= self.earliest_champion_epoch:
                        if self.compare_fn(current_metric, self.best_metric):
                            self.best_metric = current_metric
                            self.no_improvement_count = 0

                            label_encoder = getattr(self.train_loader.dataset, "label_encoder", None)
                            checkpoint_path, config_path = save_champion(
                                self.model,
                                self.config,
                                epoch + 1,
                                self.champion_key,
                                self.champs_dir,
                                label_encoder
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
                            self.logger.info(
                                f"Champion updated for {self.champion_key} at epoch {epoch+1}"
                            )
                        else:
                            self.no_improvement_count += 1

                else:
                    # -------- Finetuning: Overwrite best model for *this run* --------
                    # Summarize epoch results
                    print(
                        f"Epoch {epoch+1}: "
                        f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                    )
                    wandb.log({
                        f"train_{task_prefix}_loss": train_metrics["loss"],
                        f"train_{task_prefix}_accuracy": train_metrics["accuracy"],
                        f"val_{task_prefix}_loss": val_loss,
                        f"val_{task_prefix}_accuracy": val_accuracy,
                        "learning_rate": current_lr
                    })

                    # For finetuning, we maximize val_accuracy
                    current_metric = val_accuracy
                    if (epoch + 1) >= self.earliest_champion_epoch:
                        if self.compare_fn(current_metric, self.best_metric):
                            # Found a new best for *this run*
                            self.best_metric = current_metric
                            self.no_improvement_count = 0

                            # Create run folder if not already done
                            if self.run_folder is None:
                                os.makedirs(self.finetune_dir, exist_ok=True)

                                kmer = self.config["preprocessing"]["tokenization"].get("k", 3)
                                num_heads = self.config["model"].get("num_attention_heads", 4)
                                modprob = self.config.get("modification_probability", 0.0)
                                modprob_str = f"{int(modprob*100):03d}"

                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                run_name = f"{kmer}mer_{num_heads}heads_{modprob_str}_{timestamp}"
                                self.run_folder = os.path.join(self.finetune_dir, run_name)
                                os.makedirs(self.run_folder, exist_ok=True)

                                # Save finetuning config
                                config_path = os.path.join(self.run_folder, "finetuning_config.json")
                                with open(config_path, "w") as f:
                                    json.dump(self.config, f, indent=2)

                                # If label encoder is present
                                if hasattr(self.train_loader.dataset, "label_encoder"):
                                    le = self.train_loader.dataset.label_encoder
                                    le_path = os.path.join(self.run_folder, "label_encoder.json")
                                    with open(le_path, "w") as f:
                                        json.dump({
                                            "label_to_index": le.label_to_index,
                                            "index_to_label": le.index_to_label
                                        }, f, indent=2)

                                # If vocab is present
                                if hasattr(self.train_loader.dataset, "vocab"):
                                    vocab = self.train_loader.dataset.vocab
                                    vocab_path = os.path.join(self.run_folder, "vocab.json")
                                    with open(vocab_path, "w") as f:
                                        json.dump(vocab.__dict__, f, indent=2)

                            # Overwrite the best-model checkpoint for this run
                            best_model_path = os.path.join(self.run_folder, "best_model.pt")
                            torch.save(self.model.state_dict(), best_model_path)

                            # Append to or create metrics.csv
                            metrics_csv = os.path.join(self.run_folder, "metrics.csv")
                            row = {
                                "epoch": epoch+1,
                                "val_accuracy": current_metric,
                                "val_loss": val_loss
                            }
                            file_exists = os.path.exists(metrics_csv)
                            with open(metrics_csv, "a", newline="") as csvfile:
                                writer = csv.DictWriter(csvfile, fieldnames=["epoch", "val_accuracy", "val_loss"])
                                if not file_exists:
                                    writer.writeheader()
                                writer.writerow(row)

                            self.logger.info(
                                f"[Finetuning] Best model updated at epoch {epoch+1}, saved to {best_model_path}"
                            )
                        else:
                            self.no_improvement_count += 1

            else:
                # If we skip validation (val_interval > 1), just log training metrics
                wandb.log({
                    f"train_{task_prefix}_loss": train_metrics["loss"],
                    f"train_{task_prefix}_accuracy": train_metrics["accuracy"],
                    "learning_rate": current_lr
                })

            # Summarize each epoch in logs
            self.logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train: {train_metrics}, LR: {current_lr:.6f}"
            )

            # Cleanup
            torch.cuda.empty_cache()
            gc.collect()

            # Early stopping check
            if (epoch + 1) >= self.earliest_stop_epoch and self.no_improvement_count >= self.patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
