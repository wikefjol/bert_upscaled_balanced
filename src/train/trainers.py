
import json
import time
import logging
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from src.utils.updates import send_discord_notification

from torch.utils.data import DataLoader

class MLMtrainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 best_weight_save_path: str = "best_pre_train_weights.pt",
                 metrics_jsonl_path: str = "pretraining.jsonl",
                 num_epochs: int = 50,
                 warmup_ratio: float = 0.2,
                 initial_lr: float = 5e-5,
                 no_scheduling: bool = False,  # Fixed syntax
                 patience: int = 10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_weight_save_path = best_weight_save_path
        self.metrics_jsonl_path = metrics_jsonl_path
        self.num_epochs = num_epochs
        self.patience = patience
        self.no_scheduling = no_scheduling  # Store flag properly

        self.system_logger = logging.getLogger("system_logger")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize optimizer & loss
        self.criteronMLM = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=initial_lr)

        # Set up warmup + linear decay if scheduling is enabled
        total_steps = len(self.train_loader) * self.num_epochs
        warmup_steps = int(warmup_ratio * total_steps)

        self.scheduler = None
        if not self.no_scheduling:  # Use scheduler if not disabled
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

        self.best_val_loss = float("inf")
        self.global_step = 0  # Count how many batches have been processed overall

    def _run_epoch(self, epoch_nr):
        self.model.train()
        total_loss, mlm_correct, total_mlm = 0.0, 0, 0
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch_nr + 1}", leave=False)
        ignore_index = self.train_loader.dataset.ignore_index

        for idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            mlm_labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            mlm_logits = self.model(input_ids, attention_mask)

            # Compute MLM loss
            loss = self.criteronMLM(
                mlm_logits.view(-1, self.model.mlm_head.out_features),
                mlm_labels.view(-1)
            )
            loss.backward()
            self.optimizer.step()

            # Update LR with scheduler only if it's enabled
            if self.scheduler is not None:
                self.scheduler.step()
                self.global_step += 1

            # Accumulate total loss
            total_loss += loss.item()

            # Compute MLM accuracy
            mlm_preds = mlm_logits.argmax(dim=-1)
            mlm_correct += (mlm_preds == mlm_labels).masked_select(mlm_labels != ignore_index).sum().item()
            total_mlm += (mlm_labels != ignore_index).sum().item()

            

        avg_loss = total_loss / len(self.train_loader)
        mlm_acc = mlm_correct / total_mlm if total_mlm > 0 else 0.0
        wandb.log({"train_loss": avg_loss, "mlm_acc": mlm_acc, "epoch": epoch_nr})
        return avg_loss, mlm_acc

    def _validate_epoch(self, epoch_nr):
        self.model.eval()
        total_loss, mlm_correct, total_mlm = 0.0, 0, 0
        progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch_nr + 1}", leave=False)
        ignore_index = self.val_loader.dataset.ignore_index

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                mlm_labels = batch["labels"].to(self.device)

                mlm_logits = self.model(input_ids, attention_mask)
                loss = self.criteronMLM(
                    mlm_logits.view(-1, self.model.mlm_head.out_features),
                    mlm_labels.view(-1)
                )
                total_loss += loss.item()

                mlm_preds = mlm_logits.argmax(dim=-1)
                mlm_correct += (mlm_preds == mlm_labels).masked_select(mlm_labels != ignore_index).sum().item()
                total_mlm += (mlm_labels != ignore_index).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        mlm_acc = mlm_correct / total_mlm if total_mlm > 0 else 0.0
        return avg_loss, mlm_acc

    def train(self):
        no_improvement_count = 0  # Track epochs with no improvement
        for epoch_nr in range(self.num_epochs):
            train_avg_loss, train_mlm_acc = self._run_epoch(epoch_nr)
            val_avg_loss, val_mlm_acc = self._validate_epoch(epoch_nr)

            # Check for best weights
            if val_avg_loss < self.best_val_loss:
                self.best_val_loss = val_avg_loss
                torch.save(self.model.state_dict(), self.best_weight_save_path)
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Get current learning rate
            if self.scheduler is not None:
                current_lr = self.optimizer.param_groups[0]["lr"]
            else:
                current_lr = self.optimizer.defaults["lr"]

            print(f"\nEpoch {epoch_nr + 1}/{self.num_epochs}")
            print(f"Train Avg Loss: {train_avg_loss:.6f}, Train MLM Accuracy: {train_mlm_acc:.6f}")
            print(f"Val Avg Loss: {val_avg_loss:.6f}, Val MLM Accuracy: {val_mlm_acc:.6f}")
            print(f"Current LR: {current_lr:.8f}")

            # Log metrics
            metrics_entry = {
                "epoch": epoch_nr + 1,
                "train_avg_loss": float(train_avg_loss),
                "train_mlm_accuracy": float(train_mlm_acc),
                "val_avg_loss": float(val_avg_loss),
                "val_mlm_accuracy": float(val_mlm_acc),
                "learning_rate": float(current_lr),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }

            # message = (
            #     f"MLM epoch `{epoch_nr + 1}/{self.num_epochs}`: "
            #     f"`train_acc = {float(train_mlm_acc):.4f}`, `val_acc = {float(val_mlm_acc):.4f}`."
            # )
            # send_discord_notification(None, message)
            with open(self.metrics_jsonl_path, 'a') as f:
                f.write(json.dumps(metrics_entry) + "\n")

            # Early stopping check
            if no_improvement_count >= self.patience:
                print(f"Early stopping triggered. No improvement in val loss for {self.patience} consecutive epochs.")
                break



class ClassificationTrainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 best_weight_save_path: str = "best_finetuning_weights.pt",
                 metrics_jsonl_path: str = "finetuning.jsonl",
                 num_epochs: int = 10,
                 warmup_ratio: float = 0.2,
                 initial_lr: float = 5e-5,
                 no_scheduling: bool = False,  # Fixed syntax
                 patience: int = 3):  # <-- Add a patience parameter here as well
        """
        A classification trainer that includes early stopping if the validation loss 
        fails to improve for 'patience' consecutive epochs.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_weight_save_path = best_weight_save_path
        self.metrics_jsonl_path = metrics_jsonl_path
        self.num_epochs = num_epochs
        self.patience = patience  # <-- store patience
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=initial_lr)

        # Scheduler (warmup + linear decay)
        total_steps = len(self.train_loader) * self.num_epochs
        warmup_steps = int(warmup_ratio * total_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        self.best_val_loss = float("inf")
        self.global_step = 0
        
        # Ensure we start with a clean JSONL file
        with open(self.metrics_jsonl_path, 'w'):
            pass

    def _run_epoch(self, epoch_nr):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch_nr + 1}", leave=False)

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["encoded_label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += len(labels)

        avg_loss = total_loss / len(self.train_loader)
        acc = correct / total if total > 0 else 0.0
        return avg_loss, acc

    def _validate_epoch(self, epoch_nr):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch_nr + 1}", leave=False)

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["encoded_label"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                correct += (logits.argmax(dim=-1) == labels).sum().item()
                total += len(labels)

        avg_loss = total_loss / len(self.val_loader)
        acc = correct / total if total > 0 else 0.0
        return avg_loss, acc

    def train(self):
        """
        Main training loop with early stopping based on validation loss.
        If validation loss does not improve for 'patience' consecutive epochs, stop early.
        """
        no_improvement_count = 0  # <-- tracks epochs with no improvement
        
        for epoch_nr in range(self.num_epochs):
            train_loss, train_acc = self._run_epoch(epoch_nr)
            val_loss, val_acc = self._validate_epoch(epoch_nr)

            # Check if validation loss improves
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_weight_save_path)
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Print status
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch_nr + 1}/{self.num_epochs}")
            print(f"Train Loss: {train_loss:.6f}, Train Accuracy: {train_acc:.6f}")
            print(f"Val Loss: {val_loss:.6f},   Val Accuracy: {val_acc:.6f}")
            print(f"Current LR: {current_lr:.8f}")

            # Log metrics
            metrics_entry = {
                "epoch": epoch_nr + 1,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
                "learning_rate": float(current_lr),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }

            
            # if epoch_nr%5:
            #     message = (
            #     f"Classification epoch `{epoch_nr + 1}/{self.num_epochs}`: "
            #     f"`train_acc = {float(train_acc):.4f}`, `val_acc = {float(val_acc):.4f}`."
            #     )
            #     send_discord_notification(None, message)

            with open(self.metrics_jsonl_path, 'a') as f:
                f.write(json.dumps(metrics_entry) + "\n")

            # Early stopping check
            if no_improvement_count >= self.patience:
                print(f"Early stopping triggered. No improvement in val loss for {self.patience} consecutive epochs.")
                break