import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from src.train.trainers_base import BaseTrainer
from sklearn.metrics import f1_score, balanced_accuracy_score
import numpy as np
from tqdm import tqdm

class ClassificationTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_loader,
        # CHANGES: val_loader is "closely", add val_loader_distantly for "distant" set
        val_loader,
        val_loader_distantly=None,
        optimizer=None,
        scheduler=None,
        **kwargs
    ):
        self.criterion = nn.CrossEntropyLoss()
        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,      # "closely" set becomes the base-trainer's official val_loader
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs
        )
        self.val_loader_distantly = val_loader_distantly

    def _run_epoch(self, epoch_nr):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch_nr+1}", leave=False)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["encoded_label"].to(self.device).long()
            
            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += len(labels)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total if total > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}

    def _validate_epoch(self, epoch_nr):
        metrics_closely = self._eval_single_val_loader(self.val_loader)
        # If we don’t have a second loader, skip it
        if self.val_loader_distantly is not None:
            metrics_distantly = self._eval_single_val_loader(self.val_loader_distantly)
        else:
            metrics_distantly = {}

        # Merge them: champion logic sees .get("loss") and .get("accuracy") from the closely set
        # The “distant” ones are stored under “dist_*”
        combined = {
            "loss": metrics_closely["loss"],
            "accuracy": metrics_closely["accuracy"],
            "macro_f1": metrics_closely["macro_f1"],
            "weighted_f1": metrics_closely["weighted_f1"],
            "balanced_accuracy": metrics_closely["balanced_accuracy"],

            # Add "dist_" prefix for the second dataset
            "dist_loss": metrics_distantly.get("loss", 0.0),
            "dist_accuracy": metrics_distantly.get("accuracy", 0.0),
            "dist_macro_f1": metrics_distantly.get("macro_f1", 0.0),
            "dist_weighted_f1": metrics_distantly.get("weighted_f1", 0.0),
            "dist_balanced_accuracy": metrics_distantly.get("balanced_accuracy", 0.0),
        }
        return combined

    def _eval_single_val_loader(self, loader):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["encoded_label"].to(self.device).long()
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "balanced_accuracy": bal_acc,
        }