import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from src.train.trainers_base import BaseTrainer
from tqdm import tqdm

class ClassificationTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, **kwargs):
        # CrossEntropyLoss is used for multi-class classification
        self.criterion = nn.CrossEntropyLoss()
        # Allow override of optimizer/scheduler via kwargs; default is AdamW
        optimizer = kwargs.pop("optimizer", torch.optim.AdamW(model.parameters(), lr=kwargs.pop("initial_lr", 5e-5)))
        scheduler = kwargs.pop("scheduler", None)
        # Initialize common training routines from BaseTrainer
        super().__init__(model, train_loader, val_loader,
                         optimizer=optimizer, scheduler=scheduler, **kwargs)

    def _run_epoch(self, epoch_nr):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        # Standard tqdm progress bar for training epoch; refresh rate can be adjusted
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch_nr+1}", leave=False)

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            # Use encoded labels (converted to long) for classification targets
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
            # Compare predicted class indices with true labels
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += len(labels)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total if total > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}

    def _validate_epoch(self, epoch_nr):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch_nr+1}", leave=False)

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["encoded_label"].to(self.device).long()  # Ensure labels are of type long
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                correct += (logits.argmax(dim=-1) == labels).sum().item()
                total += len(labels)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}
