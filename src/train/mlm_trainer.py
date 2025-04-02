import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import get_cosine_schedule_with_warmup
from src.train.trainers_base import BaseTrainer
from tqdm import tqdm

# Enable cuDNN autotuning to select optimal algorithms for consistent input shapes
cudnn.benchmark = True

class MLMTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, accumulation_steps=1, **kwargs):
        """
        accumulation_steps: Number of sub-batches to accumulate before performing optimizer.step().
        """
        self.accumulation_steps = accumulation_steps
        # CrossEntropyLoss automatically ignores target indices set to ignore_index
        self.criterion = nn.CrossEntropyLoss()

        # Allow optimizer and scheduler to be passed via kwargs; otherwise, default to AdamW
        optimizer = kwargs.pop("optimizer", 
            torch.optim.AdamW(model.parameters(), lr=kwargs.pop("initial_lr", 5e-5))
        )
        scheduler = kwargs.pop("scheduler", None)

        # Initialize BaseTrainer with common training components
        super().__init__(model, train_loader, val_loader,
                         optimizer=optimizer, scheduler=scheduler, **kwargs)

    def _run_epoch(self, epoch_nr):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        # Use tqdm with a reduced refresh rate for efficiency in long epochs
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch_nr+1}",
                            leave=False, mininterval=5)
        # Use dataset-provided ignore_index to filter out non-contributing tokens
        ignore_index = self.train_loader.dataset.ignore_index

        for i, batch in enumerate(progress_bar):
            inp = batch["input_ids"].to(self.device, non_blocking=True)
            msk = batch["attention_mask"].to(self.device, non_blocking=True)
            lbl = batch["labels"].to(self.device, non_blocking=True)

            if self.use_amp:
                # Automatic mixed precision block for improved performance on compatible hardware
                with torch.cuda.amp.autocast():
                    logits = self.model(inp, msk)
                    # Flatten logits and labels to compute token-level loss
                    loss = self.criterion(logits.view(-1, logits.size(-1)), lbl.view(-1))
                self.scaler.scale(loss).backward()
            else:
                logits = self.model(inp, msk)
                loss = self.criterion(logits.view(-1, logits.size(-1)), lbl.view(-1))
                loss.backward()
###########################################################################
            ###### TEMOPRARY DEBUG SHIT###
            if self.use_amp:
                # Unscale gradients to get true values
                self.scaler.unscale_(self.optimizer)
            # Compute total gradient norm
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            #print(f"Gradient norm before optimizer step: {total_norm:.4f}")
##########################################################################################

            # Gradient accumulation: perform optimizer step every accumulation_steps iterations
            if (i + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Reset gradients; set_to_none can improve memory efficiency
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()

            total_loss += loss.item()
            # Compute predictions and count correct ones (ignoring positions set by ignore_index)
            preds = logits.argmax(dim=-1)
            correct += (preds == lbl).masked_select(lbl != ignore_index).sum().item()
            total += (lbl != ignore_index).sum().item()

        # Compute average loss and accuracy over the epoch
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total if total > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy, "grad_norm": total_norm}

    def _validate_epoch(self, epoch_nr):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch_nr+1}",
                            leave=False, mininterval=5)
        ignore_index = self.val_loader.dataset.ignore_index

        with torch.no_grad():
            for batch in progress_bar:
                inp = batch["input_ids"].to(self.device, non_blocking=True)
                msk = batch["attention_mask"].to(self.device, non_blocking=True)
                lbl = batch["labels"].to(self.device, non_blocking=True)

                logits = self.model(inp, msk)
                loss = self.criterion(logits.view(-1, logits.size(-1)), lbl.view(-1))
                total_loss += loss.item()

                preds = logits.argmax(dim=-1)
                correct += (preds == lbl).masked_select(lbl != ignore_index).sum().item()
                total += (lbl != ignore_index).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}
