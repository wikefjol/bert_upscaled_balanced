#!/usr/bin/env python
import os
import json
import logging
import argparse
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel
from dotenv import load_dotenv
import wandb

from src.train.mlm_trainer import MLMTrainer
from src.data_module.data_tools import fasta2pandas
from src.data_module.dataset import MLMDataset
from src.utils.vocab import Vocabulary, KmerVocabConstructor
from src.factories.preprocessing_factory import create_preprocessor
from src.model.backbone import ModularBertax
from src.model.heads import MLMHead
from src.utils.config_utils import expand_config  # new unified config expansion

load_dotenv()

def nest_dotted_keys(flat_dict):
    """Convert dotted keys in a flat dict into a nested dict."""
    nested = {}
    for key, value in flat_dict.items():
        parts = key.split(".")
        current = nested
        for p in parts[:-1]:
            current = current.setdefault(p, {})
        if isinstance(value, dict):
            current[parts[-1]] = nest_dotted_keys(value)
        else:
            current[parts[-1]] = value
    return nested

def recursive_update(base, updates):
    """Deep-merge wandb updates into the base config."""
    for k, v in updates.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            recursive_update(base[k], v)
        else:
            base[k] = v
    return base

def create_lr_lambda(steps_per_epoch, warmup_epochs, plateau_epochs, decay_epochs, peak_lr, final_lr):
    """Create a piecewise LR schedule for warmup, plateau, and decay."""
    def lr_lambda(current_step):
        current_epoch = current_step / steps_per_epoch
        if current_epoch <= warmup_epochs:
            return current_epoch / warmup_epochs
        elif current_epoch <= warmup_epochs + plateau_epochs:
            return 1.0
        elif current_epoch <= warmup_epochs + plateau_epochs + decay_epochs:
            decay_progress = (current_epoch - warmup_epochs - plateau_epochs) / decay_epochs
            return 1.0 - decay_progress * (1.0 - (final_lr / peak_lr))
        else:
            return final_lr / peak_lr
    return lr_lambda

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the pretraining configuration JSON file")
    args = parser.parse_args()
    config_file = args.config

    logging.info(f"Loading configuration from {config_file}")
    with open(config_file, "r") as f:
        base_config = json.load(f)

    wandb.init(project=base_config.get("PROJECT_NAME", "default_project"), config=base_config)
    raw_updates = dict(wandb.config)
    updates_nested = nest_dotted_keys(raw_updates)
    config = recursive_update(base_config, updates_nested)
    # Expand config for pretraining
    config = expand_config(config, mode="pretrain")

    logging.basicConfig(level=config.get("system_log_lvl", logging.INFO))
    champs_dir = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/champs"

    logging.info("Building vocabulary")
    constructor = KmerVocabConstructor(k=config["preprocessing"]["tokenization"]["k"], alphabet=config["alphabet"])
    vocab = Vocabulary()
    vocab.build_from_constructor(constructor, data=[])

    logging.info("Building preprocessors")
    train_preprocessor = create_preprocessor(config["preprocessing"], vocab, training=True)
    val_preprocessor = create_preprocessor(config["preprocessing"], vocab, training=False)

    logging.info(f"Loading data from {config['DATA_PATH']}")
    all_data = fasta2pandas(config["DATA_PATH"])
    if config.get("small_set", False):
        all_data = all_data[:config.get("n_test", 10)]
    train_df, val_df = train_test_split(
        all_data,
        test_size=config.get("test_size", 0.1),
        random_state=42,
        shuffle=True
    )

    mlm_train_set = MLMDataset(
        df=train_df["sequence"],
        preprocessor=train_preprocessor,
        masking_percentage=config.get("masking_percentage", 0.15)
    )
    mlm_val_set = MLMDataset(
        df=val_df["sequence"],
        preprocessor=val_preprocessor,
        masking_percentage=config.get("masking_percentage", 0.15)
    )

    logging.info("Building model components")
    bert_cfg = BertConfig(
        vocab_size=len(vocab),
        hidden_size=config["model"]["hidden_size"],
        num_hidden_layers=config["model"]["num_layers"],
        num_attention_heads=config["model"]["num_attention_heads"],
        intermediate_size=config["model"]["intermediate_size"],
        max_position_embeddings=config["model"]["max_position_embeddings"],
        hidden_dropout_prob=config["model"]["dropout_rate"],
        attention_probs_dropout_prob=config["model"]["dropout_rate"]
    )
    encoder = BertModel(bert_cfg)
    mlm_head = MLMHead(
        in_features=config["model"]["hidden_size"],
        hidden_layer_size=(config["model"]["hidden_size"] + len(vocab)) // 2,
        out_features=len(vocab),
        dropout_rate=config["model"].get("mlm_dropout_rate", config["model"]["dropout_rate"])
    )
    model = ModularBertax(encoder=encoder, mlm_head=mlm_head, classification_head=None)
    model.mode = "pretrain"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logging.info("Setting up DataLoaders")
    batch_size = config["batch_size"]
    accumulation_steps = config.get("accumulation_steps", 1)
    num_workers = max(4, (os.cpu_count() or 1) // 4)
    print(f'K: {config["preprocessing"]["tokenization"]["k"]}, Batch size: {batch_size}, num_workers: {num_workers}')

    mlm_trainloader = DataLoader(
        mlm_train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    mlm_valloader = DataLoader(
        mlm_val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    steps_per_epoch = max(1, len(mlm_trainloader) // accumulation_steps)
    peak_lr = wandb.config.get("peak_lr", config["peak_lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr)

    lr_lambda = create_lr_lambda(
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=config.get("warmup_epochs", 3),
        plateau_epochs=config.get("plateau_epochs", 3),
        decay_epochs=config.get("decay_epochs", 3),
        peak_lr=config["peak_lr"],
        final_lr=config["peak_lr"] / 100
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    wandb.config.update({
        "preprocessing": {"tokenization": {"k": config["preprocessing"]["tokenization"]["k"]}},
        "batch_size": config["batch_size"],
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": config["batch_size"] * accumulation_steps,
        "peak_lr": peak_lr,
        "num_workers": num_workers
    }, allow_val_change=True)

    logging.info("Initializing MLMTrainer for pretraining")
    pretrainer = MLMTrainer(
        model=model,
        train_loader=mlm_trainloader,
        val_loader=mlm_valloader,
        num_epochs=config["max_epochs"]["pre_training"],
        patience=config["patience"]["pre_training"],
        metrics_jsonl_path="pretraining.jsonl",
        optimizer=optimizer,
        scheduler=scheduler,
        use_amp=True,
        accumulation_steps=accumulation_steps,
        champs_dir=champs_dir,
        earliest_champion_epoch=1,
        earliest_stop_epoch=2 * config.get("plateau_epochs", 1),
        config=config
    )

    logging.info("Starting pretraining")
    pretrainer.train()
    wandb.finish()

if __name__ == "__main__":
    main()
