#!/usr/bin/env python
import os
import json
import logging
import argparse
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel
from dotenv import load_dotenv
import numpy as np
from torch.utils.data import WeightedRandomSampler

import wandb

from src.train.classification_trainer import ClassificationTrainer
from src.data_module.data_tools import fasta2pandas
from src.data_module.dataset import ClassificationDataset
from src.utils.vocab import Vocabulary, KmerVocabConstructor
from src.factories.preprocessing_factory import create_preprocessor
from src.model.backbone import ModularBertax
from src.model.heads import SingleClassHead
from src.model.encoders import LabelEncoder
from src.utils.config_utils import expand_config  # new unified config expansion

load_dotenv()

CHAMPS_DIR = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/champs"
CHAMPIONS_FILE = os.path.join(CHAMPS_DIR, "champions.json")

def load_pretrained_encoder(config):
    """
    Loads the best pretraining checkpoint for the config's model parameters.
    Uses the champion logic from pretraining.
    """
    from src.utils.champions import build_champion_key
    champion_key = build_champion_key("pretrain", config)
    if not os.path.exists(CHAMPIONS_FILE):
        raise ValueError(f"No champions.json found at {CHAMPIONS_FILE}")
    with open(CHAMPIONS_FILE, "r") as f:
        champs_data = json.load(f)
    if champion_key not in champs_data or "entries" not in champs_data[champion_key]:
        raise ValueError(f"No champion data found for '{champion_key}' in {CHAMPIONS_FILE}")
    best_entry = champs_data[champion_key]["entries"][0]
    checkpoint_path = best_entry["checkpoint_path"]
    logging.info(f"Resolved champion checkpoint at {checkpoint_path} for key={champion_key}")
    return checkpoint_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to finetuning config JSON")
    args = parser.parse_args()

    # Define paths for training and validation CSV files
    train_path = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/data/final_train_set.csv"
    val_path_closely = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/data/closely_related_val_set.csv" 
    val_path_distantly = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/data/distantly_related_val_set.csv" 

    # Load and expand config for finetuning
    with open(args.config, "r") as f:
        config = json.load(f)
    config = expand_config(config, mode="finetune")

    logging.basicConfig(level=config.get("system_log_lvl", logging.INFO))
    wandb.init(project=config.get("PROJECT_NAME", "default_project"), config=config)

    logging.info("Building vocabulary")
    k_val = config["preprocessing"]["tokenization"]["k"]
    constructor = KmerVocabConstructor(k=k_val, alphabet=config["alphabet"])
    vocab = Vocabulary()
    vocab.build_from_constructor(constructor, data=[])

    logging.info("Building preprocessor for finetuning")
    preprocessor = create_preprocessor(config["preprocessing"], vocab, training=True)

    train_df = pd.read_csv(train_path)
    val_df_closely = pd.read_csv(val_path_closely)
    val_df_distantly = pd.read_csv(val_path_distantly)

    if config["small_set"]:
        small_set_size = config["n_test"]
        ratio_close = len(val_df_closely) / len(train_df)
        ratio_dist = len(val_df_distantly) / len(train_df)
        train_df = train_df.sample(n=small_set_size, random_state=42)
        val_df_closely = val_df_closely.sample(n=int(round(small_set_size * ratio_close)), random_state=42)
        val_df_distantly = val_df_distantly.sample(n=int(round(small_set_size * ratio_dist)), random_state=42)

    target_cols = config.get("target_labels", [])
    if isinstance(target_cols, list) and len(target_cols) == 1:
        target_col = target_cols[0]
        combined_df = pd.concat([train_df, val_df_closely, val_df_distantly], ignore_index=True)
        num_classes = combined_df[target_col].nunique()
        config.setdefault("model", {})["num_classes"] = num_classes
    else:
        target_col = target_cols if target_cols else "label"
        config.setdefault("model", {})["num_classes"] = 2
        logging.info("Multiple/ambiguous target labels not fully supported; defaulting num_classes=2.")

    label_encoder = LabelEncoder(labels=combined_df[target_col])

    class_counts = combined_df[target_col].value_counts().to_dict()
    class_weights = {cls: 1.0/(count + 1) for cls, count in class_counts.items()}
    sample_weights = train_df[target_col].map(class_weights).tolist()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_df), replacement=True)

    train_set = ClassificationDataset(
        train_df, preprocessor=preprocessor, label_encoder=label_encoder, target_column=target_col
    )
    val_set_closely = ClassificationDataset(
        val_df_closely, preprocessor=preprocessor, label_encoder=label_encoder, target_column=target_col
    )
    val_set_distantly = ClassificationDataset(
        val_df_distantly, preprocessor=preprocessor, label_encoder=label_encoder, target_column=target_col
    )

    batch_size = config["batch_size"]
    num_workers = min(4, (os.cpu_count() or 1) // 2)
    trainloader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    valloader_closely = DataLoader(val_set_closely, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    valloader_distantly = DataLoader(val_set_distantly, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    logging.info("Building encoder for finetuning")
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
    checkpoint_path = load_pretrained_encoder(config)
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    def strip_prefix(sd, prefix="bert."):
        return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items() }

    state_dict = strip_prefix(state_dict, "bert.")
    encoder_state = encoder.state_dict()
    filtered_sd = {k: v for k, v in state_dict.items() if k in encoder_state}
    encoder.load_state_dict(filtered_sd, strict=False)

    num_classes = config["model"]["num_classes"]
    classification_head = SingleClassHead(
        in_features=config["model"]["hidden_size"],
        hidden_layer_size=(config["model"]["hidden_size"] + num_classes) // 2,
        out_features=num_classes,
        dropout_rate=config["model"]["dropout_rate"]
    )

    model = ModularBertax(encoder=encoder, mlm_head=None, classification_head=classification_head)
    model.classifyMode()  # Mark mode for finetuning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    peak_lr = config["peak_lr"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr)

    def create_lr_lambda(steps_per_epoch, warmup_epochs, plateau_epochs, decay_epochs, peak_lr, final_lr):
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

    steps_per_epoch = max(1, len(trainloader) // config.get("accumulation_steps", 1))
    lr_lambda = create_lr_lambda(
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=config.get("warmup_epochs", 3),
        plateau_epochs=config.get("plateau_epochs", 3),
        decay_epochs=config.get("decay_epochs", 3),
        peak_lr=peak_lr,
        final_lr=peak_lr / 500
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    logging.info("Initializing ClassificationTrainer for finetuning")
    finetuner = ClassificationTrainer(
        model=model,
        train_loader=trainloader,
        val_loader=valloader_closely,
        val_loader_distantly=valloader_distantly,
        num_epochs=config["max_epochs"]["fine_tuning"],
        patience=config["patience"]["fine_tuning"],
        metrics_jsonl_path="finetuning_metrics.jsonl",
        optimizer=optimizer,
        scheduler=scheduler,
        use_amp=True,
        champs_dir=CHAMPS_DIR,
        earliest_champion_epoch=1,
        earliest_stop_epoch=2 * config.get("plateau_epochs", 1),
        config=config,
        finetune_dir = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/finetuned_models"
    )

    logging.info("Starting fine-tuning")
    finetuner.train()
    wandb.finish()

if __name__ == "__main__":
    main()
