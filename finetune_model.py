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

# We can import build_champion_key if needed for loading a champion:
from src.utils.champions import build_champion_key

load_dotenv()

CHAMPS_DIR = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/champs"
CHAMPIONS_FILE = os.path.join(CHAMPS_DIR, "champions.json")

def expand_config(cfg):
    """
    Expands finetuning config with environment paths and derived fields.
    """
    try:
        # Alphabet for tokenization & augmentation
        cfg.setdefault("preprocessing", {}).setdefault("tokenization", {})["alphabet"] = cfg["alphabet"]
        cfg.setdefault("preprocessing", {}).setdefault("augmentation", {}).setdefault("training", {})["alphabet"] = cfg["alphabet"]
        cfg["preprocessing"]["augmentation"].setdefault("evaluation", {})["alphabet"] = cfg["alphabet"]

        # Derive max_position_embeddings and optimal lengths
        k = cfg["preprocessing"]["tokenization"]["k"]
        # If overlapping is enabled, adjust max_sequence_length.
        if cfg["preprocessing"]["tokenization"].get("overlapping"):
            # Multiply by k to get the effective sequence length.
            cfg["max_sequence_length"] = cfg["max_sequence_length"] * cfg["preprocessing"]["tokenization"]["k"]
            
        cfg["model"]["max_position_embeddings"] = cfg["max_sequence_length"] // k + 2
        optimal = cfg["max_sequence_length"] // k
        cfg["preprocessing"].setdefault("padding", {})["optimal_length"] = optimal
        cfg["preprocessing"].setdefault("truncation", {})["optimal_length"] = optimal

        # Data path from environment
        data_dir = os.getenv("DATA_DIR")
        if not data_dir:
            raise ValueError("DATA_DIR is not set in the environment (.env)")
        cfg["DATA_PATH"] = os.path.join(data_dir, cfg["DATA_FILE"])
    except KeyError as e:
        logging.error(f"Missing key in config expansion: {e}")
        raise e
    return cfg

def load_pretrained_encoder(config):
    """
    Loads the best pretraining checkpoint for the config's model parameters.
    Uses the new champion logic, i.e. build_champion_key('pretrain', config).
    """
    
    champion_key = build_champion_key("pretrain", config)
    if not os.path.exists(CHAMPIONS_FILE):
        raise ValueError(f"No champions.json found at {CHAMPIONS_FILE}")

    with open(CHAMPIONS_FILE, "r") as f:
        champs_data = json.load(f)

    if champion_key not in champs_data or "entries" not in champs_data[champion_key]:
        raise ValueError(f"No champion data found for '{champion_key}' in {CHAMPIONS_FILE}")

    # Best entry is always index 0
    best_entry = champs_data[champion_key]["entries"][0]
    checkpoint_path = best_entry["checkpoint_path"]
    logging.info(f"Resolved champion checkpoint at {checkpoint_path} for key={champion_key}")
    return checkpoint_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to finetuning config JSON")
    args = parser.parse_args()

    # TODO: Adapt configs to take in these paths. Or dont. 
    train_path = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/data/final_train_set.csv"
    val_path_closely = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/data/closely_related_val_set.csv" 
    val_path_distantly = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/data/distantly_related_val_set.csv" 

    # Load and expand config
    with open(args.config, "r") as f:
        config = json.load(f)
    config = expand_config(config)

    logging.basicConfig(level=config.get("system_log_lvl", logging.INFO))
    wandb.init(project=config.get("PROJECT_NAME", "default_project"), config=config)

    # --- Build Vocabulary ---
    logging.info("Building vocabulary")
    k_val = config["preprocessing"]["tokenization"]["k"]
    constructor = KmerVocabConstructor(k=k_val, alphabet=config["alphabet"])
    vocab = Vocabulary()
    vocab.build_from_constructor(constructor, data=[])

    # --- Build Preprocessor ---
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

    # --- Build combined DataFrame for label encoding and weight computation ---
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

    # --- Initialize label encoder using the combined DataFrame ---
    label_encoder = LabelEncoder(labels=combined_df[target_col])

    # --- Compute per-sample weights for training ---
    
    # Use combined_df to ensure all classes are represented
    class_counts = combined_df[target_col].value_counts().to_dict()
    
    # Apply sqrt scaling to tone down extreme differences
    # class_weights = {cls: 1.0 / np.sqrt(count) for cls, count in class_counts.items()}

    # Apply inverse scaling to tone down extreme differences
    class_weights = {cls: 1.0 /(count + 1) for cls, count in class_counts.items()}
    
    # Map each training sample's label to its weight
    sample_weights = train_df[target_col].map(class_weights).tolist()

    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_df), replacement=True)

    # --- Build datasets ---
    train_set = ClassificationDataset(
        train_df, preprocessor=preprocessor, label_encoder=label_encoder, target_column=target_col
    )
    val_set_closely = ClassificationDataset(
        val_df_closely, preprocessor=preprocessor, label_encoder=label_encoder, target_column=target_col
    )
    val_set_distantly = ClassificationDataset(
        val_df_distantly, preprocessor=preprocessor, label_encoder=label_encoder, target_column=target_col
    )


    # --- Setup DataLoaders ---
    batch_size = config["batch_size"]
    num_workers = min(4, (os.cpu_count() or 1) // 2)
    trainloader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    valloader_closely = DataLoader(val_set_closely, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    valloader_distantly = DataLoader(val_set_distantly, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # --- Build BERT encoder and load champion weights ---
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
        return {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in sd.items()
        }

    # Filter out any non-encoder layers
    state_dict = strip_prefix(state_dict, "bert.")
    encoder_state = encoder.state_dict()
    filtered_sd = {k: v for k, v in state_dict.items() if k in encoder_state}
    encoder.load_state_dict(filtered_sd, strict=False)

    # --- Classification head ---
    num_classes = config["model"]["num_classes"]
    classification_head = SingleClassHead(
        in_features=config["model"]["hidden_size"],
        hidden_layer_size=(config["model"]["hidden_size"] + num_classes) // 2,
        out_features=num_classes,
        dropout_rate=config["model"]["dropout_rate"]
    )

    # Wrap in ModularBertax
    model = ModularBertax(encoder=encoder, mlm_head=None, classification_head=classification_head)
    model.classifyMode()# = "finetune"  # Mark mode for champion logic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Setup optimizer & LR schedule ---
    peak_lr = config["peak_lr"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr)

    def create_lr_lambda(
        steps_per_epoch,
        warmup_epochs,
        plateau_epochs,
        decay_epochs,
        peak_lr,
        final_lr
    ):
        def lr_lambda(current_step):
            current_epoch = current_step / steps_per_epoch
            if current_epoch <= warmup_epochs:
                return current_epoch / warmup_epochs
            elif current_epoch <= warmup_epochs + plateau_epochs:
                return 1.0
            elif current_epoch <= warmup_epochs + plateau_epochs + decay_epochs:
                decay_progress = (current_epoch - warmup_epochs - plateau_epochs) / decay_epochs
                return 1.0 - decay_progress * (
                    1.0 - (final_lr / peak_lr)
                )
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

    # --- Fine-tune ---
    logging.info("Initializing ClassificationTrainer for finetuning")
    finetuner = ClassificationTrainer(
        model=model,
        train_loader=trainloader,
        # CHANGES: provide two val loaders
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
        config=config
    )

    logging.info("Starting fine-tuning")
    finetuner.train()
    wandb.finish()

if __name__ == "__main__":
    main()
