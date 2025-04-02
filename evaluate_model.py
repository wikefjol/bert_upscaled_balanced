#!/usr/bin/env python
import os
import json
import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel
from sklearn.metrics import f1_score, balanced_accuracy_score

from src.data_module.dataset import ClassificationDataset
from src.factories.preprocessing_factory import create_preprocessor
from src.utils.vocab import Vocabulary, KmerVocabConstructor
from src.model.backbone import ModularBertax
from src.model.heads import SingleClassHead
from src.model.encoders import LabelEncoder

logging.basicConfig(level=logging.INFO)

# Directories and default paths
CHAMPS_DIR = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/champs"
CHAMPIONS_FILE = os.path.join(CHAMPS_DIR, "champions.json")
DEFAULT_EVAL_CSV = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/data/naive_val.csv"

def evaluate_model(model, data_loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["encoded_label"].to(device).long()
            logits = model(input_ids, attention_mask)
            if criterion is not None:
                total_loss += criterion(logits, labels).item()
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    n_batches = len(data_loader)
    avg_loss = total_loss / n_batches if (criterion and n_batches > 0) else None
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return {
        "loss": avg_loss,
        "accuracy": np.mean(all_preds == all_labels),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds)
    }

def choose_champion():
    # Load champions.json and filter classification champions.
    if not os.path.exists(CHAMPIONS_FILE):
        raise ValueError(f"No champions.json found at {CHAMPIONS_FILE}")
    with open(CHAMPIONS_FILE, "r") as f:
        champs_data = json.load(f)
    classify_keys = [k for k in champs_data.keys() if k.startswith("classify_")]
    if not classify_keys:
        raise ValueError("No classification champions found.")
    print("Available classification champions:")
    for i, key in enumerate(classify_keys, start=1):
        print(f"{i}: {key}")
    selection = input("Enter the number of the champion to evaluate: ")
    try:
        idx = int(selection) - 1
        if idx < 0 or idx >= len(classify_keys):
            raise ValueError()
    except ValueError:
        raise ValueError("Invalid selection.")
    chosen_key = classify_keys[idx]
    # Use the first entry (assumed best)
    entry = champs_data[chosen_key]["entries"][0]
    return entry  # contains "config_path", "checkpoint_path", etc.

def main():
    # Choose champion by enumerating in terminal.
    entry = choose_champion()
    config_path = entry["config_path"]
    checkpoint_path = entry["checkpoint_path"]
    logging.info(f"Selected champion config: {config_path}")
    logging.info(f"Selected champion weights: {checkpoint_path}")
    
    # Load expanded config from config_path
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Build vocabulary & preprocessor from the champion config.
    k_val = config["preprocessing"]["tokenization"]["k"]
    constructor = KmerVocabConstructor(k=k_val, alphabet=config["alphabet"])
    vocab = Vocabulary()
    vocab.build_from_constructor(constructor, data=[])
    preprocessor = create_preprocessor(config["preprocessing"], vocab, training=False)
    
    # Read evaluation CSV
    eval_csv = input(f"Enter evaluation CSV path (or press Enter to use default: {DEFAULT_EVAL_CSV}): ").strip()
    if not eval_csv:
        eval_csv = DEFAULT_EVAL_CSV
    logging.info(f"Using evaluation CSV: {eval_csv}")
    df_eval = pd.read_csv(eval_csv)
    
    # Use target column from config.
    target_cols = config.get("target_labels", ["label"])
    target_col = target_cols[0] if isinstance(target_cols, list) else target_cols
    logging.info(f"Using target column '{target_col}' with champion num_classes {config['model'].get('num_classes')}")
    label_encoder = LabelEncoder(labels=df_eval[target_col])
    
    eval_dataset = ClassificationDataset(
        df_eval,
        preprocessor=preprocessor,
        label_encoder=label_encoder,
        target_column=target_col
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_loader = DataLoader(
        eval_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    # Rebuild model from the champion config.
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
    num_classes = config["model"].get("num_classes", 2)
    classification_head = SingleClassHead(
        in_features=config["model"]["hidden_size"],
        hidden_layer_size=(config["model"]["hidden_size"] + num_classes) // 2,
        out_features=num_classes,
        dropout_rate=config["model"]["dropout_rate"]
    )
    model = ModularBertax(encoder=encoder, mlm_head=None, classification_head=classification_head)
    model.classifyMode()
    model.to(device)
    
    # Load champion weights.
    logging.info(f"Loading champion weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    # Extract champion's num_classes.
    champ_num_classes = model.classification_head.sequential[-1].out_features
    logging.info(f"Champion model expects {champ_num_classes} classes.")
    if df_eval[target_col].nunique() != champ_num_classes:
        logging.warning(f"Evaluation CSV has {df_eval[target_col].nunique()} unique classes but champion expects {champ_num_classes}.")
    
    # Evaluate.
    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate_model(model, eval_loader, device, criterion)
    logging.info(f"Evaluation metrics: {metrics}")
    print("---- Evaluation Results ----")
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and v is not None:
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
