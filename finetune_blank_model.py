#!/usr/bin/env python
import os, json, logging, argparse, torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel
from dotenv import load_dotenv
import wandb

from src.train.classification_trainer import ClassificationTrainer
from src.data_module.data_tools import fasta2pandas
from src.data_module.dataset import ClassificationDataset
from src.utils.vocab import Vocabulary, KmerVocabConstructor
from src.factories.preprocessing_factory import create_preprocessor
from src.model.backbone import ModularBertax
from src.model.heads import SingleClassHead
from src.model.encoders import LabelEncoder

load_dotenv()

def expand_config(cfg):
    """Sets up paths and parameters just like in train_single_model.py."""
    try:
        cfg.setdefault("preprocessing", {}).setdefault("tokenization", {})["alphabet"] = cfg["alphabet"]
        cfg.setdefault("preprocessing", {}).setdefault("augmentation", {}).setdefault("training", {})["alphabet"] = cfg["alphabet"]
        cfg["preprocessing"]["augmentation"].setdefault("evaluation", {})["alphabet"] = cfg["alphabet"]
        k = cfg["preprocessing"]["tokenization"]["k"]
        cfg["model"]["max_position_embeddings"] = cfg["max_sequence_length"] // k + 2
        optimal = cfg["max_sequence_length"] // k
        cfg["preprocessing"].setdefault("padding", {})["optimal_length"] = optimal
        cfg["preprocessing"].setdefault("truncation", {})["optimal_length"] = optimal
        data_dir = os.getenv("DATA_DIR")
        if not data_dir:
            raise ValueError("DATA_DIR is not set in the environment (.env)")
        cfg["DATA_PATH"] = os.path.join(data_dir, cfg["DATA_FILE"])
    except KeyError as e:
        logging.error(f"Missing key in config expansion: {e}")
        raise e
    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to finetuning config JSON")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)
    config = expand_config(config)
    logging.basicConfig(level=config.get("system_log_lvl", logging.INFO))

    # Initialize wandb
    wandb.init(project=config.get("PROJECT_NAME", "default_project"), config=config)

    # --- Build Vocabulary ---
    logging.info("Building vocabulary")
    k_val = config["preprocessing"]["tokenization"]["k"]
    constructor = KmerVocabConstructor(k=k_val, alphabet=config["alphabet"])
    vocab = Vocabulary()
    vocab.build_from_constructor(constructor, data=[])

    # --- Build Preprocessor ---
    logging.info("Building preprocessor")
    preprocessor = create_preprocessor(config["preprocessing"], vocab, training=True)

    # --- Load & Split Data ---
    all_data = fasta2pandas(config["DATA_PATH"])
    if config.get("small_set", False):
        all_data = all_data[: config.get("n_test", 10)]

    target = config.get("target_labels", [])
    target_col = target[0] if isinstance(target, list) and len(target) == 1 else target
    if len(target) == 1:
        num_classes = all_data[target[0]].nunique()
        config.setdefault("model", {})["num_classes"] = num_classes
        logging.info(f"Found {num_classes} unique classes for '{target[0]}'.")
    else:
        config.setdefault("model", {})["num_classes"] = 2
        logging.info("Multiple target labels not handled; defaulting num_classes = 2.")

    label_encoder = LabelEncoder(labels=all_data[target_col])
    train_df, val_df = train_test_split(
        all_data, test_size=config.get("test_size", 0.1), random_state=42, shuffle=True
    )

    train_set = ClassificationDataset(
        train_df, preprocessor=preprocessor,
        label_encoder=label_encoder, target_column=target_col
    )
    val_set = ClassificationDataset(
        val_df, preprocessor=preprocessor,
        label_encoder=label_encoder, target_column=target_col
    )

    # --- Build Model (fresh initialization) ---
    logging.info("Building newly initialized model for fine-tuning")
    bert_cfg = BertConfig(
        vocab_size=len(vocab),
        hidden_size=config["model"]["hidden_size"],
        num_hidden_layers=config["model"]["num_layers"],
        num_attention_heads=config["model"]["num_attention_heads"],
        intermediate_size=config["model"]["intermediate_size"],
        max_position_embeddings=config["model"]["max_position_embeddings"],
        hidden_dropout_prob=config["model"]["dropout_rate"],
        attention_probs_dropout_prob=config["model"]["dropout_rate"],
    )
    encoder = BertModel(bert_cfg)

    classification_head = SingleClassHead(
        in_features=config["model"]["hidden_size"],
        hidden_layer_size=(config["model"]["hidden_size"] + config["model"]["num_classes"]) // 2,
        out_features=config["model"]["num_classes"],
        dropout_rate=config["model"]["dropout_rate"]
    )
    model = ModularBertax(encoder=encoder, mlm_head=None, classification_head=classification_head)

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Setup DataLoaders ---
    batch_size = config["batch_size"]
    num_workers = min(4, (os.cpu_count() or 1) // 2)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # --- Setup Optimizer ---
    peak_lr = config["peak_lr"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr)
    model.classifyMode()

    # Learning rate scheduling
    def create_lr_lambda(steps_per_epoch, warmup_epochs, plateau_epochs, decay_epochs, peak_lr, final_lr):
        def lr_lambda(current_step):
            current_epoch = current_step / steps_per_epoch
            if current_epoch <= warmup_epochs:
                return current_epoch / warmup_epochs
            elif current_epoch <= warmup_epochs + plateau_epochs:
                return 1.0
            elif current_epoch <= warmup_epochs + plateau_epochs + decay_epochs:
                decay_progress = (current_epoch - warmup_epochs - plateau_epochs) / decay_epochs
                return 1.0 - decay_progress*(1.0 - (final_lr/peak_lr))
            else:
                return final_lr/peak_lr
        return lr_lambda

    lr_lambda = create_lr_lambda(
        steps_per_epoch=len(trainloader) // config["accumulation_steps"],
        warmup_epochs=config.get("warmup_epochs", 3),
        plateau_epochs=config.get("plateau_epochs", 3),
        decay_epochs=config.get("decay_epochs", 3),
        peak_lr=config["peak_lr"],
        final_lr=config["peak_lr"]/500
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # --- Fine-tune ---
    logging.info("Starting fine-tuning from scratch")
    finetuner = ClassificationTrainer(
        model=model,
        train_loader=trainloader,
        val_loader=valloader,
        num_epochs=config["max_epochs"]["fine_tuning"],
        patience=config["patience"]["fine_tuning"],
        metrics_jsonl_path="finetuning_metrics.jsonl",
        optimizer=optimizer,
        scheduler=scheduler,
        use_amp=True
    )
    finetuner.train()
    wandb.finish()

if __name__ == "__main__":
    main()
