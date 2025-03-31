import os
import json
import datetime
import hashlib
import torch
from filelock import FileLock

def build_champion_key(mode, config):
    """
    Build a unique key for pretrain/finetune + 
    (k, overlapping, num_layers, heads, hidden_size, intermediate_size).
    """
    model_cfg = config["model"]
    token_cfg = config["preprocessing"]["tokenization"]
    k = token_cfg["k"]
    overlap_str = "overlap" if token_cfg.get("overlapping") else "nonoverlap"
    return (
        f"{mode}_"
        f"k{k}_{overlap_str}_"
        f"{model_cfg['num_layers']}layers_"
        f"{model_cfg['num_attention_heads']}heads_"
        f"{model_cfg['hidden_size']}hidden_"
        f"{model_cfg['intermediate_size']}intermediate"
    )

def get_run_id(champion_key, config):
    """
    Derive a stable folder ID from the champion key plus model config.
    We hash only the 'model' section for uniqueness.
    """
    model_str = json.dumps(config["model"], sort_keys=True)
    config_hash = hashlib.md5(model_str.encode("utf-8")).hexdigest()[:6]
    return f"{champion_key}_{config_hash}"

def save_champion(model, config, epoch, champion_key, champs_dir):
    """
    Saves a model checkpoint + config in a stable, hashed run folder
    (one folder per champion_key + model config).
    Overwrites 'encoder_best.pt' and 'config_best.json' each time.
    Returns (checkpoint_path, config_path).
    """
    run_id = get_run_id(champion_key, config)
    run_folder = os.path.join(champs_dir, run_id)
    os.makedirs(run_folder, exist_ok=True)

    checkpoint_path = os.path.join(run_folder, "encoder_best.pt")
    torch.save(model.state_dict(), checkpoint_path)

    config_path = os.path.join(run_folder, "config_best.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return checkpoint_path, config_path

def update_champion_metadata(
    metadata_path,
    champion_key,
    new_val_metric,
    new_checkpoint_path,
    new_config_path,
    epoch,
    champion_metric_name,
    mode
):
    """
    Update champion metadata for this champion_key. 
    We keep up to 4 total entries (champion + 3 runner-ups),
    sorting by ascending val_metric if mode == 'pretrain' 
    or descending if mode == 'finetune'.

    champion_key: output of build_champion_key()
    new_val_metric: float (loss or accuracy)
    champion_metric_name: 'val_loss' or 'val_accuracy'
    mode: 'pretrain' or 'finetune'
    """
    lock_path = metadata_path + ".lock"
    with FileLock(lock_path):
        # Load existing metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Ensure there's a place to store multiple entries
        if champion_key not in metadata:
            metadata[champion_key] = {"entries": []}

        champion_data = metadata[champion_key]["entries"]

        new_entry = {
            "epoch": epoch,
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "val_metric": new_val_metric,
            "checkpoint_path": new_checkpoint_path,
            "config_path": new_config_path,
            "metric_name": champion_metric_name,
            "mode": mode
        }

        # Insert the new result
        champion_data.insert(0, new_entry)
        # Sort ascending for pretrain (lower loss is better), descending for finetune (higher acc is better)
        if mode == "pretrain":
            champion_data.sort(key=lambda x: x["val_metric"])
        else:
            champion_data.sort(key=lambda x: x["val_metric"], reverse=True)

        # Keep top 4
        champion_data[:] = champion_data[:4]

        # Write updated metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # The best champion is champion_data[0]
        return champion_data[0]
