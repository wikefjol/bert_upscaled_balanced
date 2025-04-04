import os
import json
import datetime
import hashlib
import torch
from filelock import FileLock

def build_champion_key(mode, config):
    """
    Build a unique key for pretrain/finetune using:
    (k, overlapping, num_layers, num_attention_heads, hidden_size, intermediate_size).
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
    Only the 'model' section is hashed for uniqueness.
    """
    model_str = json.dumps(config["model"], sort_keys=True)
    config_hash = hashlib.md5(model_str.encode("utf-8")).hexdigest()[:6]
    return f"{champion_key}_{config_hash}"

def save_champion(model, config, epoch, champion_key, champs_dir, label_encoder):
    """
    Save a champion checkpoint for pretraining.
    
    Parameters:
      - model: the model instance.
      - config: the training configuration.
      - epoch: current epoch number.
      - champion_key: unique key identifying this run.
      - champs_dir: directory to store champion checkpoints.
      - label_encoder: optional label encoder; for pretraining this is typically None.
    
    Saves:
      - Model checkpoint (encoder_best.pt)
      - If provided, the label encoder mapping (label_encoder.json)
      - The config (config_best.json)
    
    Returns:
      Tuple(checkpoint_path, config_path)
    """
    run_id = get_run_id(champion_key, config)
    run_folder = os.path.join(champs_dir, run_id)
    os.makedirs(run_folder, exist_ok=True)

    checkpoint_path = os.path.join(run_folder, "encoder_best.pt")
    torch.save(model.state_dict(), checkpoint_path)

    # Save label mapping if available
    if label_encoder is not None:
        mapping_path = os.path.join(run_folder, "label_encoder.json")
        mapping_data = {
            "label_to_index": label_encoder.label_to_index,
            "index_to_label": label_encoder.index_to_label,
        }
        with open(mapping_path, "w") as f:
            json.dump(mapping_data, f, indent=2)
        # Optionally record mapping path in config for later use
        config["label_encoder_path"] = mapping_path

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
    For pretraining (mode 'pretrain'), lower loss is better.
    
    Keeps up to 4 entries (champion + 3 runner-ups).
    """
    lock_path = metadata_path + ".lock"
    with FileLock(lock_path):
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

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

        champion_data.insert(0, new_entry)
        if mode == "pretrain":
            champion_data.sort(key=lambda x: x["val_metric"])
        else:
            champion_data.sort(key=lambda x: x["val_metric"], reverse=True)

        champion_data[:] = champion_data[:4]

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return champion_data[0]
