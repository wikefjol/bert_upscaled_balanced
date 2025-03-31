#!/usr/bin/env python
import os
import json

# Hardcoded champions file path
CHAMPS_PATH = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/champs/champions.json"

def parse_champion_key(key):
    """
    Parse a champion key of the form:
      "{mode}_k{k}_{overlap}_{num_layers}layers_{num_attention_heads}heads_{hidden_size}hidden_{intermediate_size}intermediate"
    Returns a dict with keys: mode, k (int), overlap, num_layers (int), num_heads (int),
    hidden (int), intermediate (int).
    """
    try:
        parts = key.split('_')
        if len(parts) != 7:
            raise ValueError(f"Unexpected format: {key}")
        mode = parts[0]
        k = int(parts[1].lstrip('k'))
        overlap = parts[2]
        num_layers = int(parts[3].replace("layers", ""))
        num_heads = int(parts[4].replace("heads", ""))
        hidden = int(parts[5].replace("hidden", ""))
        intermediate = int(parts[6].replace("intermediate", ""))
        return {
            "mode": mode,
            "k": k,
            "overlap": overlap,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "hidden": hidden,
            "intermediate": intermediate
        }
    except Exception as e:
        print(f"Error parsing key '{key}': {e}")
        return None

def group_champions(champions):
    """
    Group champions hierarchically in this order:
      Architecture -> Mode -> k -> Overlap.
    Mode is displayed as "Pretraining" if key starts with "pretrain", else "Finetuning".
    Overlap is converted to "Overlapping" or "Nonoverlapping".
    """
    groups = {}
    for key, data in champions.items():
        if key.lower() == "history":
            continue
        parsed = parse_champion_key(key)
        if not parsed:
            continue
        # Architecture grouping: tuple of (num_layers, num_heads, hidden, intermediate)
        arch = (parsed["num_layers"], parsed["num_heads"], parsed["hidden"], parsed["intermediate"])
        # Mode: map "pretrain" -> "Pretraining", else "Finetuning"
        mode_disp = "Pretraining" if parsed["mode"] == "pretrain" else "Finetuning"
        k_val = parsed["k"]
        overlap_disp = "Overlapping" if parsed["overlap"] == "overlap" else "Nonoverlapping"
        groups.setdefault(arch, {}).setdefault(mode_disp, {}).setdefault(k_val, {})[overlap_disp] = data
    return groups

def print_groups(groups):
    """
    Print the champion information hierarchically in the order:
      Architecture, Mode, k, Overlap.
    For each final group, prints the best champion's epoch and metric (with its name).
    """
    # Define fixed order for modes and overlap
    mode_order = ["Pretraining", "Finetuning"]
    overlap_order = ["Overlapping", "Nonoverlapping"]
    for arch, arch_group in sorted(groups.items()):
        num_layers, num_heads, hidden, intermediate = arch
        print(f"Architecture: {num_layers} layers, {num_heads} heads, {hidden} hidden, {intermediate} intermediate")
        for mode in mode_order:
            if mode not in arch_group:
                continue
            print(f"  Mode: {mode}")
            mode_group = arch_group[mode]
            for k_val in sorted(mode_group.keys()):
                print(f"    k = {k_val}")
                for overlap in overlap_order:
                    if overlap not in mode_group[k_val]:
                        continue
                    data = mode_group[k_val][overlap]
                    entries = data.get("entries", [])
                    if not entries:
                        continue
                    best = entries[0]
                    epoch = best.get("epoch", "")
                    metric_name = best.get("metric_name", "Metric")
                    metric = best.get("val_metric", "")
                    print(f"      Overlap: {overlap:<15} Epoch: {epoch:<4} {metric_name}: {metric}")
            print()
        print("="*80)

def main():
    if not os.path.exists(CHAMPS_PATH):
        print(f"Champions file not found: {CHAMPS_PATH}")
        exit(1)
    with open(CHAMPS_PATH, "r") as f:
        champions = json.load(f)
    groups = group_champions(champions)
    print_groups(groups)

if __name__ == "__main__":
    main()
