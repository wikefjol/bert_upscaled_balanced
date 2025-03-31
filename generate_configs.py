import json, copy, os

# Output directory for all configs
configs_dir = "configs"
os.makedirs(configs_dir, exist_ok=True)

# Load the combined model configurations from JSON
config_file = "model_configurations.json"
with open(config_file, "r") as f:
    models_data = json.load(f)

# Base config for pretraining (from previous working version)
base_config_pt = {
    "DATA_FILE": "unite_reps.fasta",
    "PROJECT_NAME": "gold_standard_pt",
    "small_set": False,
    "n_test": 1000,
    "alphabet": ["A", "C", "G", "T"],
    "max_sequence_length": 600,
    "target_labels": ["genus"],
    "model": {},  # to be replaced with hyperparams from JSON
    "preprocessing": {
        "tokenization": {"strategy": "kmer", "k": 3, "overlapping": True},
        "padding": {"strategy": "end"},
        "truncation": {"strategy": "slidingwindow"},
        "augmentation": {
            "training": {"strategy": "base", "modification_probability": 0.05},
            "evaluation": {"strategy": "identity"}
        }
    },
    "max_epochs": {"pre_training": 600, "fine_tuning": 0},
    "patience": {"pre_training": 10, "fine_tuning": 10},
    "peak_lr": 5e-3,      # will be overridden
    "warmup_epochs": 3,
    "plateau_epochs": 50,
    "decay_epochs": 500,
    "batch_size": 64,     # will be overridden
    "accumulation_steps": 1,
    "masking_percentage": 0.15,
    "test_size": 0.1,
    "system_log_lvl": 20
}

# Base config for finetuning
base_config_ft = {
    "DATA_FILE": "unite_reps.fasta",
    "PROJECT_NAME": "gold_standard_ft",
    "small_set": False,
    "n_test": 1000,
    "alphabet": ["A", "C", "G", "T"],
    "max_sequence_length": 600,
    "target_labels": ["genus"],
    "model": {},  # to be replaced with hyperparams from JSON
    "preprocessing": {
        "tokenization": {"strategy": "kmer", "k": 6, "overlapping": False},
        "padding": {"strategy": "end"},
        "truncation": {"strategy": "slidingwindow"},
        "augmentation": {
            "training": {"strategy": "base", "modification_probability": 0.05},
            "evaluation": {"strategy": "identity"}
        }
    },
    "max_epochs": {"fine_tuning": 20},
    "patience": {"fine_tuning": 10},
    "peak_lr": 0.0005,     # will be overridden
    "warmup_epochs": 1,
    "plateau_epochs": 3,
    "decay_epochs": 3,
    "accumulation_steps": 1,
    "batch_size": 1248,    # will be overridden
    "test_size": 0.1,
    "system_log_lvl": 20
}

# Iterate over each model in the JSON
for model_id, model_config in models_data.items():
    hyperparams = model_config["hyperparams"]
    pretrain_map = model_config["pretrain"]
    finetune_map = model_config["finetune"]

    # Update the base configs with the model's hyperparameters
    base_config_pt["model"] = copy.deepcopy(hyperparams)
    base_config_ft["model"] = copy.deepcopy(hyperparams)

    # Create a subfolder for this model architecture
    model_folder = os.path.join(configs_dir, model_id)
    os.makedirs(model_folder, exist_ok=True)

    # Loop over both overlapping settings: True ("ov") and False ("no")
    for overlapping, ov_key in [(True, "ov"), (False, "no")]:
        # Set tokenization overlapping flag for both configs
        base_config_pt["preprocessing"]["tokenization"]["overlapping"] = overlapping
        base_config_ft["preprocessing"]["tokenization"]["overlapping"] = overlapping

        # Pretraining configs
        for k in range(1, 8):
            config = copy.deepcopy(base_config_pt)
            config["preprocessing"]["tokenization"]["k"] = k
            k_str = str(k)
            mapping = pretrain_map[ov_key][k_str]
            config["batch_size"] = mapping["batch_size"]
            config["peak_lr"] = mapping["peak_lr"]

            # Filename: k{k}_{ov|no}_pt_{layers}layers_{hidden_size}dim_{intermediate}int_{heads}heads.json
            file_name = (f"k{k}_{ov_key}_pt_{hyperparams['num_layers']}layers_"
                         f"{hyperparams['hidden_size']}dim_{hyperparams['intermediate_size']}int_"
                         f"{hyperparams['num_attention_heads']}heads.json")
            file_path = os.path.join(model_folder, file_name)
            with open(file_path, "w") as f:
                json.dump(config, f, indent=4)

        # Finetuning configs (include _g_ in filename)
        for k in range(1, 8):
            config = copy.deepcopy(base_config_ft)
            config["preprocessing"]["tokenization"]["k"] = k
            k_str = str(k)
            mapping = finetune_map[ov_key][k_str]
            config["batch_size"] = mapping["batch_size"]
            config["peak_lr"] = mapping["peak_lr"]

            # Filename: k{k}_{ov|no}_ft_g_{layers}layers_{hidden_size}dim_{intermediate}int_{heads}heads.json
            file_name = (f"k{k}_{ov_key}_ft_g_{hyperparams['num_layers']}layers_"
                         f"{hyperparams['hidden_size']}dim_{hyperparams['intermediate_size']}int_"
                         f"{hyperparams['num_attention_heads']}heads.json")
            file_path = os.path.join(model_folder, file_name)
            with open(file_path, "w") as f:
                json.dump(config, f, indent=4)
