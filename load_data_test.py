import pandas as pd
from src.data_module.dataset import ClassificationDataset
train_path = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/data/naive_train.csv"
val_path = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/data/naive_val.csv"

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

train_set = ClassificationDataset(
    train_df,
    preprocessor=preprocessor,
    label_encoder=label_encoder,
    target_column=target_col
)
val_set = ClassificationDataset(
    val_df,
    preprocessor=preprocessor,
    label_encoder=label_encoder,
    target_column=target_col
)