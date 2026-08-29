import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

class StockTimeSeriesDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int = 20):
        self.seq_len = seq_len
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len]

def prepare_leakage_free_splits(
    df: pd.DataFrame, 
    feature_cols: list, 
    target_col: str = "target",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seq_len: int = 20,
    batch_size: int = 32
):
    """
    Executes chronological 70/15/15 train-val-test split and fits RobustScaler
    EXCLUSIVELY on the training partition to prevent lookahead data leakage.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # 1. Fit scaler strictly on training features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(train_df[feature_cols].values)
    X_val_scaled = scaler.transform(val_df[feature_cols].values)
    X_test_scaled = scaler.transform(test_df[feature_cols].values)

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    # 2. Build PyTorch sliding window datasets
    train_dataset = StockTimeSeriesDataset(X_train_scaled, y_train, seq_len=seq_len)
    val_dataset = StockTimeSeriesDataset(X_val_scaled, y_val, seq_len=seq_len)
    test_dataset = StockTimeSeriesDataset(X_test_scaled, y_test, seq_len=seq_len)

    # 3. Create DataLoaders (chronological order, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    split_info = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "scaler": scaler,
        "y_test_aligned": y_test[seq_len:]
    }

    return train_loader, val_loader, test_loader, split_info