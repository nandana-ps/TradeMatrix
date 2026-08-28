import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1. Global Seed Anchors for Exact Scientific Reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)

# 2. Sliding Temporal Window Dataset (Lookback T=20)
class TimeSeriesDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int = 20):
        self.seq_len = seq_len
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        # Window: [t-T+1, ..., t] -> Target: y_t = r_{t+1}
        return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len]

# 3. Baseline Unidirectional LSTM Network
class BaselineLSTM(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len=20, Feature_Dim]
        lstm_out, (hn, _) = self.lstm(x)
        last_hidden = hn[-1]  # Extract representation from the final step
        return self.regressor(last_hidden).squeeze(-1)

# 4. Standard Training & Validation Loop with Huber Loss
def train_baseline_model(X_train, y_train, X_val, y_val, seq_len=20, epochs=5, batch_size=32, lr=1e-3):
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train, seq_len), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val, seq_len), batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineLSTM(input_dim=X_train.shape[1]).to(device)
    
    # Huber Loss prevents extreme market return outliers from corrupting gradients
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()
            preds = model(b_x)
            loss = criterion(preds, b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_x, b_y in val_loader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                preds = model(b_x)
                val_loss += criterion(preds, b_y).item()
                
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(val_loader):.6f}")

    return model
