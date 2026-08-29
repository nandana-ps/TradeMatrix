import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.download import fetch_and_transform_stock_data
from src.data.features import compute_quantitative_features
from src.models.attention_lstm import AttentionLSTM
from src.models.baseline_lstm import TimeSeriesDataset
from src.backtest.engine import run_friction_backtest

print("=== 1. FETCHING DATA & COMPUTING FEATURES ===")
raw_df = fetch_and_transform_stock_data("AAPL")
feat_df = compute_quantitative_features(raw_df)
print(f"Dataset ready with shape: {feat_df.shape}")

feature_cols = ["log_ret", "rsi_14", "macd", "volatility_20", "volume_zscore"]
split = int(len(feat_df) * 0.8)

train_data = feat_df.iloc[:split]
test_data = feat_df.iloc[split:]

X_train, y_train = train_data[feature_cols].values, train_data["target"].values
X_test, y_test = test_data[feature_cols].values, test_data["target"].values

train_loader = DataLoader(TimeSeriesDataset(X_train, y_train, seq_len=20), batch_size=32, shuffle=False)
test_loader = DataLoader(TimeSeriesDataset(X_test, y_test, seq_len=20), batch_size=32, shuffle=False)

print("\n=== 2. TRAINING ATTENTION-LSTM MODEL (5 EPOCHS) ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionLSTM(input_dim=len(feature_cols), hidden_dim=32, num_heads=4).to(device)
criterion = torch.nn.HuberLoss(delta=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(1, 6):
    model.train()
    total_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch:02d}/05 | Huber Loss: {total_loss/len(train_loader):.6f}")

print("\n=== 3. OUT-OF-SAMPLE EVALUATION & FRICTION BACKTEST ===")
model.eval()
preds = []
with torch.no_grad():
    for bx, _ in test_loader:
        preds.extend(model(bx.to(device)).cpu().numpy())

preds = np.array(preds)
test_returns = y_test[20:]

metrics = run_friction_backtest(test_returns, preds)
for k, v in metrics.items():
    print(f"{k:<25}: {v:.2f}")
