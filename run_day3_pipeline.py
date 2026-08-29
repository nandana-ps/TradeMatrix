import torch
import numpy as np

from src.data.download import fetch_and_transform_stock_data
from src.data.features import compute_quantitative_features
from src.data.dataset import prepare_leakage_free_splits
from src.models.attention_lstm import AttentionLSTM
from src.models.trainer import ModelTrainer
from src.backtest.optimizer import optimize_signal_threshold
from src.backtest.engine import run_friction_backtest

print("=== 1. INGESTING DATA & COMPUTING FEATURES ===")
raw_df = fetch_and_transform_stock_data("AAPL")
feat_df = compute_quantitative_features(raw_df)
feature_cols = ["log_ret", "rsi_14", "macd", "volatility_20", "volume_zscore"]

print("\n=== 2. STRICT 70/15/15 CHRONOLOGICAL SPLIT & ROBUST SCALING ===")
train_loader, val_loader, test_loader, split_info = prepare_leakage_free_splits(
    feat_df, 
    feature_cols=feature_cols,
    target_col="target",
    train_ratio=0.70,
    val_ratio=0.15,
    seq_len=20,
    batch_size=32
)

print(f"Train rows: {split_info['train_rows']} | Val rows: {split_info['val_rows']} | Test rows: {split_info['test_rows']}")

print("\n=== 3. TRAINING WITH CHECKPOINTING & EARLY STOPPING ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AttentionLSTM(input_dim=len(feature_cols), hidden_dim=32, num_heads=4)

trainer = ModelTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    lr=1e-3,
    patience=5,
    checkpoint_path="checkpoints/best_attention_model.pt",
    device=device
)
history = trainer.fit(max_epochs=15)

print("\n=== 4. VALIDATION THRESHOLD OPTIMIZATION ===")
model.eval()
val_preds = []
val_targets = []
with torch.no_grad():
    for bx, by in val_loader:
        val_preds.extend(model(bx.to(device)).cpu().numpy())
        val_targets.extend(by.numpy())

opt_results = optimize_signal_threshold(np.array(val_targets), np.array(val_preds))
print(f"Optimal Conviction Threshold (\u03c4): {opt_results['optimal_tau']:.5f}")
print(f"Validation Post-Fee Sharpe: {opt_results['val_sharpe']:.2f}")
print(f"Validation Filtered Trades: {opt_results['val_trades']}")

print("\n=== 5. OUT-OF-SAMPLE TEST EVALUATION (ZERO LEAKAGE) ===")
test_preds = []
with torch.no_grad():
    for bx, _ in test_loader:
        test_preds.extend(model(bx.to(device)).cpu().numpy())

test_preds = np.array(test_preds)
test_returns = split_info["y_test_aligned"]

# Evaluate with the threshold calibrated strictly on validation
filtered_signals = np.zeros_like(test_preds)
optimal_tau = opt_results["optimal_tau"]
filtered_signals[test_preds > optimal_tau] = 1.0
filtered_signals[test_preds < -optimal_tau] = -1.0

test_metrics = run_friction_backtest(test_returns, filtered_signals)
for k, v in test_metrics.items():
    print(f"{k:<25}: {v:.2f}")