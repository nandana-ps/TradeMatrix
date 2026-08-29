import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.download import fetch_and_transform_stock_data
from src.data.features import compute_quantitative_features
from src.data.dataset import prepare_leakage_free_splits
from src.models.trainer import ModelTrainer
from src.backtest.optimizer import optimize_signal_threshold
from src.backtest.metrics import compute_institutional_metrics
from src.backtest.benchmark import calculate_buy_and_hold

# -------------------------------------------------------------
# 1. Model Definitions for Ablation Variants
# -------------------------------------------------------------
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.mha(x, x, x)
        return self.norm(x + self.dropout(attn_out))

class AblationLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, use_attention: bool = False, bidirectional: bool = False, dropout: float = 0.2):
        super().__init__()
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )

        effective_dim = hidden_dim * num_directions
        if use_attention:
            self.attention = MultiHeadAttentionBlock(embed_dim=effective_dim, num_heads=4, dropout=dropout)
        else:
            self.attention = None

        self.head = nn.Sequential(
            nn.Linear(effective_dim, 16),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        lstm_out, (hn, _) = self.lstm(x)
        if self.use_attention:
            attn_out = self.attention(lstm_out)
            rep = attn_out[:, -1, :]
        else:
            if self.bidirectional:
                rep = torch.cat((hn[-2], hn[-1]), dim=1)
            else:
                rep = hn[-1]
        return self.head(rep).squeeze(-1)

# -------------------------------------------------------------
# 2. Experiment Execution Runner
# -------------------------------------------------------------
def run_single_experiment(exp_name, feat_df, feature_cols, use_attention, bidirectional, epochs=15):
    print(f"\n>>> Running {exp_name} | Features: {len(feature_cols)} | Attention: {use_attention} | Bi-LSTM: {bidirectional}")
    
    train_loader, val_loader, test_loader, split_info = prepare_leakage_free_splits(
        feat_df,
        feature_cols=feature_cols,
        target_col="target",
        train_ratio=0.70,
        val_ratio=0.15,
        seq_len=20,
        batch_size=32
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AblationLSTM(
        input_dim=len(feature_cols),
        hidden_dim=32,
        use_attention=use_attention,
        bidirectional=bidirectional
    ).to(device)

    chk_path = f"checkpoints/ablation_{exp_name.replace(' ', '_').lower()}.pt"
    trainer = ModelTrainer(model=model, train_loader=train_loader, val_loader=val_loader, lr=1e-3, patience=4, checkpoint_path=chk_path, device=device)
    trainer.fit(max_epochs=epochs)

    # Validation Threshold Optimization
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            val_preds.extend(model(bx.to(device)).cpu().numpy())
            val_targets.extend(by.numpy())

    opt_results = optimize_signal_threshold(np.array(val_targets), np.array(val_preds))
    tau = opt_results["optimal_tau"]

    # Test Evaluation
    test_preds = []
    with torch.no_grad():
        for bx, _ in test_loader:
            test_preds.extend(model(bx.to(device)).cpu().numpy())

    test_preds = np.array(test_preds)
    test_returns = split_info["y_test_aligned"]

    # Generate signals
    signals = np.zeros_like(test_preds)
    signals[test_preds > tau] = 1.0
    signals[test_preds < -tau] = -1.0

    # Deduct friction (5 bps fee + 2 bps slippage)
    pos_changes = np.abs(np.diff(signals, prepend=0))
    friction = pos_changes * 0.0007
    net_returns = (signals * test_returns) - friction

    metrics = compute_institutional_metrics(net_returns)
    hit_rate = np.mean((signals > 0) == (test_returns > 0)) * 100

    metrics.update({
        "Model": exp_name,
        "Optimal Tau": tau,
        "Hit Rate (%)": hit_rate,
        "Total Trades": int(np.sum(pos_changes > 0))
    })
    return metrics

# -------------------------------------------------------------
# 3. Main Pipeline
# -------------------------------------------------------------
if __name__ == "__main__":
    raw_df = fetch_and_transform_stock_data("AAPL")
    feat_df = compute_quantitative_features(raw_df)

    all_features = ["log_ret", "rsi_14", "macd", "volatility_20", "volume_zscore"]
    univ_feature = ["log_ret"]

    experiments = [
        ("Exp A (Vanilla Univariate LSTM)", univ_feature, False, False),
        ("Exp B (Feature-Enhanced LSTM)", all_features, False, False),
        ("Exp C (Uni-LSTM + Self-Attention)", all_features, True, False),
        ("Exp D (Bi-LSTM + Self-Attention)", all_features, True, True),
    ]

    results = []
    for name, feats, attn, bi in experiments:
        res = run_single_experiment(name, feat_df, feats, use_attention=attn, bidirectional=bi, epochs=15)
        results.append(res)

    # Add Buy & Hold Benchmark
    test_split_idx = int(len(feat_df) * 0.85)
    test_prices = feat_df["Close"].iloc[test_split_idx:]
    b_and_h = calculate_buy_and_hold(test_prices)
    results.append({
        "Model": "Benchmark: Buy & Hold",
        "Total Return (%)": b_and_h["Total Return (%)"],
        "Annualized Return (%)": b_and_h["Annualized Return (%)"],
        "Annualized Volatility (%)": b_and_h["Annualized Volatility (%)"],
        "Sharpe Ratio": b_and_h["Sharpe Ratio"],
        "Sortino Ratio": 0.0,
        "Calmar Ratio": 0.0,
        "Max Drawdown (%)": b_and_h["Max Drawdown (%)"],
        "Hit Rate (%)": 0.0,
        "Win Rate (%)": 0.0,
        "Profit Factor": 0.0,
        "Optimal Tau": 0.0,
        "Total Trades": 1
    })

    summary_df = pd.DataFrame(results)
    summary_cols = ["Model", "Total Return (%)", "Annualized Return (%)", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown (%)", "Win Rate (%)", "Total Trades"]

    print("\n========================= MASTER RESEARCH COMPARISON MATRIX =========================")
    print(summary_df[summary_cols].to_string(index=False))