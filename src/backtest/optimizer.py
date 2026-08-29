import numpy as np

def optimize_signal_threshold(
    val_returns: np.ndarray, 
    val_preds: np.ndarray, 
    threshold_grid: np.ndarray = np.linspace(0.0, 0.005, 51),
    transaction_cost: float = 0.0005,
    slippage: float = 0.0002
) -> dict:
    """
    Optimizes the decision threshold tau strictly on the Validation set to maximize 
    post-friction Sharpe ratio, preventing data leakage into the Test set.
    """
    best_tau = 0.0
    best_sharpe = -float("inf")
    best_metrics = {}

    for tau in threshold_grid:
        # Generate filtered signals based on conviction threshold
        signals = np.zeros_like(val_preds)
        signals[val_preds > tau] = 1.0
        signals[val_preds < -tau] = -1.0

        # Calculate position shifts and deductions
        position_changes = np.abs(np.diff(signals, prepend=0))
        total_friction = position_changes * (transaction_cost + slippage)

        net_returns = (signals * val_returns) - total_friction
        ann_return = (1 + np.prod(1 + net_returns) - 1.0) ** (252 / max(len(net_returns), 1)) - 1
        ann_vol = np.std(net_returns) * np.sqrt(252)
        sharpe = ann_return / (ann_vol + 1e-9)

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_tau = tau
            best_metrics = {
                "optimal_tau": float(best_tau),
                "val_sharpe": float(best_sharpe),
                "val_trades": int(np.sum(position_changes > 0)),
                "val_annualized_return": float(ann_return * 100)
            }

    return best_metrics