import numpy as np

def compute_institutional_metrics(net_returns: np.ndarray, trading_days: int = 252) -> dict:
    """
    Calculates institutional risk and performance metrics on out-of-sample returns.
    """
    if len(net_returns) == 0 or np.all(net_returns == 0):
        return {
            "Total Return (%)": 0.0,
            "Annualized Return (%)": 0.0,
            "Annualized Volatility (%)": 0.0,
            "Sharpe Ratio": 0.0,
            "Sortino Ratio": 0.0,
            "Calmar Ratio": 0.0,
            "Max Drawdown (%)": 0.0,
            "Win Rate (%)": 0.0,
            "Profit Factor": 0.0
        }

    # 1. Compounded Cumulative Return & Annualization
    cum_returns = np.cumprod(1 + net_returns) - 1.0
    total_return = cum_returns[-1]
    n_days = len(net_returns)
    ann_return = (1 + total_return) ** (trading_days / max(n_days, 1)) - 1.0

    # 2. Volatility & Downside Deviation
    ann_vol = np.std(net_returns) * np.sqrt(trading_days)
    downside_returns = net_returns[net_returns < 0]
    downside_vol = (
        np.std(downside_returns) * np.sqrt(trading_days)
        if len(downside_returns) > 0
        else 1e-9
    )

    # 3. Risk-Adjusted Ratios
    sharpe = ann_return / (ann_vol + 1e-9)
    sortino = ann_return / (downside_vol + 1e-9)

    # 4. Maximum Drawdown & Calmar Ratio
    equity_curve = np.cumprod(1 + net_returns)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_drawdown = float(np.min(drawdowns))
    calmar = ann_return / (abs(max_drawdown) + 1e-9) if max_drawdown < 0 else 0.0

    # 5. Trade Quality Metrics
    positive_days = net_returns[net_returns > 0]
    negative_days = net_returns[net_returns < 0]
    win_rate = (len(positive_days) / max(n_days, 1)) * 100
    profit_factor = (
        np.sum(positive_days) / (abs(np.sum(negative_days)) + 1e-9)
        if len(negative_days) > 0
        else 0.0
    )

    return {
        "Total Return (%)": float(total_return * 100),
        "Annualized Return (%)": float(ann_return * 100),
        "Annualized Volatility (%)": float(ann_vol * 100),
        "Sharpe Ratio": float(sharpe),
        "Sortino Ratio": float(sortino),
        "Calmar Ratio": float(calmar),
        "Max Drawdown (%)": float(max_drawdown * 100),
        "Win Rate (%)": float(win_rate),
        "Profit Factor": float(profit_factor)
    }