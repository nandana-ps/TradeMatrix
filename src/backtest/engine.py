import numpy as np
import pandas as pd

def run_friction_backtest(
    returns: np.ndarray, 
    predictions: np.ndarray, 
    transaction_cost: float = 0.0005, 
    slippage: float = 0.0002
) -> dict:
    """
    Executes a vectorized strategy backtest with 5 bps transaction fees + 2 bps slippage.
    """
    # 1. Generate directional trade signal: +1 (Long) or -1 (Short)
    signals = np.sign(predictions)
    
    # 2. Deduct fees on position changes: |signal_t - signal_{t-1}|
    position_changes = np.abs(np.diff(signals, prepend=0))
    total_friction = position_changes * (transaction_cost + slippage)
    
    # 3. Strategy Net Returns
    gross_returns = signals * returns
    net_returns = gross_returns - total_friction
    
    # 4. Cumulative compounding
    cum_returns = np.cumprod(1 + net_returns) - 1.0
    total_return = cum_returns[-1]
    
    # 5. Financial KPIs
    ann_return = (1 + total_return) ** (252 / len(net_returns)) - 1
    ann_vol = np.std(net_returns) * np.sqrt(252)
    sharpe = ann_return / (ann_vol + 1e-9)
    
    downside = net_returns[net_returns < 0]
    downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1e-9
    sortino = ann_return / (downside_vol + 1e-9)
    
    # 6. Maximum Drawdown
    equity = np.cumprod(1 + net_returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = np.min(drawdown)
    
    return {
        "Total Return (%)": total_return * 100,
        "Annualized Return (%)": ann_return * 100,
        "Annualized Volatility (%)": ann_vol * 100,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown (%)": max_drawdown * 100,
        "Total Trades": int(np.sum(position_changes > 0))
    }