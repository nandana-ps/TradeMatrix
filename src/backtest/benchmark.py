import numpy as np
import pandas as pd

def calculate_buy_and_hold(prices: pd.Series) -> dict:
    """
    Computes passive Buy-and-Hold financial metrics from a price series.
    """
    # Daily log returns and compounded cumulative return
    log_returns = np.log(prices / prices.shift(1)).dropna()
    cumulative_returns = np.exp(np.cumsum(log_returns)) - 1.0
    
    # Financial KPI Computations
    total_return = cumulative_returns.iloc[-1]
    annualized_return = (1 + total_return) ** (252 / len(log_returns)) - 1
    annualized_vol = log_returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return) / (annualized_vol + 1e-9)
    
    # Drawdown computation
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / (1 + peak)
    max_drawdown = drawdown.min()
    
    return {
        "Total Return (%)": total_return * 100,
        "Annualized Return (%)": annualized_return * 100,
        "Annualized Volatility (%)": annualized_vol * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown * 100
    }
