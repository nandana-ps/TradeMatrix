import numpy as np
import pandas as pd
import yfinance as yf

def fetch_and_transform_stock_data(
    ticker: str = "AAPL", 
    start_date: str = "2018-01-01", 
    end_date: str = "2026-01-01"
) -> pd.DataFrame:
    """
    Pulls raw OHLCV market data and transforms it into stationary features 
    and forward target returns with zero lookahead bias.
    """
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
        
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    
    # 1. Stationary Log Returns: r_t = ln(P_t / P_{t-1})
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    
    # 2. Volume Standardized Rate (Volume Z-Score baseline)
    df["vol_zscore"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / (df["Volume"].rolling(20).std() + 1e-9)
    
    # 3. Supervised Forward Target: y_t = r_{t+1} (Predict next day return using time t)
    df["target"] = df["log_ret"].shift(-1)
    
    # 4. Directional Binary Target (for Hit Rate evaluation)
    df["target_direction"] = (df["target"] > 0).astype(int)
    
    # Drop first 20 rows (rolling warmup) and last row (unobserved target)
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    df = fetch_and_transform_stock_data("AAPL")
    df.to_csv("data/processed/AAPL_day1_stationary.csv")
    print(f"Ingested {len(df)} stationary rows. Target statistics:\n{df[['log_ret', 'target']].describe()}")
