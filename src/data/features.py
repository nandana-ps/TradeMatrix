import numpy as np
import pandas as pd

def compute_quantitative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators and volatility features with strict zero lookahead bias.
    All rolling metrics use trailing windows only.
    """
    df = df.copy()

    # 1. 14-Day Trailing Relative Strength Index (RSI)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Exponential rolling mean for smoothing
    avg_gain = gain.ewm(com=13, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # 2. MACD Normalized by Price (12-EMA - 26-EMA)
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = (ema_12 - ema_26) / df["Close"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # 3. 20-Day Trailing Rolling Return Volatility
    df["volatility_20"] = df["log_ret"].rolling(window=20).std()

    # 4. 20-Day Trailing Volume Z-Score
    vol_mean = df["Volume"].rolling(window=20).mean()
    vol_std = df["Volume"].rolling(window=20).std()
    df["volume_zscore"] = (df["Volume"] - vol_mean) / (vol_std + 1e-9)

    # Clean warmup rows created by rolling windows
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    from src.data.download import fetch_and_transform_stock_data
    raw_df = fetch_and_transform_stock_data("AAPL")
    feat_df = compute_quantitative_features(raw_df)
    print(f"Features created successfully: {feat_df.shape}")
    print(feat_df[["log_ret", "rsi_14", "macd", "volatility_20", "volume_zscore"]].tail())