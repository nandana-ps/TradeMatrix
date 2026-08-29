import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def audit_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Performs stationarity checks (ADF test) and distributional summary
    for all engineered feature inputs.
    """
    audit_records = []

    for col in feature_cols:
        series = df[col].dropna()
        adf_result = adfuller(series, autolag="AIC")
        p_val = adf_result[1]
        test_stat = adf_result[0]
        crit_5pct = adf_result[4]["5%"]

        audit_records.append({
            "Feature": col,
            "ADF Stat": round(test_stat, 4),
            "p-value": round(p_val, 6),
            "5% Crit Value": round(crit_5pct, 4),
            "Stationary (p < 0.05)": p_val < 0.05,
            "Mean": round(series.mean(), 4),
            "Std Dev": round(series.std(), 4),
            "Skewness": round(series.skew(), 4),
            "Kurtosis": round(series.kurtosis(), 4)
        })

    audit_df = pd.DataFrame(audit_records)
    return audit_df

def compute_correlation_matrix(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Calculates pairwise Pearson correlation across all feature inputs.
    """
    return df[feature_cols].corr()

if __name__ == "__main__":
    from src.data.download import fetch_and_transform_stock_data
    from src.data.features import compute_quantitative_features

    raw_df = fetch_and_transform_stock_data("AAPL")
    feat_df = compute_quantitative_features(raw_df)
    features = ["log_ret", "rsi_14", "macd", "volatility_20", "volume_zscore"]

    print("=== FEATURE AUDIT & ADF STATIONARITY REPORT ===")
    report = audit_features(feat_df, features)
    print(report.to_string(index=False))

    print("\n=== FEATURE CORRELATION MATRIX ===")
    corr_matrix = compute_correlation_matrix(feat_df, features)
    print(corr_matrix.round(3))