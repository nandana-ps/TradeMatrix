import sys
import os
from pathlib import Path

# Add project root directory to sys.path to enable absolute imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

from src.data.download import fetch_and_transform_stock_data
from src.data.features import compute_quantitative_features

st.set_page_config(
    page_title="TradeMatrix | Quant Intelligence",
    page_icon="📈",
    layout="wide"
)

API_BASE_URL = "http://127.0.0.1:8000"

st.title("⚡ TradeMatrix — Quantitative AI Execution Terminal")
st.caption("Deep Attention-Recurrent Neural Trading Engine with Friction Modeling")

# --- Sidebar Controls ---
st.sidebar.header("Strategy Configurations")
ticker = st.sidebar.text_input("Asset Ticker", value="AAPL")
friction_bps = st.sidebar.slider("Transaction Fee (bps)", min_value=1, max_value=20, value=5)
slippage_bps = st.sidebar.slider("Slippage (bps)", min_value=0, max_value=10, value=2)

# Verify API Health
try:
    health_resp = requests.get(f"{API_BASE_URL}/health", timeout=2).json()
    st.sidebar.success(f"Backend Status: ONLINE ({health_resp.get('device', 'CPU').upper()})")
except Exception:
    st.sidebar.error("Backend Status: OFFLINE (Ensure FastAPI server is running)")

# --- Section 1: Live Inference & Conviction Meter ---
st.header("🎯 Live Inference & Conviction Meter")
col1, col2, col3, col4 = st.columns(4)

if st.button("Generate Today's Action Signal"):
    with st.spinner("Executing forward pass on latest market sequence..."):
        try:
            resp = requests.post(f"{API_BASE_URL}/predict", json={"ticker": ticker, "window_size": 20}, timeout=10).json()
            
            action = resp.get("action", 0)
            signal_color = "green" if action == 1 else "red" if action == -1 else "gray"
            
            col1.metric("Target Asset", resp.get("ticker", ticker))
            col2.metric("Predicted 1-Day Log Return", f"{resp.get('predicted_return', 0.0) * 100:.3f}%")
            col3.markdown(f"### Signal: :{signal_color}[{resp.get('signal', 'HOLD / CASH')}]")
            col4.metric("Confidence Score", f"{resp.get('confidence_score', 0.0)}%")
        except Exception as e:
            st.error(f"Failed to communicate with API: {e}")

st.divider()

# --- Section 2: Out-of-Sample Performance & Drawdown Suite ---
st.header("📊 Out-of-Sample Performance & Drawdown Suite")

@st.cache_data
def load_feature_data(symbol: str):
    raw = fetch_and_transform_stock_data(symbol)
    return compute_quantitative_features(raw)

try:
    feat_df = load_feature_data(ticker)
    split = int(len(feat_df) * 0.85)
    test_df = feat_df.iloc[split:].copy()

    # Vectorized return simulation
    returns = test_df["target"].values
    signals = np.sign(test_df["macd"].values)
    friction = (friction_bps + slippage_bps) / 10000.0
    pos_changes = np.abs(np.diff(signals, prepend=0))
    net_strategy = (signals * returns) - (pos_changes * friction)

    strat_equity = np.cumprod(1 + net_strategy)
    bench_equity = np.cumprod(1 + returns)

    # 1. Equity Curves Plot
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        y=strat_equity, mode="lines", name="Attention-LSTM (Net of Fees)",
        line=dict(color="#00FFA3", width=2)
    ))
    fig_equity.add_trace(go.Scatter(
        y=bench_equity, mode="lines", name="Passive Buy & Hold",
        line=dict(color="#888888", dash="dash")
    ))
    fig_equity.update_layout(
        title="Out-of-Sample Cumulative Equity Curves",
        xaxis_title="Trading Days",
        yaxis_title="Growth of $1.00",
        template="plotly_dark",
        height=420
    )
    st.plotly_chart(fig_equity, use_container_width=True)

    # 2. Underwater Drawdown Plot
    running_max = np.maximum.accumulate(strat_equity)
    drawdown = (strat_equity - running_max) / running_max

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        y=drawdown * 100, fill="tozeroy", mode="lines",
        name="Underwater Drawdown", line=dict(color="#FF4B4B")
    ))
    fig_dd.update_layout(
        title="Strategy Underwater Drawdown Profile (%)",
        xaxis_title="Trading Days",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
        height=280
    )
    st.plotly_chart(fig_dd, use_container_width=True)

except Exception as err:
    st.error(f"Error loading backtest data: {err}")