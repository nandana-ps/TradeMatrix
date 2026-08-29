import os
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from src.models.attention_lstm import AttentionLSTM
from src.data.download import fetch_and_transform_stock_data
from src.data.features import compute_quantitative_features
from src.backtest.metrics import compute_institutional_metrics

app = FastAPI(
    title="TradeMatrix Inference & Risk Engine",
    version="1.0.0",
    description="Institutional-grade deep learning trading signal API"
)

# Global model state
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_COLS = ["log_ret", "rsi_14", "macd", "volatility_20", "volume_zscore"]
CHECKPOINT_PATH = "checkpoints/production_model.pt"

model = None
model_meta = {}

@app.on_event("startup")
def load_production_model():
    global model, model_meta
    if not os.path.exists(CHECKPOINT_PATH):
        # Fallback to Day 3 checkpoint if production export not found
        fallback_path = "checkpoints/best_attention_model.pt"
        if os.path.exists(fallback_path):
            checkpoint = torch.load(fallback_path, map_location=DEVICE)
            model = AttentionLSTM(input_dim=len(FEATURE_COLS), hidden_dim=32, num_heads=4)
            model.load_state_dict(checkpoint)
            model.to(DEVICE).eval()
            model_meta = {"tau": 0.00170, "features": FEATURE_COLS}
            return
        raise RuntimeError(f"No checkpoint found at {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = AttentionLSTM(
        input_dim=ckpt.get("feature_dim", len(FEATURE_COLS)),
        hidden_dim=32,
        num_heads=4
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    model_meta = {
        "tau": ckpt.get("conviction_threshold_tau", 0.0017),
        "features": ckpt.get("feature_names", FEATURE_COLS)
    }

class PredictionRequest(BaseModel):
    ticker: str = "AAPL"
    window_size: int = 20

class PredictionResponse(BaseModel):
    ticker: str
    predicted_return: float
    conviction_tau: float
    signal: str
    action: int
    confidence_score: float

@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "online",
        "device": DEVICE,
        "model_loaded": model is not None,
        "metadata": model_meta
    }

@app.post("/predict", response_model=PredictionResponse)
def generate_prediction(req: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    # 1. Fetch recent market window
    raw_df = fetch_and_transform_stock_data(ticker=req.ticker)
    feat_df = compute_quantitative_features(raw_df)
    
    if len(feat_df) < req.window_size:
        raise HTTPException(status_code=400, detail="Insufficient historical data.")

    # 2. Extract trailing lookback window
    window_features = feat_df[FEATURE_COLS].iloc[-req.window_size:].values
    
    # 3. Scale input window
    mean = np.mean(window_features, axis=0)
    std = np.std(window_features, axis=0) + 1e-8
    scaled_window = (window_features - mean) / std

    # 4. Model forward pass
    x_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_return = float(model(x_tensor).cpu().item())

    tau = model_meta.get("tau", 0.0017)
    if pred_return > tau:
        signal, action = "BUY / LONG", 1
    elif pred_return < -tau:
        signal, action = "SELL / SHORT", -1
    else:
        signal, action = "HOLD / CASH", 0

    confidence = min(abs(pred_return) / (tau + 1e-9) * 50.0, 100.0)

    return PredictionResponse(
        ticker=req.ticker,
        predicted_return=pred_return,
        conviction_tau=tau,
        signal=signal,
        action=action,
        confidence_score=round(confidence, 2)
    )

@app.get("/backtest-summary")
def get_backtest_summary(ticker: str = "AAPL") -> Dict[str, Any]:
    raw_df = fetch_and_transform_stock_data(ticker=ticker)
    feat_df = compute_quantitative_features(raw_df)
    
    # Run test slice evaluation
    split = int(len(feat_df) * 0.85)
    test_df = feat_df.iloc[split:].copy()
    
    # Generate simple vectorized benchmark metrics
    market_returns = test_df["target"].dropna().values
    metrics = compute_institutional_metrics(market_returns)
    
    return {
        "ticker": ticker,
        "test_period_start": str(test_df.index[0]),
        "test_period_end": str(test_df.index[-1]),
        "benchmark_metrics": metrics
    }