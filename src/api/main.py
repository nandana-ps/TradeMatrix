import sys
import os
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from src.models.export import ProductionInferenceEngine
from src.data.download import fetch_and_transform_stock_data
from src.data.features import compute_quantitative_features
from src.backtest.metrics import compute_institutional_metrics

app = FastAPI(
    title="TradeMatrix Inference & Risk Engine",
    version="1.0.0",
    description="Institutional-grade deep learning trading signal API"
)

FEATURE_COLS = ["log_ret", "rsi_14", "macd", "volatility_20", "volume_zscore"]
CHECKPOINT_PATH = "checkpoints/production_model.pt"
TAU = 0.00170

engine: ProductionInferenceEngine = None

@app.on_event("startup")
def load_production_engine():
    global engine
    if not os.path.exists(CHECKPOINT_PATH):
        raise RuntimeError(f"Production checkpoint not found at {CHECKPOINT_PATH}. Run 'python -m src.models.export' first.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = ProductionInferenceEngine(
        model_path=CHECKPOINT_PATH,
        feature_dim=len(FEATURE_COLS),
        hidden_dim=32,
        device=device
    )

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
        "device": str(engine.device) if engine else "offline",
        "model_loaded": engine is not None,
        "tau": TAU
    }

@app.post("/predict", response_model=PredictionResponse)
def generate_prediction(req: PredictionRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized.")

    raw_df = fetch_and_transform_stock_data(ticker=req.ticker)
    feat_df = compute_quantitative_features(raw_df)

    if len(feat_df) < req.window_size:
        raise HTTPException(status_code=400, detail="Insufficient historical data window.")

    window = feat_df[FEATURE_COLS].iloc[-req.window_size:].values
    
    # Standardize input window
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0) + 1e-8
    scaled_window = (window - mean) / std

    res = engine.predict_next_return(scaled_window, tau=TAU)
    confidence = min(abs(res["predicted_return"]) / (TAU + 1e-9) * 50.0, 100.0)

    return PredictionResponse(
        ticker=req.ticker,
        predicted_return=res["predicted_return"],
        conviction_tau=res["conviction_tau"],
        signal=res["signal"],
        action=res["action"],
        confidence_score=round(confidence, 2)
    )

@app.get("/backtest-summary")
def get_backtest_summary(ticker: str = "AAPL") -> Dict[str, Any]:
    raw_df = fetch_and_transform_stock_data(ticker=ticker)
    feat_df = compute_quantitative_features(raw_df)
    
    split = int(len(feat_df) * 0.85)
    test_df = feat_df.iloc[split:].copy()
    
    market_returns = test_df["target"].dropna().values
    metrics = compute_institutional_metrics(market_returns)
    
    return {
        "ticker": ticker,
        "test_period_start": str(test_df.index[0]),
        "test_period_end": str(test_df.index[-1]),
        "benchmark_metrics": metrics
    }