import numpy as np
from src.data.download import fetch_and_transform_stock_data
from src.models.baseline_lstm import train_baseline_model
from src.backtest.benchmark import calculate_buy_and_hold

print('=== 1. INGESTING DATA ===')
df = fetch_and_transform_stock_data('AAPL')
print(f'Ingested {len(df)} rows.')

print('\n=== 2. SPLITTING DATA (80/20 CHRONOLOGICAL) ===')
split = int(len(df) * 0.8)
features = ['log_ret', 'vol_zscore']

X_train, y_train = df[features].values[:split], df['target'].values[:split]
X_val, y_val = df[features].values[split:], df['target'].values[split:]

print('\n=== 3. TRAINING BASELINE LSTM (5 EPOCHS) ===')
model = train_baseline_model(X_train, y_train, X_val, y_val, epochs=5)

print('\n=== 4. CALCULATING BUY & HOLD BENCHMARK ===')
metrics = calculate_buy_and_hold(df['Close'].iloc[split:])
for k, v in metrics.items():
    print(f'{k:<25}: {v:.2f}')
