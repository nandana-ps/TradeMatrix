import os
import torch
import torch.nn as nn
import numpy as np

class ExportableLSTM(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.head(hn[-1]).squeeze(-1)

def export_champion_model(
    source_chk: str = "checkpoints/ablation_exp_b_(feature-enhanced_lstm).pt",
    target_path: str = "checkpoints/production_model.pt"
):
    """
    Exports the validated champion model weights for production serving.
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(source_chk):
        state_dict = torch.load(source_chk, map_location="cpu")
        torch.save(state_dict, target_path)
        print(f"Champion model checkpoint saved to: {target_path}")
    else:
        print(f"Source checkpoint '{source_chk}' not found. Please verify ablation run.")

class ProductionInferenceEngine:
    def __init__(
        self,
        model_path: str = "checkpoints/production_model.pt",
        feature_dim: int = 5,
        hidden_dim: int = 32,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = ExportableLSTM(input_dim=feature_dim, hidden_dim=hidden_dim)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict_next_return(self, feature_tensor: np.ndarray, tau: float = 0.00170) -> dict:
        """
        Runs low-latency forward pass on the latest (20, 5) feature window.
        """
        if feature_tensor.ndim == 2:
            feature_tensor = np.expand_dims(feature_tensor, axis=0)

        tensor_x = torch.tensor(feature_tensor, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predicted_return = self.model(tensor_x).item()

        if predicted_return > tau:
            signal = "BUY / LONG"
            action = 1
        elif predicted_return < -tau:
            signal = "SELL / SHORT"
            action = -1
        else:
            signal = "HOLD / CASH"
            action = 0

        return {
            "predicted_return": round(predicted_return, 6),
            "signal": signal,
            "action": action,
            "conviction_tau": tau
        }

if __name__ == "__main__":
    export_champion_model()
    engine = ProductionInferenceEngine()
    sample_window = np.random.randn(20, 5)
    result = engine.predict_next_return(sample_window)
    print("\nInference Engine Verification:")
    for k, v in result.items():
        print(f"  {k}: {v}")