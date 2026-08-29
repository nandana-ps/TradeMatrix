import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 5,
        checkpoint_path: str = "checkpoints/best_model.pt",
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.patience = patience
        self.checkpoint_path = checkpoint_path

        # Huber Loss to mitigate extreme outlier gradient spikes
        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for b_x, b_y in self.train_loader:
            b_x, b_y = b_x.to(self.device), b_y.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(b_x)
            loss = self.criterion(preds, b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for b_x, b_y in loader:
                b_x, b_y = b_x.to(self.device), b_y.to(self.device)
                preds = self.model(b_x)
                total_loss += self.criterion(preds, b_y).item()
        return total_loss / len(loader)

    def fit(self, max_epochs: int = 25) -> dict:
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        best_val_loss = float("inf")
        epochs_no_improve = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, max_epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.evaluate(self.val_loader)
            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train Huber Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered after {epoch} epochs. Best Val Loss: {best_val_loss:.6f}")
                    break

        # Load best saved weights
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        return history