"""
training/trainer.py
===================
Training loop for the chess evaluation network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from tqdm import tqdm

from eval.model import ChessEvalNet


class Trainer:
    """
    Handles model training, validation, and checkpointing.
    """

    def __init__(
        self,
        model: ChessEvalNet,
        device: str = "cpu",
        lr: float = 1e-3,
        save_dir: str = "models",
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, factor=0.5)
        self.criterion = nn.MSELoss()

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = {"train_loss": [], "val_loss": []}
        self.best_val_loss = float("inf")

    def train(self, train_loader, val_loader, epochs: int = 20):
        """Full training loop."""
        print(f"\nTraining on {self.device} for {epochs} epochs\n{'='*50}")

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._val_epoch(val_loader)

            self.scheduler.step(val_loss)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best_model.pt")
                print(f"           ✓ Best model saved (val_loss={val_loss:.4f})")

        # Save training history
        with open(self.save_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")
        print(f"Model saved to: {self.save_dir / 'best_model.pt'}")

    def _train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0

        for features, labels in tqdm(loader, desc="Training", leave=False):
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _val_epoch(self, loader) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

        return total_loss / len(loader)

    def _save_checkpoint(self, filename: str):
        torch.save(self.model.state_dict(), self.save_dir / filename)
