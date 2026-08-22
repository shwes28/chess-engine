"""
eval/model.py
=============
Neural Network for chess position evaluation.

Architecture:
    Input:  Flat board features (773,)
    Hidden: 3 fully connected layers with BatchNorm + Dropout
    Output: Single scalar (centipawn evaluation)

Positive output = good for White
Negative output = good for Black
"""

import torch
import torch.nn as nn
import numpy as np


class ChessEvalNet(nn.Module):
    """
    Feed-forward neural network for chess position evaluation.
    
    Input features (773 total):
        - 12 * 64 = 768: one-hot board planes (piece type + color per square)
        - 5: extra features (castling rights x4, side to move x1)
    
    Output:
        - Single float: position evaluation in centipawns
          (scaled to [-1, 1] during training, then multiplied by 10000)
    """

    def __init__(self, input_size: int = 773, hidden_sizes: list = None, dropout: float = 0.3):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128, 64]

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size

        # Output layer (no activation — raw evaluation score)
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Tanh())  # Clamp output to [-1, 1]

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, 773)
        Returns:
            Tensor of shape (batch_size, 1) with values in [-1, 1]
        """
        return self.network(x)


class NNEvaluator:
    """
    Wraps ChessEvalNet for use in the search engine.
    Converts board → features → NN call → centipawn score.
    """

    def __init__(self, model: ChessEvalNet, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # Scale factor: NN outputs [-1, 1], we scale to centipawns
        self.scale = 10000

    def __call__(self, chess_board) -> int:
        """
        Evaluate a board position using the neural network.
        
        Args:
            chess_board: ChessBoard instance
        Returns:
            int: centipawn score (positive = good for White)
        """
        features = chess_board.to_flat_features()  # (773,)
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            score = self.model(tensor).item()

        return int(score * self.scale)

    def save(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {path}")
