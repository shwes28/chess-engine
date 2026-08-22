"""
training/dataset.py
===================
Dataset utilities for training the chess evaluation network.

Data source: Lichess database (https://database.lichess.org/)
Format:      PGN (Portable Game Notation) files

What we do:
    1. Parse PGN files to extract board positions
    2. Use the game result as a training label (+1 = White won, -1 = Black won)
    3. Convert each position to our feature vector
    4. Train the NN to predict the outcome from the position
"""

import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from tqdm import tqdm


# Label mapping: game result → float
RESULT_TO_LABEL = {
    "1-0":     1.0,   # White won
    "0-1":    -1.0,   # Black won
    "1/2-1/2": 0.0,   # Draw
}


class ChessPositionDataset(Dataset):
    """
    PyTorch Dataset of chess positions.
    Each sample: (feature_vector, label)
        feature_vector: (773,) float32
        label:          float in {-1.0, 0.0, 1.0}
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_pgn_file(pgn_path: str, max_games: int = 5000, positions_per_game: int = 10) -> tuple:
    """
    Parse a PGN file and extract board positions with labels.

    Args:
        pgn_path:            Path to the .pgn file
        max_games:           Max number of games to parse (for speed)
        positions_per_game:  Random positions to sample per game

    Returns:
        (features, labels): numpy arrays
    """
    features_list = []
    labels_list = []

    pgn_file = open(pgn_path, encoding="utf-8", errors="ignore")
    games_processed = 0

    print(f"Loading games from {pgn_path}...")

    with tqdm(total=max_games, desc="Parsing PGN") as pbar:
        while games_processed < max_games:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break  # End of file

            result = game.headers.get("Result", "*")
            if result not in RESULT_TO_LABEL:
                continue

            label = RESULT_TO_LABEL[result]

            # Collect all positions in this game
            board = game.board()
            positions = []
            for move in game.mainline_moves():
                board.push(move)
                positions.append(board.fen())

            if not positions:
                continue

            # Sample random positions from the game
            sampled = random.sample(positions, min(positions_per_game, len(positions)))
            for fen in sampled:
                b = chess.Board(fen)
                feat = _board_to_features(b)
                features_list.append(feat)
                labels_list.append(label)

            games_processed += 1
            pbar.update(1)

    pgn_file.close()

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)

    print(f"Loaded {len(features):,} positions from {games_processed:,} games")
    return features, labels


def _board_to_features(board: chess.Board) -> np.ndarray:
    """
    Convert a chess.Board to a flat feature vector (773,).
    Mirrors ChessBoard.to_flat_features() but works on raw chess.Board.
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                   chess.ROOK, chess.QUEEN, chess.KING]

    for i, piece_type in enumerate(piece_types):
        for square in board.pieces(piece_type, chess.WHITE):
            row, col = divmod(square, 8)
            tensor[i][row][col] = 1.0
        for square in board.pieces(piece_type, chess.BLACK):
            row, col = divmod(square, 8)
            tensor[i + 6][row][col] = 1.0

    flat = tensor.flatten()
    extras = np.array([
        float(board.has_kingside_castling_rights(chess.WHITE)),
        float(board.has_queenside_castling_rights(chess.WHITE)),
        float(board.has_kingside_castling_rights(chess.BLACK)),
        float(board.has_queenside_castling_rights(chess.BLACK)),
        float(board.turn),
    ], dtype=np.float32)

    return np.concatenate([flat, extras])


def get_data_loaders(features: np.ndarray, labels: np.ndarray,
                     train_split: float = 0.9, batch_size: int = 256):
    """
    Split data and create PyTorch DataLoaders.

    Returns:
        (train_loader, val_loader)
    """
    n = len(features)
    split = int(n * train_split)

    # Shuffle
    indices = np.random.permutation(n)
    features = features[indices]
    labels = labels[indices]

    train_ds = ChessPositionDataset(features[:split], labels[:split])
    val_ds = ChessPositionDataset(features[split:], labels[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    return train_loader, val_loader
