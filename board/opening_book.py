"""
board/opening_book.py
=====================
Simple opening book with well-known chess openings.
Maps FEN position (just the move sequence hash) to a list of good replies.
Engine picks randomly from book moves for variety.
"""

import random
import chess

# Opening book: maps board FEN prefix → list of good UCI moves
# Format: "moves played so far" → [candidate moves]
# Using move sequences for reliability across FEN variations

BOOK = {
    # === Starting position ===
    "": ["e2e4", "d2d4", "g1f3", "c2c4"],

    # === After 1. e4 ===
    "e2e4": ["e7e5", "c7c5", "e7e6", "c7c6", "d7d5", "g8f6"],

    # === After 1. d4 ===
    "d2d4": ["d7d5", "g8f6", "e7e6", "c7c5", "f7f5"],

    # === After 1. e4 e5 (Open Game) ===
    "e2e4 e7e5": ["g1f3", "f2f4", "d2d4", "f1c4"],

    # === After 1. e4 e5 2. Nf3 (Ruy Lopez / Italian / Scotch) ===
    "e2e4 e7e5 g1f3": ["b8c6", "g8f6", "d7d6"],

    # === After 1. e4 e5 2. Nf3 Nc6 ===
    "e2e4 e7e5 g1f3 b8c6": ["f1b5", "f1c4", "d2d4", "b1c3"],

    # === Ruy Lopez: 1. e4 e5 2. Nf3 Nc6 3. Bb5 ===
    "e2e4 e7e5 g1f3 b8c6 f1b5": ["a7a6", "g8f6", "f8c5", "d7d6"],

    # === Italian: 1. e4 e5 2. Nf3 Nc6 3. Bc4 ===
    "e2e4 e7e5 g1f3 b8c6 f1c4": ["f8c5", "g8f6", "h7h6"],

    # === Sicilian: 1. e4 c5 ===
    "e2e4 c7c5": ["g1f3", "b1c3", "c2c3"],

    # === After 1. e4 c5 2. Nf3 ===
    "e2e4 c7c5 g1f3": ["d7d6", "b8c6", "e7e6", "g7g6"],

    # === French: 1. e4 e6 ===
    "e2e4 e7e6": ["d2d4", "d2d3", "g1f3"],

    # === French 1. e4 e6 2. d4 ===
    "e2e4 e7e6 d2d4": ["d7d5", "b8c6"],

    # === Caro-Kann: 1. e4 c6 ===
    "e2e4 c7c6": ["d2d4", "b1c3", "g1f3"],

    # === Queen's Gambit: 1. d4 d5 2. c4 ===
    "d2d4 d7d5": ["c2c4", "g1f3", "b1c3"],
    "d2d4 d7d5 c2c4": ["e7e6", "c7c6", "d5c4", "g8f6"],

    # === King's Indian: 1. d4 Nf6 ===
    "d2d4 g8f6": ["c2c4", "g1f3", "b1c3"],
    "d2d4 g8f6 c2c4": ["g7g6", "e7e6", "c7c5"],

    # === 1. Nf3 ===
    "g1f3": ["d7d5", "g8f6", "c7c5", "e7e6"],
}


class OpeningBook:
    """Simple opening book — returns a book move if available."""

    def __init__(self):
        self.move_history = []  # list of UCI strings

    def reset(self):
        self.move_history = []

    def record_move(self, uci: str):
        self.move_history.append(uci)

    def get_book_move(self, board: chess.Board) -> chess.Move | None:
        """
        Return a book move for current position, or None if out of book.
        """
        key = " ".join(self.move_history)
        candidates = BOOK.get(key)

        if not candidates:
            return None

        # Filter to only legal moves
        legal_ucis = {m.uci() for m in board.legal_moves}
        valid = [m for m in candidates if m in legal_ucis]

        if not valid:
            return None

        chosen = random.choice(valid)
        return chess.Move.from_uci(chosen)
