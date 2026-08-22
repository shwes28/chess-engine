"""
board/board.py
==============
Chess board representation and utilities using python-chess.

This is Phase 1 of the chess engine.
python-chess handles all the complex rules (castling, en passant, promotion, etc.)
so we focus on the engine logic.
"""

import chess
import chess.svg
import numpy as np


# ──────────────────────────────────────────────
# Piece values (used in handcrafted eval later)
# ──────────────────────────────────────────────
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}

# Piece-Square Tables (encourage good piece positioning)
# Values are from White's perspective. Flip for Black.
# Source: chess programming wiki (standard PST tables)

PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

KING_TABLE_MIDDLEGAME = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

PST = {
    chess.PAWN:   PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK:   ROOK_TABLE,
    chess.QUEEN:  QUEEN_TABLE,
    chess.KING:   KING_TABLE_MIDDLEGAME,
}


class ChessBoard:
    """
    Wrapper around python-chess Board.
    Provides board utilities needed by the search and eval modules.
    """

    def __init__(self, fen: str = chess.STARTING_FEN):
        self.board = chess.Board(fen)

    # ── Basic helpers ─────────────────────────────────────────────────

    def reset(self):
        """Reset to starting position."""
        self.board.reset()

    def push_move(self, move: chess.Move) -> bool:
        """
        Apply a move to the board.
        Returns True if successful, False if illegal.
        """
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False

    def push_uci(self, uci: str) -> bool:
        """Apply a move given in UCI format (e.g. 'e2e4')."""
        try:
            move = chess.Move.from_uci(uci)
            return self.push_move(move)
        except Exception:
            return False

    def pop_move(self):
        """Undo the last move."""
        if self.board.move_stack:
            self.board.pop()

    def legal_moves(self):
        """Return list of all legal moves."""
        return list(self.board.legal_moves)

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def result(self) -> str:
        """Returns '1-0', '0-1', '1/2-1/2', or '*'"""
        return self.board.result()

    def turn(self) -> chess.Color:
        """Returns chess.WHITE or chess.BLACK"""
        return self.board.turn

    def fen(self) -> str:
        return self.board.fen()

    # ── Feature extraction for neural network ─────────────────────────

    def to_tensor(self) -> np.ndarray:
        """
        Convert board to a (12, 8, 8) numpy array — one plane per piece type per color.
        This is the input representation for the neural network.

        Planes:
            0-5  : White pieces  (P, N, B, R, Q, K)
            6-11 : Black pieces  (P, N, B, R, Q, K)
        """
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                       chess.ROOK, chess.QUEEN, chess.KING]

        for i, piece_type in enumerate(piece_types):
            for square in self.board.pieces(piece_type, chess.WHITE):
                row, col = divmod(square, 8)
                tensor[i][row][col] = 1.0

            for square in self.board.pieces(piece_type, chess.BLACK):
                row, col = divmod(square, 8)
                tensor[i + 6][row][col] = 1.0

        return tensor

    def to_flat_features(self) -> np.ndarray:
        """
        Flat feature vector for a simpler NN architecture.
        Shape: (773,) — 768 board squares + 5 extra features

        Extra features: castling rights (4) + side to move (1)
        """
        flat = self.to_tensor().flatten()  # 12 * 8 * 8 = 768

        extras = np.array([
            float(self.board.has_kingside_castling_rights(chess.WHITE)),
            float(self.board.has_queenside_castling_rights(chess.WHITE)),
            float(self.board.has_kingside_castling_rights(chess.BLACK)),
            float(self.board.has_queenside_castling_rights(chess.BLACK)),
            float(self.board.turn),  # 1.0 = White to move
        ], dtype=np.float32)

        return np.concatenate([flat, extras])

    # ── Handcrafted evaluation (baseline before NN) ───────────────────

    def handcrafted_eval(self) -> int:
        """
        Simple material + positional evaluation.
        Positive = good for White, Negative = good for Black.
        """
        if self.board.is_checkmate():
            return -99999 if self.board.turn == chess.WHITE else 99999
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0

        score = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None:
                continue

            value = PIECE_VALUES[piece.piece_type]
            # PST index: flip for Black (mirror vertically)
            pst_index = square if piece.color == chess.WHITE else chess.square_mirror(square)
            positional = PST[piece.piece_type][pst_index]

            if piece.color == chess.WHITE:
                score += value + positional
            else:
                score -= value + positional

        return score

    # ── Display ───────────────────────────────────────────────────────

    def display(self):
        """Print the board to the terminal."""
        print(self.board)
        print(f"\nFEN: {self.fen()}")
        print(f"Turn: {'White' if self.board.turn == chess.WHITE else 'Black'}")
        print(f"Legal moves: {len(self.legal_moves())}")
        if self.is_game_over():
            print(f"Game over! Result: {self.result()}")

    def __str__(self):
        return str(self.board)
