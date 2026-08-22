"""
search/minimax.py
=================
Minimax search with Alpha-Beta pruning.

Alpha-Beta pruning dramatically reduces nodes searched:
- Minimax alone: O(b^d) nodes
- Alpha-Beta:    O(b^(d/2)) nodes  (same result, ~square root of work!)

Where b = branching factor (~35 in chess), d = depth.
"""

import chess
import time
from board.board import ChessBoard


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
INF = 999999
CHECKMATE_SCORE = 99999


class SearchStats:
    """Track search statistics for debugging and optimization."""
    def __init__(self):
        self.nodes_searched = 0
        self.alpha_beta_cutoffs = 0
        self.time_elapsed = 0.0

    def reset(self):
        self.nodes_searched = 0
        self.alpha_beta_cutoffs = 0
        self.time_elapsed = 0.0

    def __str__(self):
        return (
            f"Nodes: {self.nodes_searched:,} | "
            f"Cutoffs: {self.alpha_beta_cutoffs:,} | "
            f"Time: {self.time_elapsed:.2f}s | "
            f"NPS: {int(self.nodes_searched / max(self.time_elapsed, 0.001)):,}"
        )


class MinimaxEngine:
    """
    Chess engine using Minimax search with Alpha-Beta pruning.

    Alpha-Beta explanation:
    -----------------------
    - alpha: best score the MAXIMIZER (White) is guaranteed
    - beta:  best score the MINIMIZER (Black) is guaranteed

    If we find a move that gives the opponent a worse position
    than they can already guarantee elsewhere, we PRUNE (skip)
    that entire subtree — it won't affect the final result.
    """

    def __init__(self, chess_board: ChessBoard, depth: int = 3, use_nn: bool = False, nn_eval=None):
        """
        Args:
            chess_board: ChessBoard instance
            depth:       How many half-moves (plies) to search ahead
            use_nn:      Whether to use neural network for evaluation
            nn_eval:     Callable nn_eval(board) -> int (if use_nn=True)
        """
        self.chess_board = chess_board
        self.depth = depth
        self.use_nn = use_nn
        self.nn_eval = nn_eval
        self.stats = SearchStats()

    # ── Public API ────────────────────────────────────────────────────

    def best_move(self) -> chess.Move | None:
        """
        Find the best move for the current player.
        Returns None if no legal moves (game over).
        """
        self.stats.reset()
        start = time.time()

        legal_moves = self.chess_board.legal_moves()
        if not legal_moves:
            return None

        best = None
        # Maximizing = White's turn, Minimizing = Black's turn
        is_white = self.chess_board.turn() == chess.WHITE
        best_score = -INF if is_white else INF

        for move in self._ordered_moves(legal_moves):
            self.chess_board.board.push(move)
            score = self._alpha_beta(
                depth=self.depth - 1,
                alpha=-INF,
                beta=INF,
                maximizing=not is_white,  # opponent's turn next
            )
            self.chess_board.board.pop()

            if is_white and score > best_score:
                best_score = score
                best = move
            elif not is_white and score < best_score:
                best_score = score
                best = move

        self.stats.time_elapsed = time.time() - start
        return best

    def best_move_verbose(self) -> tuple[chess.Move | None, int, SearchStats]:
        """Returns (best_move, score, stats)"""
        move = self.best_move()
        return move, self._evaluate(), self.stats

    # ── Core Alpha-Beta algorithm ─────────────────────────────────────

    def _alpha_beta(self, depth: int, alpha: int, beta: int, maximizing: bool) -> int:
        """
        Recursive Alpha-Beta search.

        Args:
            depth:      Remaining depth to search
            alpha:      Best score maximizer can guarantee (starts at -INF)
            beta:       Best score minimizer can guarantee (starts at +INF)
            maximizing: True if current player wants to maximize score (White)

        Returns:
            The best achievable score from this position
        """
        self.stats.nodes_searched += 1

        # Base cases: game over or depth reached
        if self.chess_board.board.is_game_over():
            return self._evaluate()
        if depth == 0:
            return self._evaluate()

        legal_moves = list(self.chess_board.board.legal_moves)
        ordered_moves = self._ordered_moves(legal_moves)

        if maximizing:
            max_score = -INF
            for move in ordered_moves:
                self.chess_board.board.push(move)
                score = self._alpha_beta(depth - 1, alpha, beta, False)
                self.chess_board.board.pop()

                max_score = max(max_score, score)
                alpha = max(alpha, score)

                # Beta cutoff: minimizer won't allow this
                if beta <= alpha:
                    self.stats.alpha_beta_cutoffs += 1
                    break  # Prune remaining siblings

            return max_score

        else:  # minimizing
            min_score = INF
            for move in ordered_moves:
                self.chess_board.board.push(move)
                score = self._alpha_beta(depth - 1, alpha, beta, True)
                self.chess_board.board.pop()

                min_score = min(min_score, score)
                beta = min(beta, score)

                # Alpha cutoff: maximizer won't allow this
                if beta <= alpha:
                    self.stats.alpha_beta_cutoffs += 1
                    break  # Prune remaining siblings

            return min_score

    # ── Move ordering (huge speedup for Alpha-Beta) ───────────────────

    def _ordered_moves(self, moves: list) -> list:
        """
        Order moves to improve Alpha-Beta efficiency.
        Better move ordering → more cutoffs → faster search.

        Priority:
            1. Captures (sorted by MVV-LVA: Most Valuable Victim, Least Valuable Attacker)
            2. Checks
            3. Other moves
        """
        captures = []
        checks = []
        others = []

        board = self.chess_board.board
        for move in moves:
            if board.is_capture(move):
                captures.append(move)
            elif board.gives_check(move):
                checks.append(move)
            else:
                others.append(move)

        # Sort captures by MVV-LVA score
        captures.sort(key=lambda m: self._mvv_lva(m), reverse=True)

        return captures + checks + others

    def _mvv_lva(self, move: chess.Move) -> int:
        """
        MVV-LVA: Most Valuable Victim - Least Valuable Attacker.
        Score = victim_value * 10 - attacker_value
        Higher score = better capture to try first.
        """
        from board.board import PIECE_VALUES
        board = self.chess_board.board

        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)

        if victim is None or attacker is None:
            return 0

        return PIECE_VALUES[victim.piece_type] * 10 - PIECE_VALUES[attacker.piece_type]

    # ── Evaluation ────────────────────────────────────────────────────

    def _evaluate(self) -> int:
        """Choose between handcrafted eval and neural network eval."""
        if self.use_nn and self.nn_eval is not None:
            return self.nn_eval(self.chess_board)
        return self.chess_board.handcrafted_eval()
