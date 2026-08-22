"""
ui/app.py
=========
Flask web server for the chess engine.
Supports: difficulty levels, play as Black, opening book, NN eval.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
import chess

from board.board import ChessBoard
from board.opening_book import OpeningBook
from search.minimax import MinimaxEngine
from eval.model import ChessEvalNet, NNEvaluator

app = Flask(__name__)

# ── Config ─────────────────────────────────────────────────────────────
DEPTH_MAP = {"easy": 1, "medium": 3, "hard": 5}

# ── Global state ────────────────────────────────────────────────────────
chess_board  = ChessBoard()
opening_book = OpeningBook()
nn_eval      = None
nn_loaded    = False
current_depth    = 3
player_color     = chess.WHITE   # human plays White by default

def make_engine(depth: int) -> MinimaxEngine:
    return MinimaxEngine(chess_board, depth=depth, use_nn=nn_loaded, nn_eval=nn_eval)

engine = make_engine(current_depth)


def init_nn(model_path: str = None):
    global nn_eval, nn_loaded
    if model_path and os.path.exists(model_path):
        try:
            model = ChessEvalNet()
            ev = NNEvaluator(model)
            ev.load(model_path)
            nn_eval = ev
            nn_loaded = True
            print(f"Neural network loaded from {model_path}")
        except Exception as e:
            print(f"NN load failed: {e}. Using handcrafted eval.")


def engine_move_response():
    """Let the engine (+ opening book) make a move. Returns UCI or None."""
    global engine

    # Try opening book first
    book_move = opening_book.get_book_move(chess_board.board)
    if book_move and book_move in chess_board.board.legal_moves:
        chess_board.board.push(book_move)
        opening_book.record_move(book_move.uci())
        return book_move.uci(), True   # (uci, from_book)

    # Minimax search
    move = engine.best_move()
    if move:
        chess_board.board.push(move)
        opening_book.record_move(move.uci())
        return move.uci(), False

    return None, False


# ── Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", nn_loaded=nn_loaded)


@app.route("/api/board")
def get_board():
    board = chess_board.board
    legal_moves = {}
    for move in board.legal_moves:
        src = chess.square_name(move.from_square)
        dst = chess.square_name(move.to_square)
        legal_moves.setdefault(src, []).append(dst)

    human_color = "white" if player_color == chess.WHITE else "black"
    engine_color = "black" if player_color == chess.WHITE else "white"

    return jsonify({
        "fen":          chess_board.fen(),
        "turn":         "white" if board.turn == chess.WHITE else "black",
        "human_color":  human_color,
        "engine_color": engine_color,
        "game_over":    chess_board.is_game_over(),
        "result":       chess_board.result() if chess_board.is_game_over() else None,
        "in_check":     board.is_check(),
        "legal_moves":  legal_moves,
        "move_count":   len(board.move_stack),
        "eval":         chess_board.handcrafted_eval(),
        "nn_active":    nn_loaded,
        "depth":        current_depth,
    })


@app.route("/api/move", methods=["POST"])
def make_move():
    """Human makes a move, engine responds."""
    global engine

    if chess_board.is_game_over():
        return jsonify({"error": "Game is over."}), 400

    # Sanity check: is it the human's turn?
    if chess_board.board.turn != player_color:
        return jsonify({"error": "Not your turn."}), 400

    data = request.get_json()
    uci  = data.get("move", "").strip()

    try:
        move = chess.Move.from_uci(uci)
        # Handle promotion — default queen
        if move not in chess_board.board.legal_moves:
            move = chess.Move.from_uci(uci + "q")
        if move not in chess_board.board.legal_moves:
            return jsonify({"error": f"Illegal move: {uci}"}), 400
    except Exception:
        return jsonify({"error": f"Invalid format: {uci}"}), 400

    chess_board.board.push(move)
    opening_book.record_move(move.uci())

    if chess_board.is_game_over():
        return jsonify({
            "human_move": move.uci(), "engine_move": None,
            "fen": chess_board.fen(), "game_over": True,
            "result": chess_board.result(), "in_check": False,
            "from_book": False, "eval": chess_board.handcrafted_eval(),
        })

    # Engine responds
    engine_uci, from_book = engine_move_response()

    return jsonify({
        "human_move":  move.uci(),
        "engine_move": engine_uci,
        "fen":         chess_board.fen(),
        "game_over":   chess_board.is_game_over(),
        "result":      chess_board.result() if chess_board.is_game_over() else None,
        "in_check":    chess_board.board.is_check(),
        "from_book":   from_book,
        "eval":        chess_board.handcrafted_eval(),
    })


@app.route("/api/engine_move", methods=["POST"])
def engine_first_move():
    """Used when human plays Black — engine makes the first move."""
    if chess_board.board.turn == player_color:
        return jsonify({"error": "It is the human's turn."}), 400

    engine_uci, from_book = engine_move_response()
    return jsonify({
        "engine_move": engine_uci,
        "fen": chess_board.fen(),
        "game_over": chess_board.is_game_over(),
        "result": chess_board.result() if chess_board.is_game_over() else None,
        "in_check": chess_board.board.is_check(),
        "from_book": from_book,
        "eval": chess_board.handcrafted_eval(),
    })


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """Update difficulty and player color."""
    global current_depth, player_color, engine

    data       = request.get_json()
    difficulty = data.get("difficulty", "medium")
    color      = data.get("color", "white")

    current_depth = DEPTH_MAP.get(difficulty, 3)
    player_color  = chess.WHITE if color == "white" else chess.BLACK
    engine        = make_engine(current_depth)

    # Reset board and book for new settings
    chess_board.reset()
    opening_book.reset()

    return jsonify({
        "depth": current_depth,
        "player_color": color,
        "fen": chess_board.fen(),
    })


@app.route("/api/reset", methods=["POST"])
def reset_game():
    chess_board.reset()
    opening_book.reset()
    return jsonify({"status": "ok", "fen": chess_board.fen()})


@app.route("/api/undo", methods=["POST"])
def undo_move():
    stack = chess_board.board.move_stack
    count = min(2, len(stack))
    for _ in range(count):
        chess_board.pop_move()
        if opening_book.move_history:
            opening_book.move_history.pop()
    return jsonify({"status": "ok", "fen": chess_board.fen()})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nn",    type=str, default="models/best_model.pt")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--port",  type=int, default=5000)
    args = parser.parse_args()

    init_nn(args.nn)
    current_depth = args.depth
    engine = make_engine(current_depth)

    print(f"\nChess Engine at http://localhost:{args.port}\n")
    app.run(debug=False, port=args.port)
