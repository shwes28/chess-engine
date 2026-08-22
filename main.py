"""
main.py
=======
Entry point: Play against the chess engine in the terminal.

Usage:
    python main.py                  # Play vs handcrafted eval engine (depth 3)
    python main.py --depth 4        # Deeper search (slower but stronger)
    python main.py --nn models/best_model.pt  # Use neural network eval
    python main.py --train data/games.pgn     # Train the neural network
"""

import argparse
import chess
import sys

from board.board import ChessBoard
from search.minimax import MinimaxEngine


def play_game(depth: int = 3, use_nn: bool = False, model_path: str = None):
    """Interactive game: Human (White) vs Engine (Black)."""

    cb = ChessBoard()
    nn_eval = None

    # Load NN if requested
    if use_nn and model_path:
        from eval.model import ChessEvalNet, NNEvaluator
        model = ChessEvalNet()
        evaluator = NNEvaluator(model)
        evaluator.load(model_path)
        nn_eval = evaluator
        print(f"Using neural network evaluator: {model_path}")
    else:
        print("Using handcrafted evaluation (no NN)")

    engine = MinimaxEngine(cb, depth=depth, use_nn=use_nn, nn_eval=nn_eval)

    print("\n" + "="*50)
    print("   CHESS ENGINE  —  You play White")
    print("   Enter moves in UCI format: e2e4, g1f3, etc.")
    print("   Type 'quit' to exit, 'undo' to take back a move")
    print("="*50 + "\n")

    move_number = 1

    while not cb.is_game_over():
        cb.display()
        print()

        if cb.turn() == chess.WHITE:
            # Human move
            while True:
                user_input = input(f"Move {move_number} (White): ").strip().lower()

                if user_input == "quit":
                    print("Thanks for playing!")
                    sys.exit(0)

                if user_input == "undo":
                    # Undo two moves (human + engine)
                    if len(cb.board.move_stack) >= 2:
                        cb.pop_move()
                        cb.pop_move()
                        move_number -= 1
                        print("Last move undone.\n")
                        cb.display()
                        print()
                    else:
                        print("Nothing to undo.")
                    continue

                if cb.push_uci(user_input):
                    break
                else:
                    legal = [m.uci() for m in cb.legal_moves()]
                    print(f"Illegal move. Examples: {', '.join(legal[:5])}")

        else:
            # Engine move
            print(f"Move {move_number} (Black — Engine thinking at depth {depth})...")
            move, score, stats = engine.best_move_verbose()

            if move is None:
                print("Engine has no legal moves.")
                break

            cb.push_move(move)
            print(f"Engine plays: {move.uci()}  |  Eval: {score:+d} cp  |  {stats}")
            move_number += 1

        print()

    # Game over
    print("\n" + "="*50)
    cb.display()
    result = cb.result()
    if result == "1-0":
        print("White wins! Congratulations!")
    elif result == "0-1":
        print("Black wins! The engine beat you!")
    else:
        print("It's a draw!")
    print("="*50)


def train_nn(pgn_path: str, epochs: int = 20):
    """Train the neural network on PGN data."""
    from training.dataset import load_pgn_file, get_data_loaders
    from training.trainer import Trainer
    from eval.model import ChessEvalNet
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Load data
    features, labels = load_pgn_file(pgn_path, max_games=5000)
    train_loader, val_loader = get_data_loaders(features, labels)

    # Build model
    model = ChessEvalNet()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = Trainer(model, device=device, save_dir="models")
    trainer.train(train_loader, val_loader, epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess Engine")
    parser.add_argument("--depth",  type=int, default=3,    help="Search depth (default: 3)")
    parser.add_argument("--nn",     type=str, default=None, help="Path to trained NN model")
    parser.add_argument("--train",  type=str, default=None, help="Path to PGN file for training")
    parser.add_argument("--epochs", type=int, default=20,   help="Training epochs (default: 20)")
    args = parser.parse_args()

    if args.train:
        train_nn(args.train, epochs=args.epochs)
    else:
        play_game(depth=args.depth, use_nn=bool(args.nn), model_path=args.nn)
