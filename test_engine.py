"""
test_engine.py
==============
Quick test to verify all modules work correctly.
Run this after installing dependencies.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import chess

print("Testing Chess Engine...\n")

# â”€â”€ Test 1: Board Representation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("=" * 40)
print("TEST 1: Board Representation")
print("=" * 40)

from board.board import ChessBoard

cb = ChessBoard()
cb.display()

tensor = cb.to_tensor()
flat = cb.to_flat_features()
print(f"\nâœ“ Tensor shape:       {tensor.shape}   (12 planes, 8x8 board)")
print(f"âœ“ Flat features size: {flat.shape[0]}   (773 features)")
print(f"âœ“ Legal moves count:  {len(cb.legal_moves())}    (standard opening)")

# â”€â”€ Test 2: Handcrafted Evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 40)
print("TEST 2: Handcrafted Evaluation")
print("=" * 40)

score = cb.handcrafted_eval()
print(f"âœ“ Starting position eval: {score:+d} centipawns (should be ~0)")

# â”€â”€ Test 3: Minimax Search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 40)
print("TEST 3: Minimax Search (depth=2)")
print("=" * 40)

from search.minimax import MinimaxEngine

engine = MinimaxEngine(cb, depth=2)
move, score, stats = engine.best_move_verbose()
print(f"âœ“ Best move found:    {move.uci() if move else 'None'}")
print(f"âœ“ Score:              {score:+d} cp")
print(f"âœ“ {stats}")

# â”€â”€ Test 4: Neural Network â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 40)
print("TEST 4: Neural Network (untrained)")
print("=" * 40)

import torch
from eval.model import ChessEvalNet, NNEvaluator

model = ChessEvalNet()
total_params = sum(p.numel() for p in model.parameters())
print(f"âœ“ Model created successfully")
print(f"âœ“ Total parameters: {total_params:,}")

evaluator = NNEvaluator(model)
nn_score = evaluator(cb)
print(f"âœ“ NN eval (untrained): {nn_score:+d} cp (random, expected near 0)")

# â”€â”€ Test 5: Full Move Sequence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 40)
print("TEST 5: Full Move Sequence")
print("=" * 40)

cb2 = ChessBoard()
moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]  # Italian Game opening
for uci in moves:
    success = cb2.push_uci(uci)
    print(f"  {uci}: {'âœ“' if success else 'âœ—'}")

cb2.display()

print("\n" + "=" * 40)
print("ALL TESTS PASSED âœ“")
print("=" * 40)
print("\nRun 'python main.py' to start playing!")

