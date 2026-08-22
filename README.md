# ♟️ Chess Engine with Neural Network Evaluation

> A chess engine built from scratch in Python — combining classical **Minimax + Alpha-Beta pruning** with a **neural network evaluator** trained on grandmaster games.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **Neural Network Eval** | 570K-param network trained on 400+ grandmaster games |
| ♟️ **Opening Book** | Covers Ruy Lopez, Sicilian, Italian, French, Caro-Kann, QGD, KID |
| 🎯 **Difficulty Levels** | Easy (depth 1) · Medium (depth 3) · Hard (depth 5) |
| 🔄 **Play as Black** | Board flips, engine moves first |
| 📊 **Live Eval Bar** | Real-time position evaluation in centipawns |
| 🔁 **Undo / Reset** | Take back moves anytime |
| 🌐 **Web UI** | Drag & drop chessboard in the browser |

---

## 🏗️ Architecture

```
Board Position (8×8)
      ↓
Feature Vector (773,)     ← 12 piece planes + castling + side to move
      ↓
Neural Network            ← Linear(773→512→256→128→64→1) + BatchNorm + Dropout
      ↓
Position Score (centipawns)
      ↓
Minimax + Alpha-Beta      ← depth 1–5, MVV-LVA move ordering
      ↓
Best Move
```

---

## 🚀 Quick Start

### 1. Clone
```bash
git clone https://github.com/shwes28/chess-engine.git
cd chess-engine
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download training data & train
```bash
# Download ~1000 grandmaster games from Lichess
python data/download.py --games 1000 --output data/games.pgn

# Train the neural network
python main.py --train data/games.pgn --epochs 20
```

### 4. Launch the web UI
```bash
python ui/app.py --nn models/best_model.pt --depth 3
# Open http://localhost:5000
```

### 5. Or play in the terminal
```bash
python main.py                        # Handcrafted eval
python main.py --nn models/best_model.pt  # Neural network eval
python main.py --depth 4              # Stronger search
```

---

## 📁 Project Structure

```
chess-engine/
├── board/
│   ├── board.py          # Board representation, tensor encoding, PST eval
│   └── opening_book.py   # Opening theory (10+ openings)
├── search/
│   └── minimax.py        # Minimax + Alpha-Beta + MVV-LVA move ordering
├── eval/
│   └── model.py          # PyTorch neural network + evaluator wrapper
├── training/
│   ├── dataset.py        # PGN parser → PyTorch Dataset
│   └── trainer.py        # Training loop with checkpointing
├── data/
│   └── download.py       # Lichess API game downloader
├── ui/
│   ├── app.py            # Flask REST API
│   └── templates/
│       └── index.html    # Drag & drop web interface
├── models/               # Saved model weights (not tracked by git)
├── main.py               # CLI entry point
├── test_engine.py        # Module tests
└── requirements.txt
```

---

## 🧠 How It Works

### Board Representation
Each position is encoded as a **(12 × 8 × 8)** tensor:
- 6 planes for White pieces (P, N, B, R, Q, K)
- 6 planes for Black pieces
- Flattened + castling rights + side to move = **773 features**

### Neural Network Training
- **Data:** Games downloaded from the [Lichess Open Database](https://database.lichess.org/)
- **Labels:** Game result (+1 White won, −1 Black won, 0 Draw)
- **Loss:** MSE between predicted score and actual outcome
- **Trained in:** <30 seconds on CPU

### Alpha-Beta Pruning
```
Without pruning:  O(b^d)     nodes  (b≈35 moves, d=depth)
With Alpha-Beta:  O(b^(d/2)) nodes  → ~10× speedup
With MVV-LVA:     Even fewer → captures tried first
```

---

## 📊 Training Results

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 1.023 | 0.896 |
| 5 | 0.379 | 0.490 |
| 10 | 0.199 | 0.450 |

---

## 🛠️ Tech Stack

- **python-chess** — Legal move generation, FEN parsing, PGN reading
- **PyTorch** — Neural network training and inference
- **Flask** — REST API backend
- **chessboard.js** — Interactive drag & drop board
- **chess.js** — Frontend move validation

---

## 📈 Roadmap

- [x] Board representation + feature extraction
- [x] Minimax + Alpha-Beta pruning
- [x] MVV-LVA move ordering
- [x] Handcrafted evaluation (material + PST)
- [x] Neural network evaluator
- [x] Training pipeline (PGN → Dataset → Model)
- [x] Opening book
- [x] Web UI with difficulty levels + play as Black
- [ ] MCTS (AlphaZero style)
- [ ] Endgame tablebases
- [ ] UCI protocol (play vs Stockfish)
- [ ] Cloud deployment

---

## 📚 References

- [Chess Programming Wiki](https://www.chessprogramming.org)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [Lichess Open Database](https://database.lichess.org/)
- [python-chess Documentation](https://python-chess.readthedocs.io/)

---

## 📄 License

MIT License — feel free to use, modify and distribute.
