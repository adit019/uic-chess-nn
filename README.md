# 🧠 UIC Chess Neural Network (PyTorch)

A **residual convolutional neural network** that predicts the next move’s **from** and **to** squares (0–63) from a given chess board state.
Developed as part of an **Undergraduate Research Assistantship** at the **University of Illinois Chicago** (Dept. of Mathematics).

---

## 🎯 Purpose
Explore board encoding and deep learning for chess move prediction.
Demonstrates an end-to-end ML workflow: data prep, tensor encoding, model design, training/validation, and inference.

---

## 🧠 Skills Demonstrated
- PyTorch, CNNs, residual blocks, CrossEntropy
- Parsing SAN, dataset slicing (`start`/`end`/`default`)
- Modular code; checkpoints; validation metrics
- Research communication (poster/demo ready)

---

## 📂 Project Structure
```
src/
  utils.py     # board encoding & SAN parsing
  dataset.py   # PyTorch Dataset + variations
  model.py     # Residual CNN with two heads (from/to)
  train.py     # Training loop, validation split, metrics
  predict.py   # Inference (top-k squares)
requirements.txt
```

---

## 🚀 Setup
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🏋️ Train
```bash
python -m src.train --csv /path/to/chess_games_selected.csv \
  --variation start --num_moves 10 \
  --epochs 5 --batch_size 64 --save_path chessnet.pth
```
**Flags**: `--variation {default,start,end}` · `--num_moves N` · `--limit 40000`

---

## 🔮 Predict
```bash
python -m src.predict --weights chessnet.pth \
  --an "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5"
```
Sample output
```
Top-from squares:
  12 -> e2 (p=0.42)
Top-to squares:
  36 -> e5 (p=0.48)
```

---

## 📝 Notes
Educational baseline; extend with legality masks or a value head.

---

## 📜 License
MIT License

