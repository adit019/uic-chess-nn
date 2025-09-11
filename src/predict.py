
import argparse
import torch
import torch.nn.functional as F
from .model import ChessNet
from .utils import create_move_list, idx64_to_coord, board_2_rep
import chess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--an", required=True, help="Algebraic notation string, e.g., '1. e4 e5 2. Nf3 Nc6'")
    args = ap.parse_args()

    toks = create_move_list(args.an)
    board = chess.Board()
    for t in toks:
        try:
            board.push_san(t)
        except Exception:
            pass

    x = board_2_rep(board)  # (6,8,8)
    x = torch.from_numpy(x).unsqueeze(0)  # (1,6,8,8)

    model = ChessNet()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        lf, lt = model(x)
        pf = F.softmax(lf, dim=1)[0]
        pt = F.softmax(lt, dim=1)[0]
        topk = 5
        topf = torch.topk(pf, k=topk)
        topt = torch.topk(pt, k=topk)

    print("Top-from squares:")
    for p, idx in zip(topf.values.tolist(), topf.indices.tolist()):
        print(f"  {idx:2d} -> {idx64_to_coord(idx)}  (p={p:.3f})")

    print("Top-to squares:")
    for p, idx in zip(topt.values.tolist(), topt.indices.tolist()):
        print(f"  {idx:2d} -> {idx64_to_coord(idx)}  (p={p:.3f})")

if __name__ == "__main__":
    main()
