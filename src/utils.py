
import re
import numpy as np
import chess

letter_2_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
num_2_letter = {v:k for k,v in letter_2_num.items()}

def create_move_list(s: str):
    """Remove move numbers (e.g., '1. ', '23...') and split SAN tokens; drop results like '1-0'."""
    s = re.sub(r'\d+\.(\.\.)?\s*', '', str(s))
    toks = s.strip().split()
    toks = [t for t in toks if t not in ['1-0','0-1','1/2-1/2','*']]
    return toks

def board_2_rep(board: chess.Board):
    """Return 6x8x8 tensor with channels: p,r,n,b,q,k where +1=white, -1=black, 0=empty."""
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    layers = []
    s = str(board)
    for piece in pieces:
        s_masked = re.sub(f'[^{piece}{piece.upper()} \n]', '.', s)
        s_masked = s_masked.replace(piece, '-1')
        s_masked = s_masked.replace(piece.upper(), '1')
        s_masked = re.sub(r'\.', '0', s_masked)
        mat = []
        for row in s_masked.split('\n'):
            row_vals = [int(x) for x in row.split()]
            mat.append(row_vals)
        layers.append(np.array(mat, dtype=np.float32))
    return np.stack(layers, axis=0)  # (6,8,8)

def san_list_to_from_to_indices(moves_san):
    """Play SAN list and return (from_idx, to_idx) for the LAST move only (0..63 each)."""
    board = chess.Board()
    last_move = None
    for m in moves_san:
        try:
            board.push_san(m)
            last_move = board.move_stack[-1]
        except Exception:
            continue
    if last_move is None:
        last_move = chess.Move.from_uci("e2e4")
    from_sq = last_move.from_square
    to_sq = last_move.to_square
    return from_sq, to_sq  # already 0..63

def idx64_to_coord(idx):
    file = idx % 8
    rank = idx // 8
    return f"{num_2_letter[file]}{rank+1}"
