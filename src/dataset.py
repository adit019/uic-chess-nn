
import pandas as pd
import torch
from torch.utils.data import Dataset
import chess
from .utils import create_move_list, board_2_rep, san_list_to_from_to_indices

class ChessDataset(Dataset):
    def __init__(self, csv_path, variation='default', num_moves=10, limit=40000):
        """
        csv_path: CSV with at least 'AN' column (algebraic notation string)
        variation: 'default' (entire game), 'start' (first N moves), 'end' (last N moves)
        num_moves: N for variation slicing
        limit: cap dataset length for faster experiments
        """
        super().__init__()
        self.df = pd.read_csv(csv_path, usecols=['AN'])
        self.df = self.df[~self.df['AN'].astype(str).str.contains(r'\{')]  # drop annotations
        self.df = self.df[self.df['AN'].astype(str).str.len() > 20]       # drop very short
        if limit:
            self.df = self.df.head(limit).reset_index(drop=True)
        self.variation = variation
        self.num_moves = num_moves

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        game = str(self.df.iloc[idx]['AN'])
        moves = create_move_list(game)

        if self.variation == 'start':
            moves_slice = moves[: self.num_moves]
        elif self.variation == 'end':
            moves_slice = moves[- self.num_moves :]
        else:
            moves_slice = moves

        board = chess.Board()
        for san in moves_slice:
            try:
                board.push_san(san)
            except Exception:
                break

        x = board_2_rep(board)  # (6,8,8)

        if len(moves_slice) == 0:
            y_from, y_to = 12, 28  # e2->e4
        else:
            y_from, y_to = san_list_to_from_to_indices(moves_slice)

        x = torch.from_numpy(x)                 # float32
        y_from = torch.tensor(y_from).long()    # class 0..63
        y_to   = torch.tensor(y_to).long()      # class 0..63
        return x, y_from, y_to
