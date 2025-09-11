
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch import optim

from .dataset import ChessDataset
from .model import ChessNet

def accuracy_topk(logits, target, k=1):
    topk = logits.topk(k, dim=1).indices  # (B,k)
    correct = (topk == target.view(-1,1)).any(dim=1).float().mean().item()
    return correct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="/mnt/data/chess_games_selected.csv",
                        help="CSV path with AN column. Defaults to uploaded path if available.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_layers", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=200)
    parser.add_argument("--variation", type=str, default="start", choices=["default","start","end"])
    parser.add_argument("--num_moves", type=int, default=10)
    parser.add_argument("--limit", type=int, default=40000)
    parser.add_argument("--save_path", type=str, default="chessnet.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ChessDataset(args.csv, variation=args.variation, num_moves=args.num_moves, limit=args.limit)
    n = len(ds)
    n_train = int(0.9*n)
    n_val = n - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    model = ChessNet(hidden_layers=args.hidden_layers, hidden_size=args.hidden_size).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y_from, y_to in pbar:
            x = x.to(device)
            y_from = y_from.to(device)
            y_to = y_to.to(device)

            opt.zero_grad()
            logits_from, logits_to = model(x)
            loss = F.cross_entropy(logits_from, y_from) + F.cross_entropy(logits_to, y_to)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            acc_from = 0.0
            acc_to = 0.0
            for x, y_from, y_to in val_loader:
                x = x.to(device); y_from = y_from.to(device); y_to = y_to.to(device)
                lf, lt = model(x)
                loss = F.cross_entropy(lf, y_from) + F.cross_entropy(lt, y_to)
                val_loss += loss.item()
                acc_from += accuracy_topk(lf, y_from, k=1)
                acc_to += accuracy_topk(lt, y_to, k=1)
            val_loss /= max(1, len(val_loader))
            acc_from /= max(1, len(val_loader))
            acc_to /= max(1, len(val_loader))

        print(f"[Epoch {epoch}] train_loss={total_loss/len(train_loader):.4f}  "
              f"val_loss={val_loss:.4f}  acc_from@1={acc_from:.3f}  acc_to@1={acc_to:.3f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"Saved weights to {args.save_path}")

if __name__ == "__main__":
    main()
