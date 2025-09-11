
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SELU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out

class ChessNet(nn.Module):
    def __init__(self, hidden_layers=4, hidden_size=200):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(6, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_size) for _ in range(hidden_layers)])
        self.head_from = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(hidden_size*8*8, 64)
        )
        self.head_to = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(hidden_size*8*8, 64)
        )

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        logits_from = self.head_from(h)  # (B,64)
        logits_to   = self.head_to(h)    # (B,64)
        return logits_from, logits_to
