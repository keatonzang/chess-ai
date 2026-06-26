"""AlphaZero-style residual policy/value network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import N_PLANES, N_MOVES


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class ChessNet(nn.Module):
    """Input (B, N_PLANES, 8, 8) -> policy logits (B, N_MOVES), value (B,) in [-1,1]."""

    def __init__(self, channels: int = 128, blocks: int = 10):
        super().__init__()
        self.channels = channels
        self.blocks = blocks

        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        # policy head
        self.p_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(32)
        self.p_fc = nn.Linear(32 * 8 * 8, N_MOVES)

        # value head
        self.v_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(32)
        self.v_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.v_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        h = self.stem(x)
        h = self.res(h)

        p = F.relu(self.p_bn(self.p_conv(h)))
        p = p.flatten(1)
        policy = self.p_fc(p)  # logits

        v = F.relu(self.v_bn(self.v_conv(h)))
        v = v.flatten(1)
        v = F.relu(self.v_fc1(v))
        value = torch.tanh(self.v_fc2(v)).squeeze(-1)

        return policy, value


def build_model(channels: int = 128, blocks: int = 10) -> ChessNet:
    return ChessNet(channels=channels, blocks=blocks)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
