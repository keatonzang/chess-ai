"""Dataset over JSONL shards produced by datagen.

Each record: {"fen", "value", "policy": [[idx, prob], ...]}.
Planes are encoded on the fly; policy targets are returned sparsely and turned
into a dense soft target inside the loss for memory efficiency.
"""

from __future__ import annotations

import glob
import json
import os

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from .encoding import board_to_planes, N_MOVES, N_PLANES

MAX_POLICY = 8  # max stored policy entries per position


class ShardDataset(Dataset):
    def __init__(self, shard_glob: str, limit: int | None = None):
        self.records = []
        paths = sorted(glob.glob(shard_glob))
        if not paths:
            raise FileNotFoundError(f"no shards match {shard_glob}")
        for p in paths:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.records.append(line)
                    if limit and len(self.records) >= limit:
                        break
            if limit and len(self.records) >= limit:
                break

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = json.loads(self.records[i])
        board = chess.Board(rec["fen"])
        planes = board_to_planes(board)

        idxs = np.full(MAX_POLICY, -1, dtype=np.int64)
        probs = np.zeros(MAX_POLICY, dtype=np.float32)
        for j, (idx, p) in enumerate(rec["policy"][:MAX_POLICY]):
            idxs[j] = idx
            probs[j] = p
        # renormalize in case of truncation
        s = probs.sum()
        if s > 0:
            probs /= s

        return (
            torch.from_numpy(planes),
            torch.from_numpy(idxs),
            torch.from_numpy(probs),
            torch.tensor(rec["value"], dtype=torch.float32),
        )


def soft_policy_loss(logits: torch.Tensor, idxs: torch.Tensor,
                     probs: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between predicted log-softmax and sparse soft targets.

    logits: (B, N_MOVES); idxs: (B, K) with -1 padding; probs: (B, K).
    """
    logp = torch.log_softmax(logits, dim=1)  # (B, N_MOVES)
    mask = (idxs >= 0).float()               # (B, K)
    safe_idx = idxs.clamp(min=0)
    gathered = torch.gather(logp, 1, safe_idx)  # (B, K)
    loss = -(probs * gathered * mask).sum(dim=1)
    return loss.mean()
