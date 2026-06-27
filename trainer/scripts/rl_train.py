"""Learner: trains the network on a replay buffer fed by self-play actors.

Maintains an in-memory replay deque, ingesting newly completed shards from the
buffer directory (then deleting them to bound disk). Periodically publishes the
updated checkpoint (which actors reload) and snapshots/export for deployment.

Pure RL: the only training signal is the MCTS visit policy and the eventual
game outcome from self-play. No Stockfish targets.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import time

import numpy as np
import torch

from chessai.model import build_model, count_params
from chessai.encoding import board_to_planes
from chessai.dataset import MAX_POLICY, soft_policy_loss
import chess


def encode_sample(rec):
    board = chess.Board(rec["fen"])
    planes = board_to_planes(board)
    idxs = np.full(MAX_POLICY, -1, dtype=np.int64)
    probs = np.zeros(MAX_POLICY, dtype=np.float32)
    for j, (idx, p) in enumerate(rec["policy"][:MAX_POLICY]):
        idxs[j] = idx
        probs[j] = p
    s = probs.sum()
    if s > 0:
        probs /= s
    return planes, idxs, probs, np.float32(rec["value"])


def save_atomic(model, path, meta):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), **meta}, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../models/rl_current.pt")
    ap.add_argument("--buffer", default="../data/rl_buffer")
    ap.add_argument("--snapdir", default="../models")
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--buffer-cap", type=int, default=500_000)
    ap.add_argument("--min-samples", type=int, default=15_000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--steps-per-round", type=int, default=300,
                    help="max gradient steps per round")
    ap.add_argument("--reuse", type=float, default=20.0,
                    help="target #times each fresh sample is trained on")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.buffer, exist_ok=True)
    os.makedirs(args.snapdir, exist_ok=True)

    model = build_model(args.channels, args.blocks).to(device)
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        print(f"[learner] resumed from {args.resume}", flush=True)
    print(f"[learner] params {count_params(model):,} on {device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler(device.split(":")[0])
    vloss_fn = torch.nn.MSELoss()

    # publish initial (random) checkpoint so actors can start
    save_atomic(model, args.ckpt, {"channels": args.channels,
                                   "blocks": args.blocks, "iter": 0})
    print("[learner] published initial checkpoint", flush=True)

    buffer = collections.deque(maxlen=args.buffer_cap)
    processed = set()
    rounds = 0
    total_games_samples = 0

    def ingest():
        nonlocal total_games_samples
        files = sorted(glob.glob(os.path.join(args.buffer, "worker_*.jsonl")),
                       key=lambda p: os.path.getmtime(p))
        n_new = 0
        for fp in files:
            if fp in processed:
                continue
            try:
                with open(fp) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            buffer.append(line)
                            n_new += 1
            except Exception:
                continue
            processed.add(fp)
            try:
                os.remove(fp)
            except OSError:
                pass
        total_games_samples += n_new
        return n_new

    # wait for the first samples
    while len(buffer) < args.min_samples:
        got = ingest()
        print(f"[learner] warming up: buffer={len(buffer)} (+{got})", flush=True)
        time.sleep(10)

    print("[learner] starting training loop", flush=True)
    while True:
        new = ingest()
        # self-regulate train/generate ratio to ~args.reuse views per sample
        if new < 500 and rounds > 0:
            time.sleep(5)
            continue
        steps = int(max(40, min(args.steps_per_round,
                                new * args.reuse / args.batch)))
        rounds += 1
        model.train()
        run_p = run_v = 0.0
        buf_list = list(buffer)
        for step in range(steps):
            batch_recs = random.sample(buf_list, min(args.batch, len(buf_list)))
            decoded = [encode_sample(json.loads(r)) for r in batch_recs]
            planes = torch.from_numpy(np.stack([d[0] for d in decoded])).to(device)
            idxs = torch.from_numpy(np.stack([d[1] for d in decoded])).to(device)
            probs = torch.from_numpy(np.stack([d[2] for d in decoded])).to(device)
            value = torch.from_numpy(np.array([d[3] for d in decoded])).to(device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.split(":")[0]):
                logits, v = model(planes)
                p_loss = soft_policy_loss(logits, idxs, probs)
                v_loss = vloss_fn(v, value)
                loss = p_loss + args.value_weight * v_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_p += p_loss.item()
            run_v += v_loss.item()

        save_atomic(model, args.ckpt, {"channels": args.channels,
                                       "blocks": args.blocks, "iter": rounds})
        print(f"[learner] round {rounds}: buffer={len(buffer)} (+{new}) "
              f"steps={steps} p={run_p/steps:.3f} v={run_v/steps:.3f} "
              f"seen={total_games_samples:,}", flush=True)

        if rounds % args.snapshot_every == 0:
            snap = os.path.join(args.snapdir, f"rl_iter_{rounds:04d}.pt")
            save_atomic(model, snap, {"channels": args.channels,
                                      "blocks": args.blocks, "iter": rounds})
            print(f"[learner] snapshot {snap}", flush=True)


if __name__ == "__main__":
    main()
