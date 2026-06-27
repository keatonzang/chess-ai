"""Self-play actor: continuously generates self-play games with the latest
network and writes sample shards to the replay buffer. Reloads the learner's
checkpoint as it is updated. No Stockfish involved.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time

import torch

from chessai.model import build_model
from chessai.rl import SelfPlay
from chessai import sources


def build_position_pool(size, puzzles_path, rng):
    """A pool of diverse starting positions: random midgames, endgames, weird
    openings, and puzzle positions (FEN only — no solutions used)."""
    pool = []
    pool += list(sources.gen_random_midgame(int(size * 0.35), rng=rng))
    pool += list(sources.gen_endgames(int(size * 0.25), rng=rng))
    pool += list(sources.gen_openings(int(size * 0.20), weird_frac=0.6, rng=rng))
    if puzzles_path and os.path.exists(puzzles_path):
        with open(puzzles_path) as f:
            puz = [l.strip() for l in f if l.strip()]
        rng.shuffle(puz)
        pool += puz[:int(size * 0.20)]
    rng.shuffle(pool)
    return pool


def latest_ckpt_mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def load_model(ckpt_path, device, channels, blocks):
    model = build_model(channels, blocks).to(device).eval()
    if os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ck["model"] if "model" in ck else ck)
        except Exception as e:
            print(f"[actor] failed to load ckpt: {e}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt", default="../models/rl_current.pt")
    ap.add_argument("--buffer", default="../data/rl_buffer")
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--games", type=int, default=96)
    ap.add_argument("--sims", type=int, default=80)
    ap.add_argument("--max-moves", type=int, default=160)
    ap.add_argument("--temp-moves", type=int, default=24)
    ap.add_argument("--seed-frac", type=float, default=0.4,
                    help="fraction of games started from diverse positions")
    ap.add_argument("--puzzles", default="../data/puzzle_fens.txt")
    ap.add_argument("--pool-size", type=int, default=12000)
    args = ap.parse_args()

    os.makedirs(args.buffer, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    rng = random.Random(1000 + args.id)

    pool = []
    if args.seed_frac > 0:
        pool = build_position_pool(args.pool_size, args.puzzles, rng)
        print(f"[actor {args.id}] diverse start-position pool: {len(pool)}",
              flush=True)

    model = load_model(args.ckpt, device, args.channels, args.blocks)
    ck_mtime = latest_ckpt_mtime(args.ckpt)
    print(f"[actor {args.id}] started on {device} (seed_frac={args.seed_frac})",
          flush=True)

    batch = 0
    while True:
        # reload checkpoint if updated
        m = latest_ckpt_mtime(args.ckpt)
        if m is not None and m != ck_mtime:
            model = load_model(args.ckpt, device, args.channels, args.blocks)
            ck_mtime = m
            print(f"[actor {args.id}] reloaded checkpoint", flush=True)

        sp = SelfPlay(model, device=device, sims=args.sims,
                      max_moves=args.max_moves, temp_moves=args.temp_moves)
        start_fens = None
        if pool:
            start_fens = [rng.choice(pool) if rng.random() < args.seed_frac
                          else None for _ in range(args.games)]
        t = time.time()
        samples, results = sp.play(n_games=args.games, start_fens=start_fens)
        dt = time.time() - t

        # write atomically: temp then rename
        ts = int(time.time() * 1000)
        final = os.path.join(args.buffer, f"worker_{args.id}_{ts}.jsonl")
        tmp = os.path.join(args.buffer, f".tmp_{args.id}_{ts}.jsonl")
        with open(tmp, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        os.rename(tmp, final)

        wins = sum(1 for r in results if r in ("1-0", "0-1"))
        draws = sum(1 for r in results if r == "1/2-1/2")
        batch += 1
        print(f"[actor {args.id}] batch {batch}: {len(samples)} samples "
              f"({len(samples)/dt:.0f}/s) decisive={wins} draws={draws}", flush=True)


if __name__ == "__main__":
    main()
