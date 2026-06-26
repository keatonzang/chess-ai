"""Continuously generate a diverse, Stockfish-labeled training dataset.

Runs in rounds, mixing position sources, appending JSONL shards under --out
until --target records exist. Safe to stop/restart; it counts existing records.

Example:
  python -m scripts.generate --target 400000 --out ../data/shards \
      --sf ../tools/stockfish/stockfish-ubuntu-x86-64-avx2 \
      --puzzles ../data/lichess_puzzles.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import time

from chessai import sources, datagen


def count_records(out_dir: str) -> int:
    total = 0
    for p in glob.glob(os.path.join(out_dir, "*.jsonl")):
        with open(p) as f:
            for _ in f:
                total += 1
    return total


def current_shard(out_dir: str, max_per_shard: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(out_dir, "shard_*.jsonl")))
    if not shards:
        return os.path.join(out_dir, "shard_0000.jsonl")
    last = shards[-1]
    with open(last) as f:
        n = sum(1 for _ in f)
    if n >= max_per_shard:
        idx = int(os.path.basename(last).split("_")[1].split(".")[0]) + 1
        return os.path.join(out_dir, f"shard_{idx:04d}.jsonl")
    return last


def make_round_fens(round_size: int, puzzles_path: str | None,
                    rng: random.Random) -> list[str]:
    """Mix sources for one round. Ratios tuned for broad coverage."""
    fens: list[str] = []
    # ratios
    n_open = int(round_size * 0.30)
    n_mid = int(round_size * 0.25)
    n_end = int(round_size * 0.20)
    n_puz = round_size - n_open - n_mid - n_end  # ~0.25

    fens += list(sources.gen_openings(n_open, rng=rng))
    fens += list(sources.gen_random_midgame(n_mid, rng=rng))
    fens += list(sources.gen_endgames(n_end, rng=rng))
    if puzzles_path and os.path.exists(puzzles_path):
        fens += list(sources.gen_puzzles(puzzles_path, n_puz, rng=rng))
    else:
        fens += list(sources.gen_random_midgame(n_puz, rng=rng))

    # de-dup within the round, shuffle
    fens = list(dict.fromkeys(fens))
    rng.shuffle(fens)
    return fens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=400_000)
    ap.add_argument("--out", default="../data/shards")
    ap.add_argument("--sf", default="../tools/stockfish/stockfish-ubuntu-x86-64-avx2")
    ap.add_argument("--puzzles", default="../data/lichess_puzzles.csv")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--multipv", type=int, default=4)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--round-size", type=int, default=8000)
    ap.add_argument("--max-per-shard", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    start_count = count_records(args.out)
    print(f"[generate] existing records: {start_count:,}  target: {args.target:,}")

    t0 = time.time()
    done = start_count
    while done < args.target:
        rsize = min(args.round_size, max(1000, args.target - done))
        fens = make_round_fens(rsize, args.puzzles, rng)
        shard = current_shard(args.out, args.max_per_shard)
        t = time.time()
        n = datagen.label_fens(
            fens, args.sf, shard,
            depth=args.depth, multipv=args.multipv, n_workers=args.workers,
            chunk=150,
        )
        done += n
        dt = time.time() - t
        rate = n / dt if dt else 0
        elapsed = time.time() - t0
        eta = (args.target - done) / rate / 60 if rate else 0
        print(f"[generate] +{n} -> {done:,}/{args.target:,} "
              f"({rate:.0f}/s, round {dt:.0f}s, elapsed {elapsed/60:.1f}m, eta {eta:.0f}m)",
              flush=True)

    print(f"[generate] DONE: {done:,} records in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
