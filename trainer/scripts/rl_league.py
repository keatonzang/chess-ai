"""League actor: the current network plays past snapshots of itself.

Prevents the bot from over-fitting to its own latest quirks (an opponent that
only ever sees "current vs current" can chase its tail). Samples are recorded
only for the CURRENT net's moves (MCTS policy + game outcome) — pure RL, no
external targets. Opponent snapshots are sampled with a recency bias.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time

import chess
import torch

from chessai.model import build_model
from chessai.rl import SelfPlay, outcome_white
from chessai import sources


def load_into(model, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--ckpt", default="../models/rl_current.pt")
    ap.add_argument("--snapdir", default="../models")
    ap.add_argument("--buffer", default="../data/rl_buffer")
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--max-moves", type=int, default=140)
    ap.add_argument("--temp-moves", type=int, default=12)
    ap.add_argument("--seed-frac", type=float, default=0.3)
    ap.add_argument("--puzzles", default="../data/puzzle_fens.txt")
    args = ap.parse_args()

    os.makedirs(args.buffer, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    rng = random.Random(3000 + args.id)

    cur = build_model(args.channels, args.blocks).to(device).eval()
    opp = build_model(args.channels, args.blocks).to(device).eval()
    if os.path.exists(args.ckpt):
        load_into(cur, args.ckpt, device)
    ck_mtime = os.path.getmtime(args.ckpt) if os.path.exists(args.ckpt) else None

    pool = []
    if args.seed_frac > 0:
        pool += list(sources.gen_openings(2000, weird_frac=0.5, rng=rng))
        pool += list(sources.gen_endgames(1500, rng=rng))
        if os.path.exists(args.puzzles):
            with open(args.puzzles) as f:
                pool += [l.strip() for l in f if l.strip()][:2000]
        rng.shuffle(pool)

    print(f"[league {args.id}] started on {device}", flush=True)
    sp_cur = SelfPlay(cur, device=device, sims=args.sims,
                      max_moves=args.max_moves, temp_moves=args.temp_moves)
    sp_opp = SelfPlay(opp, device=device, sims=args.sims,
                      max_moves=args.max_moves, temp_moves=0)

    def pick_opponent():
        snaps = sorted(glob.glob(os.path.join(args.snapdir, "rl_iter_*.pt")))
        if not snaps:
            return None
        # recency-biased: weight later snapshots more
        weights = [i + 1 for i in range(len(snaps))]
        return random.choices(snaps, weights=weights, k=1)[0]

    def play_game():
        snap = pick_opponent()
        if snap is not None:
            try:
                load_into(opp, snap, device)
            except Exception:
                pass
        cur_color = chess.WHITE if rng.random() < 0.5 else chess.BLACK
        start = (rng.choice(pool) if (pool and rng.random() < args.seed_frac)
                 else None)
        board = chess.Board(start) if start else chess.Board()
        records = []
        move_no = 0
        while not board.is_game_over(claim_draw=True) and move_no < args.max_moves:
            if board.turn == cur_color:
                temp = 1.0 if move_no < args.temp_moves else 0.0
                fen_before = board.fen()
                mv, policy_idx = sp_cur.search_one(board, add_noise=True,
                                                   temperature=temp)
                if mv is None:
                    break
                board.push(mv)
                records.append((fen_before, policy_idx))
            else:
                mv, _ = sp_opp.search_one(board, add_noise=False, temperature=0.0)
                if mv is None:
                    break
                board.push(mv)
            move_no += 1

        ov = outcome_white(board)  # material-adjudicated if unfinished
        outcome = ov if cur_color == chess.WHITE else -ov
        samples = [{
            "fen": fen,
            "value": round(outcome, 5),
            "policy": [[int(i), round(float(p), 5)] for i, p in policy_idx],
        } for fen, policy_idx in records]
        return samples, (outcome + 1) / 2, os.path.basename(snap) if snap else "self"

    shard = []
    while True:
        m = os.path.getmtime(args.ckpt) if os.path.exists(args.ckpt) else None
        if m is not None and m != ck_mtime:
            load_into(cur, args.ckpt, device)
            ck_mtime = m

        samples, score, opp_name = play_game()
        shard.extend(samples)
        if len(shard) >= 600:
            ts = int(time.time() * 1000)
            final = os.path.join(args.buffer, f"league_{args.id}_{ts}.jsonl")
            tmp = os.path.join(args.buffer, f".tmpleague_{args.id}_{ts}.jsonl")
            with open(tmp, "w") as f:
                for s in shard:
                    f.write(json.dumps(s) + "\n")
            os.rename(tmp, final)
            print(f"[league {args.id}] wrote {len(shard)} samples "
                  f"(last vs {opp_name}, score {score:.1f})", flush=True)
            shard = []


if __name__ == "__main__":
    main()
