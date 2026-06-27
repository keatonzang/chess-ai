"""Bot-vs-Stockfish actor with "surprise" reward shaping.

The bot (MCTS + current net) plays games against Stockfish. For each move the
bot makes, Stockfish evaluates the resulting position; if the game's eventual
outcome beats Stockfish's expectation for that position, the value target is
boosted (and dampened if the bot underperformed). This rewards unconventional
moves that Stockfish underrates but that actually work:

    sf_expectation = 2 * winprob(stockfish_cp_for_bot) - 1     # in [-1, 1]
    value_target   = clip(outcome + lam * (outcome - sf_expectation), -1, 1)

Samples (bot decision positions only) are written to the shared replay buffer,
so the learner trains on them mixed with pure self-play.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import chess
import chess.engine
import torch

from chessai.model import build_model
from chessai.rl import SelfPlay, cp_to_winprob


def load_model(ckpt_path, device, channels, blocks):
    model = build_model(channels, blocks).to(device).eval()
    if os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ck["model"] if "model" in ck else ck)
        except Exception as e:
            print(f"[vs_sf] load failed: {e}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--ckpt", default="../models/rl_current.pt")
    ap.add_argument("--buffer", default="../data/rl_buffer")
    ap.add_argument("--sf", default="../tools/stockfish/stockfish-ubuntu-x86-64-avx2")
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--sims", type=int, default=96)
    ap.add_argument("--max-moves", type=int, default=140)
    ap.add_argument("--temp-moves", type=int, default=12)
    ap.add_argument("--lam", type=float, default=0.5, help="surprise bonus weight")
    ap.add_argument("--sf-skill", type=int, default=0)
    ap.add_argument("--sf-movetime", type=float, default=0.05)
    ap.add_argument("--eval-depth", type=int, default=8)
    ap.add_argument("--games-per-shard", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.buffer, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    model = load_model(args.ckpt, device, args.channels, args.blocks)
    ck_mtime = os.path.getmtime(args.ckpt) if os.path.exists(args.ckpt) else None

    engine = chess.engine.SimpleEngine.popen_uci(args.sf)
    try:
        engine.configure({"Skill Level": args.sf_skill})
    except Exception:
        pass
    play_limit = chess.engine.Limit(time=args.sf_movetime)
    eval_limit = chess.engine.Limit(depth=args.eval_depth)

    print(f"[vs_sf {args.id}] started on {device} vs SF skill {args.sf_skill}",
          flush=True)

    shard_games = []
    shard_idx = 0
    total_surprise = 0.0
    n_bonus = 0

    def play_game():
        nonlocal total_surprise, n_bonus
        sp = SelfPlay(model, device=device, sims=args.sims,
                      max_moves=args.max_moves, temp_moves=args.temp_moves)
        board = chess.Board()
        bot_color = chess.WHITE if (args.id + shard_idx) % 2 == 0 else chess.BLACK
        records = []  # (fen_before, policy_idx, sf_expectation)
        move_no = 0
        while not board.is_game_over(claim_draw=True) and move_no < args.max_moves:
            if board.turn == bot_color:
                temp = 1.0 if move_no < args.temp_moves else 0.0
                fen_before = board.fen()
                mv, policy_idx = sp.search_one(board, add_noise=True,
                                               temperature=temp)
                if mv is None:
                    break
                board.push(mv)
                # Stockfish eval of the resulting position, from the bot's POV
                try:
                    info = engine.analyse(board, eval_limit)
                    score = info["score"].pov(bot_color)
                    if score.is_mate():
                        cp = 100000.0 / score.mate() if score.mate() else 100000.0
                    else:
                        cp = float(score.score(mate_score=100000))
                    sf_exp = 2.0 * cp_to_winprob(cp) - 1.0
                except Exception:
                    sf_exp = 0.0
                records.append((fen_before, policy_idx, sf_exp))
            else:
                result = engine.play(board, play_limit)
                if result.move is None:
                    break
                board.push(result.move)
            move_no += 1

        res = board.result(claim_draw=True)
        win = 1.0 if res == "1-0" else -1.0 if res == "0-1" else 0.0
        outcome = win if bot_color == chess.WHITE else -win

        samples = []
        for fen, policy_idx, sf_exp in records:
            surprise = outcome - sf_exp
            shaped = max(-1.0, min(1.0, outcome + args.lam * surprise))
            total_surprise += surprise
            if surprise > 0.15:
                n_bonus += 1
            samples.append({
                "fen": fen,
                "value": round(shaped, 5),
                "policy": [[int(i), round(float(p), 5)] for i, p in policy_idx],
            })
        return samples, res, bot_color

    while True:
        m = os.path.getmtime(args.ckpt) if os.path.exists(args.ckpt) else None
        if m is not None and m != ck_mtime:
            model = load_model(args.ckpt, device, args.channels, args.blocks)
            ck_mtime = m

        t = time.time()
        samples, res, bot_color = play_game()
        shard_games.append(samples)

        if len(shard_games) >= args.games_per_shard:
            flat = [s for g in shard_games for s in g]
            ts = int(time.time() * 1000)
            final = os.path.join(args.buffer, f"vssf_{args.id}_{ts}.jsonl")
            tmp = os.path.join(args.buffer, f".tmpvssf_{args.id}_{ts}.jsonl")
            with open(tmp, "w") as f:
                for s in flat:
                    f.write(json.dumps(s) + "\n")
            os.rename(tmp, final)
            avg_surp = total_surprise / max(1, sum(len(g) for g in shard_games))
            print(f"[vs_sf {args.id}] wrote {len(flat)} samples "
                  f"({len(shard_games)} games) last={res} as "
                  f"{'W' if bot_color==chess.WHITE else 'B'} "
                  f"avg_surprise={avg_surp:+.3f} bonus_moves={n_bonus}", flush=True)
            shard_games = []
            total_surprise = 0.0
            n_bonus = 0
            shard_idx += 1


if __name__ == "__main__":
    main()
