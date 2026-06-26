"""Estimate playing strength by matches vs Stockfish at limited skill/depth.

Plays N games as both colors against Stockfish configured to a target level,
reports score, and gives a rough Elo estimate from the score rate.
"""

from __future__ import annotations

import argparse
import math

import chess
import chess.engine
import torch

from chessai.model import build_model
from chessai.mcts import best_move


def score_to_elo_diff(score_rate: float) -> float:
    score_rate = min(max(score_rate, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / score_rate - 1.0)


def play_game(model, engine, device, sims, sf_limit, model_white):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        if board.turn == chess.WHITE and model_white or \
           board.turn == chess.BLACK and not model_white:
            mv, _ = best_move(model, board, n_sims=sims, device=device,
                              temperature=0.0)
            if mv is None:
                break
        else:
            result = engine.play(board, sf_limit)
            mv = result.move
        board.push(mv)
    res = board.result(claim_draw=True)
    if res == "1-0":
        return 1.0 if model_white else 0.0
    if res == "0-1":
        return 0.0 if model_white else 1.0
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../models/base_best.pt")
    ap.add_argument("--sf", default="../tools/stockfish/stockfish-ubuntu-x86-64-avx2")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--sf-skill", type=int, default=3)
    ap.add_argument("--sf-movetime", type=float, default=0.05)
    ap.add_argument("--sf-elo", type=int, default=None,
                    help="use UCI_LimitStrength at this Elo instead of skill")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_model(ckpt.get("channels", 128), ckpt.get("blocks", 10)).to(device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    engine = chess.engine.SimpleEngine.popen_uci(args.sf)
    if args.sf_elo is not None:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.sf_elo})
        opp = f"SF Elo {args.sf_elo}"
    else:
        engine.configure({"Skill Level": args.sf_skill})
        opp = f"SF skill {args.sf_skill}"
    limit = chess.engine.Limit(time=args.sf_movetime)

    total = 0.0
    for g in range(args.games):
        model_white = g % 2 == 0
        s = play_game(model, engine, device, args.sims, limit, model_white)
        total += s
        print(f"  game {g+1}/{args.games} "
              f"({'W' if model_white else 'B'}): {s}  running={total}", flush=True)
    engine.quit()

    rate = total / args.games
    diff = score_to_elo_diff(rate)
    print(f"\n[eval] vs {opp}, {args.sims} sims: "
          f"score {total}/{args.games} = {rate:.1%}")
    print(f"[eval] Elo difference vs opponent: {diff:+.0f}")


if __name__ == "__main__":
    main()
