"""Bot-vs-Stockfish actor: surprise reward + material-handicap curriculum +
multi-level opponents + novelty bonus.

For each bot move we ask Stockfish (multi-PV) about the position the bot faces:
  * sf_expectation = 2*winprob(best-line cp for bot) - 1            -> surprise
  * novelty: the bot chose a move that ISN'T Stockfish's top pick but IS among
    its top-K (i.e. a sound deviation) -> small extra reward for creativity

  value_target = clip(outcome
                      + lam*(outcome - sf_expectation)       # surprise
                      + nov_bonus * novel,                   # sound novelty
                      -1, 1)

Curriculum: games start with Stockfish DOWN material; handicap auto-scales with
the bot's recent score. Opponent strength varies across a Skill-Level set.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import time

import chess
import chess.engine
import torch

from chessai.model import build_model
from chessai.rl import SelfPlay, cp_to_winprob, outcome_white

HANDICAP_LEVELS = [
    ["q", "r", "b", "n"], ["q", "r"], ["q"], ["r"],
    ["b", "n"], ["n"], ["p"], [],
]
_SQ = {
    chess.WHITE: {"q": chess.D1, "r": chess.A1, "b": chess.C1, "n": chess.B1, "p": chess.D2},
    chess.BLACK: {"q": chess.D8, "r": chess.A8, "b": chess.C8, "n": chess.B8, "p": chess.D7},
}


def make_start_board(sf_color, level):
    board = chess.Board()
    for p in HANDICAP_LEVELS[level]:
        board.remove_piece_at(_SQ[sf_color][p])
    board.castling_rights = board.clean_castling_rights()
    board.clear_stack()
    return board


def load_model(ckpt_path, device, channels, blocks):
    model = build_model(channels, blocks).to(device).eval()
    if os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ck["model"] if "model" in ck else ck)
        except Exception as e:
            print(f"[vs_sf] load failed: {e}", flush=True)
    return model


def cp_from_score(score):
    if score.is_mate():
        return 100000.0 / score.mate() if score.mate() else 100000.0
    return float(score.score(mate_score=100000))


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
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--nov-bonus", type=float, default=0.0)
    ap.add_argument("--novelty-k", type=int, default=3)
    ap.add_argument("--sf-skill-set", default="0,1,3,5")
    ap.add_argument("--sf-movetime", type=float, default=0.05)
    ap.add_argument("--eval-depth", type=int, default=8)
    ap.add_argument("--games-per-shard", type=int, default=6)
    ap.add_argument("--start-level", type=int, default=0)
    ap.add_argument("--adapt-window", type=int, default=10)
    ap.add_argument("--up-thresh", type=float, default=0.6)
    ap.add_argument("--down-thresh", type=float, default=0.25)
    args = ap.parse_args()

    os.makedirs(args.buffer, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    rng = random.Random(2000 + args.id)
    skill_set = [int(s) for s in args.sf_skill_set.split(",")]

    model = load_model(args.ckpt, device, args.channels, args.blocks)
    ck_mtime = os.path.getmtime(args.ckpt) if os.path.exists(args.ckpt) else None

    engine = chess.engine.SimpleEngine.popen_uci(args.sf)
    play_limit = chess.engine.Limit(time=args.sf_movetime)
    eval_limit = chess.engine.Limit(depth=args.eval_depth)

    level = max(0, min(args.start_level, len(HANDICAP_LEVELS) - 1))
    recent = collections.deque(maxlen=args.adapt_window)
    print(f"[vs_sf {args.id}] started on {device}, skills {skill_set}, "
          f"level {level} ({HANDICAP_LEVELS[level]})", flush=True)

    shard_games = []
    shard_idx = 0

    def play_game(level):
        sp = SelfPlay(model, device=device, sims=args.sims,
                      max_moves=args.max_moves, temp_moves=args.temp_moves)
        skill = rng.choice(skill_set)
        try:
            engine.configure({"Skill Level": skill})
        except Exception:
            pass
        bot_color = chess.WHITE if (args.id + shard_idx) % 2 == 0 else chess.BLACK
        sf_color = not bot_color
        board = make_start_board(sf_color, level)
        records = []
        novel_count = 0
        move_no = 0
        while not board.is_game_over(claim_draw=True) and move_no < args.max_moves:
            if board.turn == bot_color:
                fen_before = board.fen()
                sf_best, topk, sf_exp = None, set(), 0.0
                try:
                    mpv = args.novelty_k if args.nov_bonus > 0 else 1
                    infos = engine.analyse(board, eval_limit, multipv=mpv)
                    if isinstance(infos, dict):
                        infos = [infos]
                    sc = infos[0]["score"].pov(bot_color)
                    sf_exp = 2.0 * cp_to_winprob(cp_from_score(sc)) - 1.0
                    for info in infos:
                        pv = info.get("pv")
                        if pv:
                            topk.add(pv[0])
                    if infos[0].get("pv"):
                        sf_best = infos[0]["pv"][0]
                except Exception:
                    pass
                temp = 1.0 if move_no < args.temp_moves else 0.0
                mv, policy_idx = sp.search_one(board, add_noise=True, temperature=temp)
                if mv is None:
                    break
                novel = sf_best is not None and mv != sf_best and mv in topk
                if novel:
                    novel_count += 1
                board.push(mv)
                records.append((fen_before, policy_idx, sf_exp, novel))
            else:
                result = engine.play(board, play_limit)
                if result.move is None:
                    break
                board.push(result.move)
            move_no += 1

        res = board.result(claim_draw=True)
        ov = outcome_white(board)  # material-adjudicated if unfinished
        outcome = ov if bot_color == chess.WHITE else -ov

        samples = []
        for fen, policy_idx, sf_exp, novel in records:
            shaped = outcome + args.lam * (outcome - sf_exp)
            if novel:
                shaped += args.nov_bonus
            shaped = max(-1.0, min(1.0, shaped))
            samples.append({
                "fen": fen,
                "value": round(shaped, 5),
                "policy": [[int(i), round(float(p), 5)] for i, p in policy_idx],
            })
        bot_score = (outcome + 1.0) / 2.0
        return samples, res, bot_score, novel_count

    while True:
        m = os.path.getmtime(args.ckpt) if os.path.exists(args.ckpt) else None
        if m is not None and m != ck_mtime:
            model = load_model(args.ckpt, device, args.channels, args.blocks)
            ck_mtime = m

        samples, res, bot_score, novel_count = play_game(level)
        shard_games.append((samples, novel_count))
        recent.append(bot_score)

        if len(recent) >= args.adapt_window:
            avg = sum(recent) / len(recent)
            if avg >= args.up_thresh and level < len(HANDICAP_LEVELS) - 1:
                level += 1
                recent.clear()
                print(f"[vs_sf {args.id}] ADVANCE -> level {level} "
                      f"({HANDICAP_LEVELS[level]}) avg={avg:.2f}", flush=True)
            elif avg <= args.down_thresh and level > 0:
                level -= 1
                recent.clear()
                print(f"[vs_sf {args.id}] REGRESS -> level {level} "
                      f"({HANDICAP_LEVELS[level]}) avg={avg:.2f}", flush=True)

        if len(shard_games) >= args.games_per_shard:
            flat = [s for g, _ in shard_games for s in g]
            nov = sum(n for _, n in shard_games)
            ts = int(time.time() * 1000)
            final = os.path.join(args.buffer, f"vssf_{args.id}_{ts}.jsonl")
            tmp = os.path.join(args.buffer, f".tmpvssf_{args.id}_{ts}.jsonl")
            with open(tmp, "w") as f:
                for s in flat:
                    f.write(json.dumps(s) + "\n")
            os.rename(tmp, final)
            score = sum(recent) / len(recent) if recent else 0.0
            print(f"[vs_sf {args.id}] wrote {len(flat)} samples, level {level} "
                  f"(SF down {HANDICAP_LEVELS[level] or 'nothing'}), "
                  f"recent_score={score:.2f}, novel_moves={nov}", flush=True)
            shard_games = []
            shard_idx += 1


if __name__ == "__main__":
    main()
