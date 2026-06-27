"""Bot-vs-Stockfish actor with surprise-shaped rewards AND a material-handicap
curriculum.

Two ideas combined:
  * Surprise reward: for each bot move, Stockfish evaluates the resulting
    position; the value target is boosted when the eventual outcome beats
    Stockfish's expectation and dampened when it underperforms:
        sf_expectation = 2 * winprob(stockfish_cp_for_bot) - 1   # [-1, 1]
        value_target   = clip(outcome + lam*(outcome - sf_expectation), -1, 1)
  * Curriculum: games start from a position where Stockfish is DOWN material, so
    even a weak bot gets winnable games. As the bot scores well at a level, the
    handicap automatically shrinks toward an even game ("harder and harder").

Samples (bot decision positions only) go to the shared replay buffer.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time

import chess
import chess.engine
import torch

from chessai.model import build_model
from chessai.rl import SelfPlay, cp_to_winprob

# Curriculum: pieces removed from Stockfish's side, easiest (most handicap) first.
HANDICAP_LEVELS = [
    ["q", "r", "b", "n"],  # ~ -20: bot should crush
    ["q", "r"],            # ~ -14
    ["q"],                 # -9
    ["r"],                 # -5
    ["b", "n"],            # -6 (two minors, different shape)
    ["n"],                 # -3
    ["p"],                 # -1
    [],                    # even
]

# squares to clear for a given color (one per piece type)
_SQ = {
    chess.WHITE: {"q": chess.D1, "r": chess.A1, "b": chess.C1, "n": chess.B1,
                  "p": chess.D2},
    chess.BLACK: {"q": chess.D8, "r": chess.A8, "b": chess.C8, "n": chess.B8,
                  "p": chess.D7},
}


def make_start_board(sf_color, level):
    """Standard start position minus HANDICAP_LEVELS[level] from sf_color's side."""
    board = chess.Board()
    for p in HANDICAP_LEVELS[level]:
        board.remove_piece_at(_SQ[sf_color][p])
    # removing a rook would leave a dangling castling right -> fix it
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
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--sf-skill", type=int, default=0)
    ap.add_argument("--sf-movetime", type=float, default=0.05)
    ap.add_argument("--eval-depth", type=int, default=8)
    ap.add_argument("--games-per-shard", type=int, default=6)
    # curriculum
    ap.add_argument("--start-level", type=int, default=0)
    ap.add_argument("--adapt-window", type=int, default=10)
    ap.add_argument("--up-thresh", type=float, default=0.6, help="advance (harder)")
    ap.add_argument("--down-thresh", type=float, default=0.25, help="regress (easier)")
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

    level = max(0, min(args.start_level, len(HANDICAP_LEVELS) - 1))
    recent = collections.deque(maxlen=args.adapt_window)
    print(f"[vs_sf {args.id}] started on {device} vs SF skill {args.sf_skill}, "
          f"level {level} (SF down {HANDICAP_LEVELS[level]})", flush=True)

    shard_games = []
    shard_idx = 0

    def play_game(level):
        sp = SelfPlay(model, device=device, sims=args.sims,
                      max_moves=args.max_moves, temp_moves=args.temp_moves)
        bot_color = chess.WHITE if (args.id + shard_idx) % 2 == 0 else chess.BLACK
        sf_color = not bot_color
        board = make_start_board(sf_color, level)
        records = []
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
            shaped = max(-1.0, min(1.0, outcome + args.lam * (outcome - sf_exp)))
            samples.append({
                "fen": fen,
                "value": round(shaped, 5),
                "policy": [[int(i), round(float(p), 5)] for i, p in policy_idx],
            })
        bot_score = (outcome + 1.0) / 2.0  # 1 win / .5 draw / 0 loss
        return samples, res, bot_color, bot_score

    while True:
        m = os.path.getmtime(args.ckpt) if os.path.exists(args.ckpt) else None
        if m is not None and m != ck_mtime:
            model = load_model(args.ckpt, device, args.channels, args.blocks)
            ck_mtime = m

        samples, res, bot_color, bot_score = play_game(level)
        shard_games.append(samples)
        recent.append(bot_score)

        # adapt curriculum
        if len(recent) >= args.adapt_window:
            avg = sum(recent) / len(recent)
            if avg >= args.up_thresh and level < len(HANDICAP_LEVELS) - 1:
                level += 1
                recent.clear()
                print(f"[vs_sf {args.id}] ADVANCE -> level {level} "
                      f"(SF down {HANDICAP_LEVELS[level]}) avg={avg:.2f}", flush=True)
            elif avg <= args.down_thresh and level > 0:
                level -= 1
                recent.clear()
                print(f"[vs_sf {args.id}] REGRESS -> level {level} "
                      f"(SF down {HANDICAP_LEVELS[level]}) avg={avg:.2f}", flush=True)

        if len(shard_games) >= args.games_per_shard:
            flat = [s for g in shard_games for s in g]
            ts = int(time.time() * 1000)
            final = os.path.join(args.buffer, f"vssf_{args.id}_{ts}.jsonl")
            tmp = os.path.join(args.buffer, f".tmpvssf_{args.id}_{ts}.jsonl")
            with open(tmp, "w") as f:
                for s in flat:
                    f.write(json.dumps(s) + "\n")
            os.rename(tmp, final)
            score = sum(recent) / len(recent) if recent else 0.0
            print(f"[vs_sf {args.id}] wrote {len(flat)} samples, "
                  f"level {level} (SF down {HANDICAP_LEVELS[level] or 'nothing'}), "
                  f"recent_score={score:.2f}", flush=True)
            shard_games = []
            shard_idx += 1


if __name__ == "__main__":
    main()
