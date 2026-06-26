"""Label positions with Stockfish to produce (policy, value) training targets.

Runs a pool of Stockfish workers in parallel. For each FEN we ask for the top-K
moves (multipv) and turn their evaluations into:
  * a soft policy target: softmax over centipawn evals (favours the best move,
    spreads weight to near-best moves)
  * a value target in [-1, 1] from the side-to-move's perspective.

Output: JSONL shards, one record per line:
  {"fen": ..., "value": float, "policy": [[move_index, prob], ...]}
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
from typing import Iterable

import chess
import chess.engine

from .encoding import move_to_index


def cp_to_winprob(cp: float) -> float:
    """Lichess-style centipawn -> win probability in [0, 1]."""
    return 1.0 / (1.0 + math.exp(-0.00368208 * cp))


def score_to_cp(score: chess.engine.PovScore, pov: bool) -> float:
    """Relative score (from `pov`) to a clamped centipawn number."""
    s = score.pov(pov)
    if s.is_mate():
        m = s.mate()
        # large but finite; closer mates -> more extreme
        return 100000.0 / m if m != 0 else 100000.0
    return float(s.score(mate_score=100000))


def _label_one(engine: chess.engine.SimpleEngine, fen: str, depth: int,
               multipv: int, policy_temp: float):
    board = chess.Board(fen)
    pov = board.turn
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    if isinstance(infos, dict):
        infos = [infos]

    entries = []
    for info in infos:
        pv = info.get("pv")
        if not pv:
            continue
        mv = pv[0]
        cp = score_to_cp(info["score"], pov)
        entries.append((mv, cp))
    if not entries:
        return None

    # value from the best line, in side-to-move perspective
    best_cp = entries[0][1]
    value = 2.0 * cp_to_winprob(best_cp) - 1.0

    # soft policy via softmax over cp (relative to best, temperature in cp)
    max_cp = max(cp for _, cp in entries)
    weights = [math.exp((cp - max_cp) / policy_temp) for _, cp in entries]
    z = sum(weights)
    policy = []
    for (mv, _), w in zip(entries, weights):
        idx = move_to_index(mv, board)
        policy.append([idx, w / z])

    return {"fen": fen, "value": round(value, 5),
            "policy": [[i, round(p, 5)] for i, p in policy]}


def _worker(args):
    fens, sf_path, depth, multipv, policy_temp, threads, hash_mb = args
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    try:
        engine.configure({"Threads": threads, "Hash": hash_mb})
    except Exception:
        pass
    out = []
    for fen in fens:
        try:
            rec = _label_one(engine, fen, depth, multipv, policy_temp)
            if rec is not None:
                out.append(rec)
        except Exception:
            continue
    engine.quit()
    return out


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def label_fens(fens: list[str], sf_path: str, out_path: str,
               depth: int = 12, multipv: int = 4, policy_temp: float = 90.0,
               n_workers: int = 14, threads: int = 1, hash_mb: int = 64,
               chunk: int = 200) -> int:
    """Label a list of FENs and append JSONL records to ``out_path``.

    Returns the number of records written.
    """
    batches = list(_chunks(fens, chunk))
    tasks = [(b, sf_path, depth, multipv, policy_temp, threads, hash_mb)
             for b in batches]

    written = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "a") as fout, mp.Pool(n_workers) as pool:
        for recs in pool.imap_unordered(_worker, tasks):
            for r in recs:
                fout.write(json.dumps(r) + "\n")
                written += 1
            fout.flush()
    return written
