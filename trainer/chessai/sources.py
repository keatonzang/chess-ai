"""Diverse position sources: puzzles, openings (incl. weird ones), endgames,
random midgames, and rare-but-legal positions.

Every generator yields FEN strings. The labeler (datagen.py) turns FENs into
training targets with Stockfish.
"""

from __future__ import annotations

import csv
import os
import random
from typing import Iterator

import chess


# A deliberately offbeat set of opening lines so the net sees weird territory.
WEIRD_OPENINGS = [
    ["e2e4", "e7e5", "g1e2"],            # Napoleon-ish
    ["g2g4"],                             # Grob
    ["b2b4"],                             # Sokolsky / Orangutan
    ["a2a3"],                             # Anderssen
    ["h2h4"],                             # Kadas
    ["e2e4", "e7e5", "d1h5"],            # Wayward Queen
    ["e2e4", "e7e5", "f1c4", "f8c5", "d1h5"],  # early queen + bishop
    ["f2f3", "e7e5", "g2g4"],            # Fool's Mate trap line
    ["e2e3", "e7e5", "d1h5"],
    ["d2d4", "g8f6", "g2g4"],            # weird g4 vs Indian
    ["b1c3", "d7d5", "e2e4"],            # Dunst
    ["e2e4", "g8f6"],                    # Alekhine invite
    ["c2c4", "b7b6"],                    # English vs odd
    ["g1f3", "d7d5", "e2e3", "c8g4", "h2h3", "g4f3"],
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "b7b5"],
    ["e2e4", "c7c5", "b2b4"],            # Wing Gambit
    ["d2d4", "f7f5", "g2g4"],            # vs Dutch, gambit
    ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"],  # Scandinavian
]


def gen_openings(n: int, max_plies: int = 16, weird_frac: float = 0.45,
                 rng: random.Random | None = None) -> Iterator[str]:
    """Opening / early-middlegame positions via random playouts, biased toward
    offbeat first moves a fraction of the time."""
    rng = rng or random.Random()
    for _ in range(n):
        board = chess.Board()
        # sometimes seed with a known weird line
        if rng.random() < weird_frac:
            line = rng.choice(WEIRD_OPENINGS)
            for uci in line:
                mv = chess.Move.from_uci(uci)
                if mv in board.legal_moves:
                    board.push(mv)
                else:
                    break
        target_plies = rng.randint(2, max_plies)
        while not board.is_game_over() and board.fullmove_number * 2 < target_plies + 2:
            if board.ply() >= target_plies:
                break
            board.push(rng.choice(list(board.legal_moves)))
        if not board.is_game_over():
            yield board.fen()


def gen_random_midgame(n: int, min_plies: int = 10, max_plies: int = 80,
                       rng: random.Random | None = None) -> Iterator[str]:
    """Positions from random self-play of varying depth (broad coverage)."""
    rng = rng or random.Random()
    for _ in range(n):
        board = chess.Board()
        depth = rng.randint(min_plies, max_plies)
        ok = True
        for _ in range(depth):
            if board.is_game_over():
                ok = False
                break
            board.push(rng.choice(list(board.legal_moves)))
        if ok and not board.is_game_over():
            yield board.fen()


_ENDGAME_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]


def gen_endgames(n: int, max_extra: int = 5, rng: random.Random | None = None,
                 max_tries: int = 60) -> Iterator[str]:
    """Random, legal, low-material endgame positions."""
    rng = rng or random.Random()
    produced = 0
    attempts = 0
    while produced < n and attempts < n * max_tries:
        attempts += 1
        board = chess.Board.empty()
        squares = rng.sample(chess.SQUARES, 64)
        si = 0
        wk, bk = squares[si], squares[si + 1]
        si += 2
        # kings not adjacent
        if chess.square_distance(wk, bk) <= 1:
            continue
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        n_extra = rng.randint(1, max_extra)
        for _ in range(n_extra):
            if si >= len(squares):
                break
            sq = squares[si]
            si += 1
            pt = rng.choice(_ENDGAME_PIECES)
            # no pawns on first/last rank
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                continue
            color = rng.choice([chess.WHITE, chess.BLACK])
            board.set_piece_at(sq, chess.Piece(pt, color))
        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        board.clear_stack()
        if board.is_valid() and not board.is_game_over():
            produced += 1
            yield board.fen()


# forced-win material for one side (vs a lone king) — teaches checkmate technique
_WIN_SETS = [
    [chess.QUEEN], [chess.ROOK, chess.ROOK], [chess.QUEEN, chess.ROOK],
    [chess.ROOK], [chess.QUEEN, chess.QUEEN],
    [chess.BISHOP, chess.BISHOP, chess.KNIGHT],
]


def gen_winning_endgames(n: int, rng: random.Random | None = None,
                         max_tries: int = 80) -> Iterator[str]:
    """Random legal 'K + winning material vs lone K' positions, so the bot gets
    dense decisive signal and learns to deliver checkmate (pure RL — these are
    just start positions, no solutions provided)."""
    rng = rng or random.Random()
    produced = 0
    attempts = 0
    while produced < n and attempts < n * max_tries:
        attempts += 1
        strong = rng.choice([chess.WHITE, chess.BLACK])
        board = chess.Board.empty()
        squares = rng.sample(chess.SQUARES, 64)
        sk, wk = squares[0], squares[1]
        if chess.square_distance(sk, wk) <= 1:
            continue
        board.set_piece_at(sk, chess.Piece(chess.KING, strong))
        board.set_piece_at(wk, chess.Piece(chess.KING, not strong))
        si = 2
        for pt in rng.choice(_WIN_SETS):
            if si >= len(squares):
                break
            sq = squares[si]; si += 1
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                continue
            board.set_piece_at(sq, chess.Piece(pt, strong))
        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        board.clear_stack()
        if board.is_valid() and not board.is_game_over():
            produced += 1
            yield board.fen()


def gen_puzzles(path: str, n: int, rng: random.Random | None = None) -> Iterator[str]:
    """Sample puzzle positions from the Lichess puzzle CSV.

    The CSV's FEN is the position *before* the setup move; we push the first
    move in ``Moves`` to land on the actual puzzle position to be solved.
    Uses reservoir-free probabilistic sampling so we don't load 5M rows.
    """
    rng = rng or random.Random()
    if not os.path.exists(path):
        return
    # estimate total lines cheaply for sampling probability
    approx_total = 4_000_000
    prob = min(1.0, (n * 3.0) / approx_total)
    yielded = 0
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if yielded >= n:
                break
            if rng.random() > prob:
                continue
            if len(row) < 3:
                continue
            fen, moves = row[1], row[2].split()
            try:
                board = chess.Board(fen)
                if moves:
                    setup = chess.Move.from_uci(moves[0])
                    if setup in board.legal_moves:
                        board.push(setup)
                if not board.is_game_over():
                    yielded += 1
                    yield board.fen()
            except Exception:
                continue
