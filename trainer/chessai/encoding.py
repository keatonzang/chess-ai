"""Board <-> tensor encoding, and move <-> policy-index mapping.

Conventions
-----------
* The board is always encoded from the perspective of the side to move: if it is
  Black's turn we mirror the board vertically and swap colors, so the network
  always "sees" itself as White moving up the board. Moves are mirrored to match.
* Policy space uses the AlphaZero 8x8x73 = 4672 encoding:
    - 56 "queen-like" planes: 8 compass directions x 7 distances
    -  8 knight planes
    -  9 underpromotion planes: 3 move directions x {knight, bishop, rook}
  Queen promotions are encoded as ordinary forward/diagonal pawn pushes and
  decoded back to a queen promotion automatically.

The plane index for a from-square is `from_square * 73 + plane`, so the flat
policy vector has length 64 * 73 = 4672.
"""

from __future__ import annotations

import chess
import numpy as np

# ----------------------------------------------------------------------------
# Board planes
# ----------------------------------------------------------------------------
# 12 piece planes (own P,N,B,R,Q,K then opponent P,N,B,R,Q,K)
# + 4 castling rights (own short, own long, opp short, opp long)
# + 1 en-passant file marker
# + 1 fifty-move-clock (scaled)
N_PLANES = 18

_PIECE_ORDER = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def board_to_planes(board: chess.Board) -> np.ndarray:
    """Return an (N_PLANES, 8, 8) float32 tensor from side-to-move perspective."""
    planes = np.zeros((N_PLANES, 8, 8), dtype=np.float32)
    stm = board.turn  # side to move
    mirror = stm == chess.BLACK

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        idx_sq = chess.square_mirror(sq) if mirror else sq
        rank = chess.square_rank(idx_sq)
        file = chess.square_file(idx_sq)
        # own pieces in planes 0-5, opponent in 6-11
        own = piece.color == stm
        base = 0 if own else 6
        plane = base + _PIECE_ORDER.index(piece.piece_type)
        planes[plane, rank, file] = 1.0

    # castling rights, from side-to-move perspective
    own_k = board.has_kingside_castling_rights(stm)
    own_q = board.has_queenside_castling_rights(stm)
    opp_k = board.has_kingside_castling_rights(not stm)
    opp_q = board.has_queenside_castling_rights(not stm)
    if own_k:
        planes[12, :, :] = 1.0
    if own_q:
        planes[13, :, :] = 1.0
    if opp_k:
        planes[14, :, :] = 1.0
    if opp_q:
        planes[15, :, :] = 1.0

    # en passant target file
    if board.ep_square is not None:
        ep = chess.square_mirror(board.ep_square) if mirror else board.ep_square
        planes[16, :, chess.square_file(ep)] = 1.0

    # fifty-move clock, scaled to ~[0,1]
    planes[17, :, :] = min(board.halfmove_clock, 100) / 100.0

    return planes


# ----------------------------------------------------------------------------
# Move <-> policy index
# ----------------------------------------------------------------------------
N_MOVES = 64 * 73  # 4672

# 8 compass directions as (file_delta, rank_delta)
_DIRECTIONS = [
    (0, 1),    # N
    (1, 1),    # NE
    (1, 0),    # E
    (1, -1),   # SE
    (0, -1),   # S
    (-1, -1),  # SW
    (-1, 0),   # W
    (-1, 1),   # NW
]

_KNIGHT_DELTAS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]

# underpromotion piece order
_UNDERPROMO = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _mirror_square(sq: int) -> int:
    return chess.square_mirror(sq)


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Encode a legal move (in board's own frame) to a policy index.

    Mirrors the move when it is Black to move so it lives in the canonical
    side-to-move frame used by ``board_to_planes``.
    """
    mirror = board.turn == chess.BLACK
    from_sq = _mirror_square(move.from_square) if mirror else move.from_square
    to_sq = _mirror_square(move.to_square) if mirror else move.to_square

    ff, fr = chess.square_file(from_sq), chess.square_rank(from_sq)
    tf, tr = chess.square_file(to_sq), chess.square_rank(to_sq)
    df, dr = tf - ff, tr - fr

    plane = _delta_to_plane(df, dr, move.promotion)
    return from_sq * 73 + plane


def _delta_to_plane(df: int, dr: int, promotion) -> int:
    # underpromotions (knight/bishop/rook): always a pawn pushing "up" one rank
    if promotion is not None and promotion != chess.QUEEN:
        piece_idx = _UNDERPROMO.index(promotion)
        # df in {-1, 0, 1} -> capture-left, push, capture-right
        return 64 + piece_idx * 3 + (df + 1)

    # knight moves
    if (df, dr) in _KNIGHT_DELTAS:
        return 56 + _KNIGHT_DELTAS.index((df, dr))

    # queen-like moves (includes queen promotions, encoded as the push)
    dist = max(abs(df), abs(dr))
    step = (0 if df == 0 else df // abs(df), 0 if dr == 0 else dr // abs(dr))
    dir_idx = _DIRECTIONS.index(step)
    return dir_idx * 7 + (dist - 1)


def index_to_move(index: int, board: chess.Board) -> chess.Move | None:
    """Decode a policy index back into a concrete move in ``board``'s frame.

    Returns None if the decoded move is illegal in this position.
    """
    mirror = board.turn == chess.BLACK
    from_sq = index // 73
    plane = index % 73

    ff, fr = chess.square_file(from_sq), chess.square_rank(from_sq)

    promotion = None
    if plane < 56:
        dir_idx = plane // 7
        dist = (plane % 7) + 1
        df = _DIRECTIONS[dir_idx][0] * dist
        dr = _DIRECTIONS[dir_idx][1] * dist
    elif plane < 64:
        df, dr = _KNIGHT_DELTAS[plane - 56]
    else:
        u = plane - 64
        promotion = _UNDERPROMO[u // 3]
        df = (u % 3) - 1
        dr = 1  # underpromotion always advances one rank in canonical frame

    tf, tr = ff + df, fr + dr
    if not (0 <= tf < 8 and 0 <= tr < 8):
        return None
    to_sq = chess.square(tf, tr)

    # un-mirror back to the board's real frame
    if mirror:
        from_sq_real = _mirror_square(from_sq)
        to_sq_real = _mirror_square(to_sq)
    else:
        from_sq_real, to_sq_real = from_sq, to_sq

    # auto queen-promotion: pawn reaching last rank via a queen-like plane
    if promotion is None and plane < 56:
        piece = board.piece_at(from_sq_real)
        if piece is not None and piece.piece_type == chess.PAWN:
            last_rank = 7 if board.turn == chess.WHITE else 0
            if chess.square_rank(to_sq_real) == last_rank:
                promotion = chess.QUEEN

    move = chess.Move(from_sq_real, to_sq_real, promotion=promotion)
    if move in board.legal_moves:
        return move
    return None


def legal_move_indices(board: chess.Board) -> dict[int, chess.Move]:
    """Map every legal move in ``board`` to its policy index."""
    out: dict[int, chess.Move] = {}
    for mv in board.legal_moves:
        out[move_to_index(mv, board)] = mv
    return out
