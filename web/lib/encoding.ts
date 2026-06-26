// TypeScript port of trainer/chessai/encoding.py — must match it EXACTLY.
// We only need the forward direction in the browser:
//   * boardToPlanes(fen)  -> Float32Array (18*8*8), side-to-move perspective
//   * moveToIndex(...)    -> policy index for a legal move
// Parity is verified against the Python implementation in scripts/parity_check.

export const N_PLANES = 18;
export const N_MOVES = 64 * 73; // 4672

// piece char -> plane offset within a color's 6 planes
const PIECE_IDX: Record<string, number> = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };

// 8 compass directions (fileDelta, rankDelta) — order matters
const DIRECTIONS: [number, number][] = [
  [0, 1],   // N
  [1, 1],   // NE
  [1, 0],   // E
  [1, -1],  // SE
  [0, -1],  // S
  [-1, -1], // SW
  [-1, 0],  // W
  [-1, 1],  // NW
];

const KNIGHT_DELTAS: [number, number][] = [
  [1, 2], [2, 1], [2, -1], [1, -2],
  [-1, -2], [-2, -1], [-2, 1], [-1, 2],
];

const UNDERPROMO = ["n", "b", "r"]; // knight, bishop, rook

export interface ParsedFen {
  pieces: ({ type: string; color: "w" | "b" } | null)[]; // index 0..63, a1=0
  turn: "w" | "b";
  castling: string;
  epSquare: number; // 0..63 or -1
  halfmove: number;
}

// square index in python-chess convention: a1=0, b1=1, ... h8=63
function fileOf(sq: number): number { return sq & 7; }
function rankOf(sq: number): number { return sq >> 3; }
function mirrorSquare(sq: number): number { return sq ^ 56; } // vertical flip

export function squareToIndex(alg: string): number {
  // 'e2' -> file*1 + rank*8
  const file = alg.charCodeAt(0) - 97; // 'a'
  const rank = alg.charCodeAt(1) - 49; // '1'
  return rank * 8 + file;
}

export function parseFen(fen: string): ParsedFen {
  const parts = fen.trim().split(/\s+/);
  const [placement, turn, castling, ep, half] = parts;
  const pieces: ParsedFen["pieces"] = new Array(64).fill(null);
  const rows = placement.split("/"); // row 0 = rank 8
  for (let r = 0; r < 8; r++) {
    const actualRank = 7 - r;
    let file = 0;
    for (const ch of rows[r]) {
      if (ch >= "1" && ch <= "8") {
        file += parseInt(ch, 10);
      } else {
        const color = ch === ch.toUpperCase() ? "w" : "b";
        const sq = actualRank * 8 + file;
        pieces[sq] = { type: ch.toLowerCase(), color };
        file += 1;
      }
    }
  }
  return {
    pieces,
    turn: turn === "w" ? "w" : "b",
    castling: castling || "-",
    epSquare: ep && ep !== "-" ? squareToIndex(ep) : -1,
    halfmove: half ? parseInt(half, 10) : 0,
  };
}

export function boardToPlanes(fen: string): Float32Array {
  const p = parseFen(fen);
  const planes = new Float32Array(N_PLANES * 64);
  const stm = p.turn;
  const mirror = stm === "b";

  const set = (plane: number, rank: number, file: number, v: number) => {
    planes[plane * 64 + rank * 8 + file] = v;
  };

  for (let sq = 0; sq < 64; sq++) {
    const piece = p.pieces[sq];
    if (!piece) continue;
    const idxSq = mirror ? mirrorSquare(sq) : sq;
    const rank = rankOf(idxSq);
    const file = fileOf(idxSq);
    const own = piece.color === stm;
    const base = own ? 0 : 6;
    const plane = base + PIECE_IDX[piece.type];
    set(plane, rank, file, 1.0);
  }

  // castling rights from side-to-move perspective
  const wK = p.castling.includes("K");
  const wQ = p.castling.includes("Q");
  const bK = p.castling.includes("k");
  const bQ = p.castling.includes("q");
  const ownK = stm === "w" ? wK : bK;
  const ownQ = stm === "w" ? wQ : bQ;
  const oppK = stm === "w" ? bK : wK;
  const oppQ = stm === "w" ? bQ : wQ;
  const fillPlane = (plane: number) => {
    for (let i = 0; i < 64; i++) planes[plane * 64 + i] = 1.0;
  };
  if (ownK) fillPlane(12);
  if (ownQ) fillPlane(13);
  if (oppK) fillPlane(14);
  if (oppQ) fillPlane(15);

  // en passant file
  if (p.epSquare >= 0) {
    const ep = mirror ? mirrorSquare(p.epSquare) : p.epSquare;
    const file = fileOf(ep);
    for (let rank = 0; rank < 8; rank++) set(16, rank, file, 1.0);
  }

  // fifty-move clock
  const hv = Math.min(p.halfmove, 100) / 100.0;
  for (let i = 0; i < 64; i++) planes[17 * 64 + i] = hv;

  return planes;
}

function deltaToPlane(df: number, dr: number, promotion: string | null): number {
  if (promotion && promotion !== "q") {
    const pieceIdx = UNDERPROMO.indexOf(promotion);
    return 64 + pieceIdx * 3 + (df + 1);
  }
  // knight
  for (let i = 0; i < KNIGHT_DELTAS.length; i++) {
    if (KNIGHT_DELTAS[i][0] === df && KNIGHT_DELTAS[i][1] === dr) return 56 + i;
  }
  // queen-like
  const dist = Math.max(Math.abs(df), Math.abs(dr));
  const sf = df === 0 ? 0 : df / Math.abs(df);
  const sr = dr === 0 ? 0 : dr / Math.abs(dr);
  let dirIdx = -1;
  for (let i = 0; i < DIRECTIONS.length; i++) {
    if (DIRECTIONS[i][0] === sf && DIRECTIONS[i][1] === sr) { dirIdx = i; break; }
  }
  return dirIdx * 7 + (dist - 1);
}

// from/to are algebraic squares ('e2'), promotion is 'q'|'n'|'b'|'r'|null
export function moveToIndex(
  from: string, to: string, promotion: string | null, turn: "w" | "b"
): number {
  let fromSq = squareToIndex(from);
  let toSq = squareToIndex(to);
  if (turn === "b") { fromSq = mirrorSquare(fromSq); toSq = mirrorSquare(toSq); }
  const df = fileOf(toSq) - fileOf(fromSq);
  const dr = rankOf(toSq) - rankOf(fromSq);
  const plane = deltaToPlane(df, dr, promotion);
  return fromSq * 73 + plane;
}
