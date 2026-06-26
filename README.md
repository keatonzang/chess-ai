# ChessRL

A neural-network chess engine you can **play in your browser**. It learns chess
by distilling Stockfish across a deliberately diverse position set — puzzles,
weird/offbeat openings, endgames, random midgames and rare-but-legal positions —
and is then sharpened with AlphaZero-style self-play. At play time it runs an
MCTS search guided by the network, **entirely client-side** (WebGPU → WASM).

> 🎮 **Live demo:** _(deployed on Vercel — link added after first deploy)_

## How it works

**Network** — an AlphaZero-style residual conv net. Input is an 18×8×8 board
tensor encoded from the side-to-move's perspective; outputs are a 4672-way move
policy (the 8×8×73 AlphaZero move encoding) and a scalar position value in
[-1, 1].

**Phase 1 — Stockfish distillation (the bulk of the strength).**
`trainer/scripts/generate.py` streams positions from five sources
(`trainer/chessai/sources.py`):

| source | what it covers |
| --- | --- |
| puzzles | tactics, from the Lichess puzzle database (~6M) |
| openings | normal *and* offbeat lines (Grob, Bongcloud, Wing Gambit, …) |
| endgames | random low-material legal positions |
| random midgames | broad coverage via random self-play of varying depth |
| rare positions | viable-but-unusual structures |

Each position is labeled by Stockfish 17 (multi-PV): a **soft policy target**
(softmax over the top moves' evaluations) and a **value target** (win
probability → [-1, 1]). `trainer/scripts/train.py` distills these into the net.

**Phase 2 — self-play RL.** Bootstrapped from the distilled net, MCTS self-play
generates fresh targets to push past pure imitation.

**Play-time search.** PUCT MCTS guided by the net — the same algorithm in Python
(`trainer/chessai/mcts.py`) and TypeScript (`web/lib/mcts.ts`). The board/move
encoding is ported to TS (`web/lib/encoding.ts`) and verified **bit-for-bit**
against the Python implementation (`npm run parity`).

## Repo layout

```
trainer/        Python training pipeline
  chessai/      encoding, model, datagen, dataset, mcts, sources
  scripts/      generate.py, train.py, export_onnx.py, eval_elo.py
web/            Next.js app deployed to Vercel (in-browser inference)
  lib/          encoding.ts, mcts.ts, engine.ts (ONNX runtime)
  components/   ChessGame.tsx
  public/model/ exported chessnet.onnx
```

## Quick start

### Train (local GPU)
```bash
cd trainer
pip install -r requirements.txt
# 1. download the Lichess puzzle DB to ../data/lichess_puzzles.csv  (optional)
# 2. generate a labeled dataset
python -m scripts.generate --target 600000 --out ../data/shards
# 3. distill
python -m scripts.train --shards "../data/shards/*.jsonl" --out ../models/base.pt
# 4. export for the browser
python -m scripts.export_onnx --ckpt ../models/base_best.pt \
    --out ../web/public/model/chessnet.onnx
# 5. estimate strength vs Stockfish
python -m scripts.eval_elo --ckpt ../models/base_best.pt --games 20 --sf-skill 3
```

Stockfish binary is expected at `tools/stockfish/...`; grab a build from
<https://stockfishchess.org/download/>.

### Web app
```bash
cd web
npm install
npm run parity   # verify TS encoding matches Python
npm run dev
```

## Status

Active demo project. Strength target: mid/low-advanced club level, driven by
distillation quality, dataset size, network capacity and MCTS simulation count.
