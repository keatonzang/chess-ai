# ChessRL

A neural-network chess engine you can **play in your browser** that learns to
play **purely by trial and error** — self-play reinforcement learning from
*random* weights, AlphaZero-style. **No Stockfish, no human games, and no engine
evaluations are used as training signal.** The only feedback the network ever
gets is the outcome of its own games (win / draw / loss) plus the policy
improvement that MCTS search provides. Stockfish is used *only at the end*, as a
measuring stick to gauge the bot's strength.

> 🎮 **Live demo:** <https://chess-rl-keaton.vercel.app>

## How it learns (pure RL)

Classic AlphaZero-style self-play, from scratch:

1. **Actors** play games against themselves using MCTS guided by the current
   network. Early moves are sampled (with Dirichlet root noise) for exploration.
2. Each visited position is stored with the **MCTS visit distribution** (the
   improved policy) and, once the game ends, the **game outcome** from that
   position's point of view (the reward).
3. The **learner** trains the network to match those visit policies and outcomes
   on a sliding replay buffer, then publishes an updated network the actors pick
   up — and the loop repeats. The network bootstraps entirely off its own play.

There is no supervised target anywhere in the loop. The network starts knowing
nothing (random init) and discovers material, tactics and mating patterns by
playing millions of its own games.

> **Compute note.** Pure self-play RL from random weights is enormously more
> compute-hungry than imitation/distillation — the original AlphaZero used
> thousands of accelerators. On a 2-GPU workstation this reaches *beginner→
> intermediate* play in a feasible time and keeps climbing the longer it trains;
> it is not going to match a datacenter run. Strength is measured honestly vs
> Stockfish (see below) rather than assumed.

**Play-time search.** PUCT MCTS guided by the net — the same algorithm in Python
(`trainer/chessai/mcts.py`, `rl.py`) and TypeScript (`web/lib/mcts.ts`). The
board/move encoding is ported to TS (`web/lib/encoding.ts`) and verified
**bit-for-bit** against the Python implementation (`npm run parity`).

**Network.** AlphaZero-style residual conv net. Input is an 18×8×8 board tensor
(side-to-move perspective); outputs are a 4672-way move policy (the 8×8×73
AlphaZero move encoding) and a scalar value in [-1, 1]. ~11M params (96ch/8blk),
small enough to run in a browser.

## Repo layout

```
trainer/        Python RL pipeline
  chessai/      encoding, model, mcts, rl (efficient batched self-play)
  scripts/      rl_selfplay.py (actor), rl_train.py (learner), eval_elo.py
web/            Next.js app deployed to Vercel (in-browser inference)
  lib/          encoding.ts, mcts.ts, engine.ts (ONNX runtime)
  components/   ChessGame.tsx
  public/model/ exported chessnet.onnx
run_rl.sh       launch the self-play RL system (learner + N actors)
```

## Quick start

### Train by self-play (local GPUs)
```bash
cd trainer && pip install -r requirements.txt
cd .. && ./run_rl.sh                 # 1 learner + 8 actors across both GPUs
tail -f data/rl_learner.log          # watch policy/value loss + games seen
# export a snapshot for the browser:
cd trainer && python -m scripts.export_onnx --ckpt ../models/rl_iter_0050.pt \
    --out ../web/public/model/chessnet.onnx
```

### Gauge strength vs Stockfish (measurement only)
```bash
cd trainer
python -m scripts.eval_elo --ckpt ../models/rl_current.pt --games 20 \
    --sims 200 --sf-elo 1320
```

### Web app
```bash
cd web && npm install
npm run parity   # verify TS encoding matches Python
npm run dev
```

## Status

Active project. The deployed model is updated with self-play RL snapshots as the
bot improves. Strength is reported from head-to-head games vs Stockfish at capped
levels.
