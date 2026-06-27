#!/usr/bin/env bash
# Launch the pure self-play RL system: 1 learner + N actors across both GPUs.
# Pure trial-and-error; no Stockfish. Logs to data/rl_*.log, PIDs in data/rl_pids.
set -u
cd "$(dirname "$0")/trainer"
ROOT="$(cd .. && pwd)"
mkdir -p "$ROOT/data/rl_buffer" "$ROOT/models"
PIDFILE="$ROOT/data/rl_pids"
: > "$PIDFILE"

N_ACTORS=${N_ACTORS:-8}
GAMES=${GAMES:-80}
SIMS=${SIMS:-64}

echo "[run_rl] starting learner on cuda:0"
nohup python -m scripts.rl_train --device cuda:0 \
  --buffer ../data/rl_buffer --ckpt ../models/rl_current.pt \
  --snapdir ../models --batch 1024 --reuse 20 --snapshot-every 10 \
  > "$ROOT/data/rl_learner.log" 2>&1 &
echo $! >> "$PIDFILE"
sleep 8   # let learner publish the initial checkpoint

for i in $(seq 0 $((N_ACTORS-1))); do
  # actors 0-2 on cuda:0 (shared w/ learner), rest on cuda:1
  if [ "$i" -lt 3 ]; then DEV="cuda:0"; else DEV="cuda:1"; fi
  nohup python -m scripts.rl_selfplay --id "$i" --device "$DEV" \
    --ckpt ../models/rl_current.pt --buffer ../data/rl_buffer \
    --games "$GAMES" --sims "$SIMS" \
    > "$ROOT/data/rl_actor_$i.log" 2>&1 &
  echo $! >> "$PIDFILE"
  echo "[run_rl] actor $i on $DEV"
  sleep 1
done

echo "[run_rl] launched learner + $N_ACTORS actors. PIDs in $PIDFILE"
