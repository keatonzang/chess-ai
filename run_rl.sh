#!/usr/bin/env bash
# Launch the pure self-play RL system: 1 learner + N actors across both GPUs.
# Pure trial-and-error; no Stockfish. Logs to data/rl_*.log, PIDs in data/rl_pids.
set -u
cd "$(dirname "$0")/trainer"
ROOT="$(cd .. && pwd)"
mkdir -p "$ROOT/data/rl_buffer" "$ROOT/models"
PIDFILE="$ROOT/data/rl_pids"
: > "$PIDFILE"

N_ACTORS=${N_ACTORS:-6}        # pure self-play actors (diverse-seeded)
N_SF=${N_SF:-2}               # bot-vs-Stockfish actors (surprise + curriculum + novelty)
N_LEAGUE=${N_LEAGUE:-2}       # league actors (current net vs past snapshots)
GAMES=${GAMES:-24}
SIMS=${SIMS:-64}
MAXMOVES=${MAXMOVES:-120}
LAM=${LAM:-0.5}
SF_SKILL_SET=${SF_SKILL_SET:-0,1,3,5}
RESUME=${RESUME:-}            # set to ../models/rl_current.pt to continue training

echo "[run_rl] starting learner on cuda:0"
RESUME_ARG=""
[ -n "$RESUME" ] && RESUME_ARG="--resume $RESUME"
nohup python -m scripts.rl_train --device cuda:0 \
  --buffer ../data/rl_buffer --ckpt ../models/rl_current.pt \
  --snapdir ../models --batch 1024 --reuse 20 --snapshot-every 10 $RESUME_ARG \
  > "$ROOT/data/rl_learner.log" 2>&1 &
echo $! >> "$PIDFILE"
sleep 8   # let learner publish the initial checkpoint

for i in $(seq 0 $((N_ACTORS-1))); do
  # actors 0-2 on cuda:0 (shared w/ learner), rest on cuda:1
  if [ "$i" -lt 3 ]; then DEV="cuda:0"; else DEV="cuda:1"; fi
  nohup python -m scripts.rl_selfplay --id "$i" --device "$DEV" \
    --ckpt ../models/rl_current.pt --buffer ../data/rl_buffer \
    --games "$GAMES" --sims "$SIMS" --max-moves "$MAXMOVES" \
    > "$ROOT/data/rl_actor_$i.log" 2>&1 &
  echo $! >> "$PIDFILE"
  echo "[run_rl] actor $i on $DEV"
  sleep 1
done

for j in $(seq 0 $((N_SF-1))); do
  nohup python -m scripts.rl_vs_sf --id "$j" --device cuda:1 \
    --ckpt ../models/rl_current.pt --buffer ../data/rl_buffer \
    --sf ../tools/stockfish/stockfish-ubuntu-x86-64-avx2 \
    --sims "$SIMS" --lam "$LAM" --sf-skill-set "$SF_SKILL_SET" --start-level 0 \
    > "$ROOT/data/rl_vssf_$j.log" 2>&1 &
  echo $! >> "$PIDFILE"
  echo "[run_rl] vs-Stockfish actor $j on cuda:1 (lam=$LAM, skills=$SF_SKILL_SET)"
  sleep 1
done

for k in $(seq 0 $((N_LEAGUE-1))); do
  nohup python -m scripts.rl_league --id "$k" --device cuda:1 \
    --ckpt ../models/rl_current.pt --snapdir ../models --buffer ../data/rl_buffer \
    --sims "$SIMS" > "$ROOT/data/rl_league_$k.log" 2>&1 &
  echo $! >> "$PIDFILE"
  echo "[run_rl] league actor $k on cuda:1 (current net vs past snapshots)"
  sleep 1
done

echo "[run_rl] launched learner + $N_ACTORS self-play + $N_SF vs-SF + $N_LEAGUE league actors."
