#!/usr/bin/env bash
# Resume RL training from the paused checkpoint (continues, does NOT restart from
# random weights). The learner loads rl_paused.pt and republishes it; actors pick
# it up. Note: the in-memory replay buffer was not persisted, so it warms up
# fresh for a few minutes before training resumes — the trained weights carry over.
set -u
cd "$(dirname "$0")"
if [ ! -f models/rl_paused.pt ]; then
  echo "no models/rl_paused.pt — nothing to resume"; exit 1
fi
# seed rl_current.pt from the pause checkpoint so actors start on trained weights
cp models/rl_paused.pt models/rl_current.pt
RESUME=../models/rl_paused.pt ./run_rl.sh
