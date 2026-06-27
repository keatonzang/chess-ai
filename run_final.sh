#!/usr/bin/env bash
# Autonomous final-model pipeline: wait for enough data, train, sanity-check,
# evaluate vs Stockfish, and redeploy if the sanity check passes.
set -u
cd "$(dirname "$0")"
ROOT="$(pwd)"
LOG="$ROOT/data/final_pipeline.log"
SENTINEL="$ROOT/data/FINAL_DONE"
THRESHOLD=${THRESHOLD:-400000}
SF="$ROOT/tools/stockfish/stockfish-ubuntu-x86-64-avx2"
rm -f "$SENTINEL"

log(){ echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

count(){ cat "$ROOT"/data/shards/*.jsonl 2>/dev/null | wc -l; }

log "=== final pipeline started, threshold=$THRESHOLD ==="

# 1) wait until dataset reaches threshold (or generation has stopped)
while :; do
  c=$(count)
  gen_alive=$(pgrep -f "scripts.generate" | head -1)
  log "waiting: records=$c gen_alive=${gen_alive:-no}"
  if [ "$c" -ge "$THRESHOLD" ]; then log "threshold reached ($c)"; break; fi
  if [ -z "$gen_alive" ]; then log "generation stopped at $c; proceeding"; break; fi
  sleep 60
done

# 2) snapshot
rm -rf "$ROOT/data/final_snapshot"; mkdir -p "$ROOT/data/final_snapshot"
cp "$ROOT"/data/shards/*.jsonl "$ROOT/data/final_snapshot/"
SNAP_COUNT=$(cat "$ROOT"/data/final_snapshot/*.jsonl | wc -l)
log "snapshot taken: $SNAP_COUNT records"

# 3) train final model (detached-safe, full output to log)
cd "$ROOT/trainer"
log "training final model (96ch/8blk)..."
python -m scripts.train --shards "../data/final_snapshot/*.jsonl" \
  --out ../models/final.pt --channels 96 --blocks 8 \
  --epochs 28 --batch 1024 --lr 2e-3 --wd 2e-4 --workers 8 --device cuda:0 \
  >> "$LOG" 2>&1
log "training done"

# 4) export to staging
python -m scripts.export_onnx --ckpt ../models/final_best.pt \
  --out ../models/final.onnx >> "$LOG" 2>&1
log "exported final.onnx"

# 5) sanity check via JS engine using the staged model
cp ../models/final.onnx ../web/public/model/chessnet.onnx
cd "$ROOT/web"
SANITY=$(npx tsx scripts/engine_test.ts 2>&1 | tail -1)
log "sanity: $SANITY"

# 6) strength eval vs Stockfish (informational)
cd "$ROOT/trainer"
log "evaluating strength vs Stockfish..."
for ELO in 1320 1500 1800; do
  R=$(python -m scripts.eval_elo --ckpt ../models/final_best.pt --sf "$SF" \
      --games 8 --sims 160 --sf-elo $ELO --sf-movetime 0.1 --device cuda:0 2>&1 \
      | grep -E "score|Elo difference")
  log "vs SF Elo $ELO -> $R"
done

# 7) deploy if sanity passed
cd "$ROOT/web"
if echo "$SANITY" | grep -q "PASS"; then
  log "deploying to Vercel..."
  DEP=$(vercel --prod --yes 2>&1 | grep -oE "https://web-[a-z0-9]+-keatonzang[a-z0-9-]*\.vercel\.app" | head -1)
  vercel alias set "$DEP" chess-rl-keaton.vercel.app >> "$LOG" 2>&1
  log "deployed: $DEP  -> https://chess-rl-keaton.vercel.app"
  cd "$ROOT"
  git add web/public/model/chessnet.onnx
  git commit -q -m "Deploy final distilled model (trained on ${SNAP_COUNT} positions)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01E4eS2Yvi4AjnrffeV2VFqR" >> "$LOG" 2>&1
  git push -q origin master >> "$LOG" 2>&1
  log "pushed final model to GitHub"
else
  log "SANITY FAILED — not deploying; reverting live model"
  cd "$ROOT" && git checkout -- web/public/model/chessnet.onnx 2>/dev/null
fi

log "=== final pipeline complete ==="
touch "$SENTINEL"
