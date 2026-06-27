#!/usr/bin/env bash
# Export a checkpoint to ONNX, sanity-check it via the JS engine, deploy to
# Vercel, and commit the model. Usage: ./deploy_snapshot.sh <ckpt.pt> [label]
set -eu
cd "$(dirname "$0")"
ROOT="$(pwd)"
CKPT="${1:?usage: deploy_snapshot.sh <ckpt.pt> [label]}"
LABEL="${2:-rl-snapshot}"

echo "[deploy] exporting $CKPT -> web model"
cd "$ROOT/trainer"
python -m scripts.export_onnx --ckpt "$CKPT" --out ../web/public/model/chessnet.onnx 2>&1 | grep -E "export|parity|size"

echo "[deploy] sanity check via JS engine"
cd "$ROOT/web"
SANITY=$(npx tsx scripts/engine_test.ts 2>&1 | tail -1)
echo "[deploy] $SANITY"
if ! echo "$SANITY" | grep -q PASS; then
  echo "[deploy] SANITY FAILED — aborting"; exit 1
fi

echo "[deploy] deploying to Vercel"
DEP=$(vercel --prod --yes 2>&1 | grep -oE "https://web-[a-z0-9]+-keatonzang[a-z0-9-]*\.vercel\.app" | head -1)
vercel alias set "$DEP" chess-rl-keaton.vercel.app
echo "[deploy] live at https://chess-rl-keaton.vercel.app  (deployment $DEP)"

cd "$ROOT"
git add web/public/model/chessnet.onnx
git commit -q -m "Deploy $LABEL model ($(basename "$CKPT"))

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01E4eS2Yvi4AjnrffeV2VFqR" || echo "[deploy] (no model change to commit)"
git push -q origin master && echo "[deploy] pushed to GitHub"
