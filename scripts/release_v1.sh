#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
QWEN="${QWEN:?set QWEN to the local Qwen3-0.6B configuration directory}"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -m jevforge.train \
  --records data/web_full --output-dir runs/release_v1 --base-model "$QWEN" \
  --max-length 768 --steps 1200 --head-steps 12 --questions-per-batch 12 \
  --microbatch-tokens 16000 --eval-every 50 \
  --wandb --wandb-project jevforge --run-name release-v1-recipe \
  > artifacts/train_release_v1.log 2>&1
$PY -m jevforge.predict --checkpoint-dir runs/release_v1 --records data/web_pilot \
  --output-dir artifacts/release_v1 > artifacts/predict_release.log 2>&1
$PY -m jevforge.evaluate --records data/web_pilot \
  --predictions release=artifacts/release_v1 --splits test,ood \
  --output-dir artifacts/report_release > artifacts/evaluate_release.log 2>&1
echo DONE > artifacts/release_v1.done
