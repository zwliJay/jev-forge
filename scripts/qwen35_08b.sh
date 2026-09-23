#!/usr/bin/env bash
set -euo pipefail

: "${QWEN35_MODEL_DIR:?Set QWEN35_MODEL_DIR to a local Qwen/Qwen3.5-0.8B snapshot}"

RUN_DIR="${RUN_DIR:-runs/qwen35_08b_same_data_v1}"
STEPS="${STEPS:-1200}"
PY="${PY:-python}"

"$PY" -m jevforge.train \
  --records data/web_full \
  --output-dir "$RUN_DIR" \
  --base-model "$QWEN35_MODEL_DIR" \
  --backbone-init pretrained \
  --max-length 768 \
  --steps "$STEPS" \
  --head-steps 12 \
  --questions-per-batch 8 \
  --microbatch-tokens 12000 \
  --eval-every 50 \
  --wandb \
  --wandb-project jevforge \
  --run-name qwen35-0.8b-same-data-v1
