#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python}"
RECORDS="${RECORDS:-data/web_full}"
SOURCE_RUN="${SOURCE_RUN:-runs/qwen35_08b_same_data_v1}"
RUN_DIR="${RUN_DIR:-runs/qwen35_08b_rlcd_v1}"
PRED_DIR="${PRED_DIR:-artifacts/qwen35_08b_rlcd_v1}"
REPORT_DIR="${REPORT_DIR:-artifacts/report_qwen35_08b_rlcd_v1}"
STEPS="${STEPS:-200}"
MODE="${MODE:-exact}"
RUN_NAME="${RUN_NAME:-qwen35-0.8b-rlcd-${MODE}-v1}"

"$PY" -m jevforge.rlcd \
  --records "$RECORDS" \
  --checkpoint-dir "$SOURCE_RUN" \
  --output-dir "$RUN_DIR" \
  --mode "$MODE" \
  --steps "$STEPS" \
  --questions-per-batch 6 \
  --microbatch-tokens 10000 \
  --group-size 8 \
  --learning-rate 5e-6 \
  --head-learning-rate 5e-5 \
  --utility-weight 1.0 \
  --calibration-weight 0.5 \
  --kl-weight 0.02 \
  --eval-every 50 \
  --dev-questions 300 \
  --final-questions 300 \
  --wandb \
  --wandb-project jevforge \
  --run-name "$RUN_NAME"

"$PY" -m jevforge.predict \
  --checkpoint-dir "$RUN_DIR" \
  --records "$RECORDS" \
  --splits test,ood \
  --output-dir "$PRED_DIR"

"$PY" -m jevforge.evaluate \
  --records "$RECORDS" \
  --predictions "rlcd=$PRED_DIR" \
  --splits test,ood \
  --output-dir "$REPORT_DIR"
