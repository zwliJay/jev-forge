#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python3}"
CACHE_DIR="${CACHE_DIR:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
RUN_DIR="${RUN_DIR:-runs/qwen35_08b_same_data_v1}"
PRED_DIR="${PRED_DIR:-artifacts/qwen35_08b_same_data_v1}"
REPORT_DIR="${REPORT_DIR:-artifacts/report_qwen35_08b_same_data_v1}"

set -a
test ! -f .env || . ./.env
set +a

while ! "$PY" scripts/fetch_hf_model.py Qwen/Qwen3.5-0.8B --cache-dir "$CACHE_DIR"; do
  echo "[$(date -Is)] model download failed; retrying in 60 seconds"
  sleep 60
done

QWEN35_MODEL_DIR="$(find "$CACHE_DIR/models--Qwen--Qwen3.5-0.8B/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
test -n "$QWEN35_MODEL_DIR"
export QWEN35_MODEL_DIR RUN_DIR PY

bash scripts/qwen35_08b.sh

"$PY" -m jevforge.predict \
  --checkpoint-dir "$RUN_DIR" \
  --records data/web_full \
  --splits test,ood \
  --output-dir "$PRED_DIR"

"$PY" -m jevforge.evaluate \
  --records data/web_full \
  --predictions qwen35_08b="$PRED_DIR" \
  --splits test,ood \
  --output-dir "$REPORT_DIR"

printf 'completed_at=%s\nmodel=%s\nrun_dir=%s\nreport_dir=%s\n' \
  "$(date -Is)" "$QWEN35_MODEL_DIR" "$RUN_DIR" "$REPORT_DIR" \
  > artifacts/qwen35_08b.done
