#!/usr/bin/env bash
# JevForge full pipeline: build -> augment -> train -> predict -> evaluate.
# Adjust paths to your snapshot of LangAGI-Lab/Mind2Web-axtree-cleaned-lite.
set -euo pipefail
cd "$(dirname "$0")/.."

PARQUET=${PARQUET:-hf_home/hub/datasets--LangAGI-Lab--Mind2Web-axtree-cleaned-lite/snapshots/*/data/*.parquet}
BASE=${BASE:?set BASE to the local Qwen3-0.6B snapshot dir}
PY=${PY:-python3}

$PY -m jevforge.build_web --dataset-glob "$PARQUET" --output-dir data/web

export JEVFORGE_API_KEY=${JEVFORGE_API_KEY:?export JEVFORGE_API_KEY first}
$PY -m jevforge.synthesize --records data/web/all.jsonl --output-dir data/web_augmented \
    --cache artifacts/augmentation_cache.jsonl --soften-lambda 0.25

$PY -m jevforge.train --records data/web_augmented --output-dir runs/web_run \
    --base-model "$BASE" --steps 600 --head-steps 12

$PY -m jevforge.predict --checkpoint-dir runs/web_run --records data/web_augmented \
    --output-dir artifacts/web_run

$PY -m jevforge.evaluate --records data/web_augmented \
    --predictions web=artifacts/web_run --output-dir artifacts/report
