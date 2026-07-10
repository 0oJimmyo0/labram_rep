#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM"
RUN_ID="${RUN_ID:-faced_labram_${LABRAM_ADAPTER_TYPE:-none}_seed${SEED:-0}}"
JOB_NAME="${JOB_NAME:-$RUN_ID}"

# Slurm opens --output/--error before the batch script starts, so create the
# run-specific parent directories at submission time.
mkdir -p "$REPO_DIR/logs/out/$RUN_ID" "$REPO_DIR/logs/err/$RUN_ID"

exec sbatch \
  --job-name="$JOB_NAME" \
  --output="$REPO_DIR/logs/out/$RUN_ID/%x_%j.out" \
  --error="$REPO_DIR/logs/err/$RUN_ID/%x_%j.err" \
  "$REPO_DIR/scripts/submit_faced_finetune_accre.slurm"
