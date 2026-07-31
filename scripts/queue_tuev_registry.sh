#!/usr/bin/env bash
# Queue the locked TUEV registry only after the seed-42 LR sweep has selected
# a learning rate. This script is intentionally inert until called explicitly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$SCRIPT_DIR/submit_tuev_registry_accre.slurm"

if [[ -z "${TUEV_LR:-}" ]]; then
  echo "Set TUEV_LR to the validation-selected LR before queueing TUEV." >&2
  exit 2
fi

DATA_DIR="${DATA_DIR:-/data/neurogroup/mingyangjiang/data/TUEV}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-25}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LAYER_DECAY="${LAYER_DECAY:-0.65}"
DROP_PATH="${DROP_PATH:-0.1}"
SMOOTHING="${SMOOTHING:-0.1}"
ADAPTER_ZERO_INIT_OUTPUT="${ADAPTER_ZERO_INIT_OUTPUT:-0}"
SEEDS="${SEEDS:-42 1024 3407}"

# Required registry sections. native_depth and axis_blind are deliberately
# opt-in: depth follows the aligned trajectory gate, while axis_blind requires
# the native adapter parameter count and enforces a +/-5% match in preflight.
METHODS="${METHODS:-full_dense frozen_dense frozen_channel frozen_patch frozen_channel_patch native_channel native_patch native_channel_patch generic lora upper2}"
INCLUDE_DEPTH="${INCLUDE_DEPTH:-0}"
INCLUDE_AXIS_BLIND="${INCLUDE_AXIS_BLIND:-0}"
TARGET_ADAPTER_PARAMS="${TARGET_ADAPTER_PARAMS:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$INCLUDE_DEPTH" == 1 ]]; then
  METHODS="$METHODS native_depth"
fi
if [[ "$INCLUDE_AXIS_BLIND" == 1 ]]; then
  if [[ -z "$TARGET_ADAPTER_PARAMS" ]]; then
    echo "Set TARGET_ADAPTER_PARAMS from the selected native run before axis_blind." >&2
    exit 2
  fi
  METHODS="$METHODS axis_blind"
fi

for seed in $SEEDS; do
  for method in $METHODS; do
    run_id="tuev_labram_${method}_lr${TUEV_LR}_s${seed}_b${BATCH_SIZE}_e${EPOCHS}_scale100"
    command=(
      sbatch
      --export="ALL,DATA_DIR=$DATA_DIR,SEED=$seed,BATCH_SIZE=$BATCH_SIZE,NUM_WORKERS=$NUM_WORKERS,EPOCHS=$EPOCHS,LR=$TUEV_LR,WARMUP_EPOCHS=$WARMUP_EPOCHS,WEIGHT_DECAY=$WEIGHT_DECAY,LAYER_DECAY=$LAYER_DECAY,DROP_PATH=$DROP_PATH,SMOOTHING=$SMOOTHING,ADAPTER_ZERO_INIT_OUTPUT=$ADAPTER_ZERO_INIT_OUTPUT,METHOD=$method,TARGET_ADAPTER_PARAMS=$TARGET_ADAPTER_PARAMS,RUN_ID=$run_id"
      "$LAUNCHER"
    )
    printf '%q ' "${command[@]}"
    printf '\n'
    if [[ "$DRY_RUN" != 1 ]]; then
      "${command[@]}"
    fi
  done
done
