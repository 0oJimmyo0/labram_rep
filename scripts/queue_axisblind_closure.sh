#!/usr/bin/env bash
# Queue only the capacity-matched axis-blind closure controls.
# The default is a dry run; set DRY_RUN=0 after reviewing the printed commands.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISRUC_LAUNCHER="$SCRIPT_DIR/submit_isruc_native_accre.slurm"
TUEV_LAUNCHER="$SCRIPT_DIR/submit_tuev_registry_accre.slurm"

DRY_RUN="${DRY_RUN:-1}"
ISRUC_DATA_DIR="${ISRUC_DATA_DIR:-/data/neurogroup/mingyangjiang/data/ISRUC}"
TUEV_DATA_DIR="${TUEV_DATA_DIR:-/data/neurogroup/mingyangjiang/data/TUEV}"
SEEDS="${SEEDS:-42 1024 3407}"
AFTERANY_JOBS="${AFTERANY_JOBS:-}"

submit_one() {
  local dependency="$1"
  local export_spec="$2"
  local launcher="$3"
  local cmd=(sbatch)
  if [[ -n "$dependency" ]]; then
    cmd+=(--dependency="afterany:$dependency")
  fi
  cmd+=(--export="$export_spec" "$launcher")
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  sbatch --parsable "${cmd[@]:1}"
}

dependency="$AFTERANY_JOBS"
for seed in $SEEDS; do
  isruc_run_id="isruc_labram_axis_blind_b400_s${seed}_b16_lr2e-4_e30"
  isruc_export="ALL,DATA_DIR=$ISRUC_DATA_DIR,SEED=$seed,BATCH_SIZE=16,NUM_WORKERS=0,EPOCHS=30,LR=2e-4,WEIGHT_DECAY=0.05,LAYER_DECAY=0.65,WARMUP_EPOCHS=1,SEQUENCE_DROPOUT=0.1,ADAPTER_TYPE=generic,ADAPTER_VARIANT=full,ADAPTER_BOTTLENECK=400,ADAPTER_HEADS=4,ADAPTER_DROPOUT=0.0,ADAPTER_INIT_ALPHA=0.01,ADAPTER_GAMMA=1.0,ADAPTER_LR_SCALE=1.0,ADAPTER_ALPHA_LR_SCALE=1.0,ADAPTER_SEED=12345,BACKBONE_MODE=frozen,BACKBONE_LR_SCALE=0.0,HEAD_LR_SCALE=1.0,FROZEN_BACKBONE_EVAL_MODE=1,EXPERIMENT_METHOD=axis_blind,TARGET_ADAPTER_PARAMS=161201,SKIP_FINAL_TEST=0,RUN_ID=$isruc_run_id"
  if [[ "$DRY_RUN" == 1 ]]; then
    submit_one "$dependency" "$isruc_export" "$ISRUC_LAUNCHER"
  else
    isruc_job_id="$(submit_one "$dependency" "$isruc_export" "$ISRUC_LAUNCHER")"
  fi

  tuev_run_id="tuev_labram_axis_blind_b213_s${seed}_b64_lr5e-4_e15_scale100"
  tuev_export="ALL,DATA_DIR=$TUEV_DATA_DIR,SEED=$seed,BATCH_SIZE=64,NUM_WORKERS=4,EPOCHS=15,LR=5e-4,WARMUP_EPOCHS=5,WEIGHT_DECAY=0.05,LAYER_DECAY=0.65,DROP_PATH=0.2,SMOOTHING=0.1,INPUT_SCALE_DIVISOR=100.0,ADAPTER_BOTTLENECK=213,ADAPTER_HEADS=4,ADAPTER_DROPOUT=0.0,ADAPTER_INIT_ALPHA=0.01,ADAPTER_GAMMA=1.0,ADAPTER_LR_SCALE=1.0,ADAPTER_ALPHA_LR_SCALE=1.0,ADAPTER_SEED=12345,METHOD=axis_blind,TARGET_ADAPTER_PARAMS=85810,RUN_ID=$tuev_run_id"
  if [[ "$DRY_RUN" == 1 ]]; then
    submit_one "$dependency" "$tuev_export" "$TUEV_LAUNCHER"
  else
    tuev_job_id="$(submit_one "$dependency" "$tuev_export" "$TUEV_LAUNCHER")"
  fi

  if [[ "$DRY_RUN" != 1 ]]; then
    dependency="$isruc_job_id:$tuev_job_id"
  fi
done
