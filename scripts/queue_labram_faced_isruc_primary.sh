#!/usr/bin/env bash
# Queue the manuscript-grade LaBraM FACED/ISRUC closure packet.
#
# The packet contains frozen dense probes, native low-rank channel+patch
# adapters, and parameter-matched axis-blind controls for both datasets and
# seeds 42/1024/3407.  Two independent dependency lanes are used, so at most
# two one-GPU jobs can run at once.
#
# Dry-run by default.  Submit with DRY_RUN=0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FACED_LAUNCHER="$SCRIPT_DIR/submit_faced_finetune_accre.slurm"
ISRUC_LAUNCHER="$SCRIPT_DIR/submit_isruc_native_accre.slurm"

DRY_RUN="${DRY_RUN:-1}"
FACED_DATA_DIR="${FACED_DATA_DIR:-/data/neurogroup/mingyangjiang/data/FACED}"
ISRUC_DATA_DIR="${ISRUC_DATA_DIR:-/data/neurogroup/mingyangjiang/data/ISRUC}"
SEEDS=(42 1024 3407)
AFTERANY_LANE_A="${AFTERANY_LANE_A:-}"
AFTERANY_LANE_B="${AFTERANY_LANE_B:-}"

if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
  echo "DRY_RUN must be 0 or 1" >&2
  exit 2
fi

submit_one() {
  local dependency="$1"
  local launcher="$2"
  local export_spec="$3"
  local -a command=(sbatch)
  if [[ -n "$dependency" ]]; then
    command+=(--dependency="afterany:$dependency")
  fi
  command+=(--export="$export_spec" "$launcher")
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    sbatch --parsable "${command[@]:1}"
  fi
}

queue_job() {
  local dependency="$1"
  local launcher="$2"
  local export_spec="$3"
  if [[ "$DRY_RUN" == 1 ]]; then
    submit_one "$dependency" "$launcher" "$export_spec"
    LAST_JOB_ID="dry-run"
  else
    LAST_JOB_ID="$(submit_one "$dependency" "$launcher" "$export_spec")"
  fi
}

faced_common="ALL,DATA_DIR=$FACED_DATA_DIR,BATCH_SIZE=32,NUM_WORKERS=4,EPOCHS=80,LR=7e-4,WARMUP_EPOCHS=10,WEIGHT_DECAY=0.05,LAYER_DECAY=0.65,DROP_PATH=0.1,INPUT_SCALE_DIVISOR=100.0,SMOOTHING=0.1,LABRAM_ADAPTER_HEADS=4,LABRAM_ADAPTER_DROPOUT=0.0,LABRAM_ADAPTER_PATCH_OUTPUT_DROPOUT=0.0,LABRAM_ADAPTER_INIT_ALPHA=0.01,LABRAM_ADAPTER_GAMMA=1.0,LABRAM_ADAPTER_ZERO_INIT_OUTPUT=0,LABRAM_ADAPTER_LR_SCALE=1.0,LABRAM_ADAPTER_ALPHA_LR_SCALE=1.0,LABRAM_ADAPTER_SEED=12345,LABRAM_ADAPTER_DEPTH_MODE=none,LABRAM_BACKBONE_MODE=frozen,BACKBONE_LR_SCALE=0.0,HEAD_LR_SCALE=1.0,FROZEN_BACKBONE_EVAL_MODE=1,STRICT_CHECKPOINT_LOAD=1,SKIP_FINAL_TEST=0"
isruc_common="ALL,DATA_DIR=$ISRUC_DATA_DIR,BATCH_SIZE=16,NUM_WORKERS=0,EPOCHS=30,LR=2e-4,WARMUP_EPOCHS=1,SEQUENCE_DROPOUT=0.1,WEIGHT_DECAY=0.05,LAYER_DECAY=0.65,ADAPTER_HEADS=4,ADAPTER_DROPOUT=0.0,ADAPTER_PATCH_OUTPUT_DROPOUT=0.0,ADAPTER_INIT_ALPHA=0.01,ADAPTER_GAMMA=1.0,ADAPTER_LR_SCALE=1.0,ADAPTER_ALPHA_LR_SCALE=1.0,ADAPTER_SEED=12345,ADAPTER_DEPTH_MODE=none,BACKBONE_MODE=frozen,BACKBONE_LR_SCALE=0.0,HEAD_LR_SCALE=1.0,FROZEN_BACKBONE_EVAL_MODE=1,SKIP_FINAL_TEST=0"

queue_lane() {
  local seed="$1"
  local dependency="$2"
  local job_id
  local run_id

  run_id="labram_faced_frozen_dense_primary_s${seed}_b32_lr7e-4_e80_scale100"
  queue_job "$dependency" "$FACED_LAUNCHER" "$faced_common,SEED=$seed,LABRAM_ADAPTER_TYPE=none,LABRAM_ADAPTER_VARIANT=full,LABRAM_ADAPTER_OPERATOR=attention,EXPERIMENT_METHOD=,TARGET_ADAPTER_PARAMS=,RUN_ID=$run_id"
  [[ "$DRY_RUN" == 1 ]] || dependency="$LAST_JOB_ID"

  run_id="labram_faced_frozen_native_cp_lowrank_b64_s${seed}_b32_lr7e-4_e80_scale100"
  queue_job "$dependency" "$FACED_LAUNCHER" "$faced_common,SEED=$seed,LABRAM_ADAPTER_TYPE=channel_patch,LABRAM_ADAPTER_VARIANT=low_rank,LABRAM_ADAPTER_OPERATOR=attention,LABRAM_ADAPTER_BOTTLENECK=64,EXPERIMENT_METHOD=interaction_aligned,TARGET_ADAPTER_PARAMS=,RUN_ID=$run_id"
  [[ "$DRY_RUN" == 1 ]] || dependency="$LAST_JOB_ID"

  run_id="labram_faced_frozen_axisblind_b213_s${seed}_b32_lr7e-4_e80_scale100"
  queue_job "$dependency" "$FACED_LAUNCHER" "$faced_common,SEED=$seed,LABRAM_ADAPTER_TYPE=generic,LABRAM_ADAPTER_VARIANT=full,LABRAM_ADAPTER_OPERATOR=attention,LABRAM_ADAPTER_BOTTLENECK=213,EXPERIMENT_METHOD=axis_blind,TARGET_ADAPTER_PARAMS=85810,RUN_ID=$run_id"
  [[ "$DRY_RUN" == 1 ]] || dependency="$LAST_JOB_ID"

  run_id="labram_isruc_frozen_dense_primary_s${seed}_b16_lr2e-4_e30_scale1"
  queue_job "$dependency" "$ISRUC_LAUNCHER" "$isruc_common,SEED=$seed,ADAPTER_TYPE=none,ADAPTER_VARIANT=full,ADAPTER_BOTTLENECK=64,EXPERIMENT_METHOD=,TARGET_ADAPTER_PARAMS=,RUN_ID=$run_id"
  [[ "$DRY_RUN" == 1 ]] || dependency="$LAST_JOB_ID"

  run_id="labram_isruc_frozen_native_cp_lowrank_b64_s${seed}_b16_lr2e-4_e30_scale1"
  queue_job "$dependency" "$ISRUC_LAUNCHER" "$isruc_common,SEED=$seed,ADAPTER_TYPE=channel_patch,ADAPTER_VARIANT=low_rank,ADAPTER_BOTTLENECK=64,EXPERIMENT_METHOD=interaction_aligned,TARGET_ADAPTER_PARAMS=,RUN_ID=$run_id"
  [[ "$DRY_RUN" == 1 ]] || dependency="$LAST_JOB_ID"

  run_id="labram_isruc_frozen_axisblind_b213_s${seed}_b16_lr2e-4_e30_scale1"
  queue_job "$dependency" "$ISRUC_LAUNCHER" "$isruc_common,SEED=$seed,ADAPTER_TYPE=generic,ADAPTER_VARIANT=full,ADAPTER_BOTTLENECK=213,EXPERIMENT_METHOD=axis_blind,TARGET_ADAPTER_PARAMS=85810,RUN_ID=$run_id"
  job_id="$LAST_JOB_ID"

  if [[ "$DRY_RUN" == 1 ]]; then
    printf '# lane seed=%s has six sequential jobs\n' "$seed"
  else
    printf 'lane seed=%s final_job=%s\n' "$seed" "$job_id" >&2
  fi
}

if [[ "$DRY_RUN" == 1 ]]; then
  queue_lane "${SEEDS[0]}" "$AFTERANY_LANE_A"
  queue_lane "${SEEDS[1]}" "$AFTERANY_LANE_B"
  queue_lane "${SEEDS[2]}" ""
else
  lane_a="$(queue_lane "${SEEDS[0]}" "$AFTERANY_LANE_A" 2>&1 | tail -1 | sed -E 's/.*final_job=([0-9]+).*/\1/')"
  lane_b="$(queue_lane "${SEEDS[1]}" "$AFTERANY_LANE_B" 2>&1 | tail -1 | sed -E 's/.*final_job=([0-9]+).*/\1/')"
  queue_lane "${SEEDS[2]}" "$lane_a:$lane_b" >/dev/null
fi
