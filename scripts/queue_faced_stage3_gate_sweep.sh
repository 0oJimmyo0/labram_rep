#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM"
SUBMIT="$REPO_DIR/scripts/submit_faced_finetune_accre.sh"

for seed in 42 1024; do
  for alpha_scale in 0.1 0.3 1.0; do
    alpha_tag="${alpha_scale/./}"
    RUN_ID="faced_stage3a_patch_lr7e4_core01_alpha${alpha_tag}_init001_s${seed}"
    env \
      RUN_ID="$RUN_ID" JOB_NAME="$RUN_ID" SEED="$seed" \
      LR=7e-4 EPOCHS=80 WARMUP_EPOCHS=10 BATCH_SIZE=32 NUM_WORKERS=4 \
      LABRAM_ADAPTER_TYPE=patch \
      LABRAM_ADAPTER_LR_SCALE=0.1 \
      LABRAM_ADAPTER_ALPHA_LR_SCALE="$alpha_scale" \
      LABRAM_ADAPTER_INIT_ALPHA=0.01 \
      LABRAM_ADAPTER_FIXED_ALPHA= \
      SKIP_FINAL_TEST=1 \
      "$SUBMIT"
  done

  RUN_ID="faced_stage3a_patch_lr7e4_core01_fixedalpha001_s${seed}"
  env \
    RUN_ID="$RUN_ID" JOB_NAME="$RUN_ID" SEED="$seed" \
    LR=7e-4 EPOCHS=80 WARMUP_EPOCHS=10 BATCH_SIZE=32 NUM_WORKERS=4 \
    LABRAM_ADAPTER_TYPE=patch \
    LABRAM_ADAPTER_LR_SCALE=0.1 \
    LABRAM_ADAPTER_ALPHA_LR_SCALE=0.1 \
    LABRAM_ADAPTER_INIT_ALPHA=0.01 \
    LABRAM_ADAPTER_FIXED_ALPHA=0.01 \
    SKIP_FINAL_TEST=1 \
    "$SUBMIT"
done
