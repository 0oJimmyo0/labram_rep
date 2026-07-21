#!/bin/bash
set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM"
cd "$REPO_DIR"

COMMIT="$(git rev-parse HEAD)"
SEEDV_CHANNEL_MANIFEST="${SEEDV_CHANNEL_MANIFEST:-$REPO_DIR/docs/seedv_channel_manifest_provisional.json}"
COMMON_EXPORT="EXPECTED_COMMIT=$COMMIT,REQUIRE_CLEAN_TREE=1,ALLOW_EXISTING_OUTPUT=0,SEED=42,SEEDV_CHANNEL_MANIFEST=$SEEDV_CHANNEL_MANIFEST,LABRAM_BACKBONE_MODE=frozen,LABRAM_ADAPTER_TYPE=none,LABRAM_ADAPTER_VARIANT=full,BACKBONE_LR_SCALE=0.0,FROZEN_BACKBONE_EVAL_MODE=1,LABRAM_ADAPTER_DEPTH_MODE=none,BATCH_SIZE=32,LR=1e-4,EPOCHS=40,WEIGHT_DECAY=0.03,LAYER_DECAY=0.65,DROP_PATH=0.1,INPUT_SCALE_DIVISOR=100.0,NUM_WORKERS=0,SKIP_FINAL_TEST=1"

# D0: existing head recipe with deterministic frozen features.
sbatch --export=ALL,$COMMON_EXPORT,HEAD_LR_SCALE=1.0,HEAD_WEIGHT_DECAY=,WARMUP_EPOCHS=5,SMOOTHING=0.1,RUN_ID=seedv_labram_frozen_dense_d0_s42_b32_lr1e-4_e40_val scripts/submit_seedv_finetune_accre.slurm

# D1: calibrated frozen linear probe; only head optimization changes.
sbatch --export=ALL,$COMMON_EXPORT,HEAD_LR_SCALE=10.0,HEAD_WEIGHT_DECAY=0.0,WARMUP_EPOCHS=0,SMOOTHING=0.0,RUN_ID=seedv_labram_frozen_dense_d1_s42_b32_headlr1e-3_e40_val scripts/submit_seedv_finetune_accre.slurm
