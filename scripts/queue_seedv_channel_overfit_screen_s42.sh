#!/bin/bash
set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM"
cd "$REPO_DIR"

COMMIT="$(git rev-parse HEAD)"
SEEDV_CHANNEL_MANIFEST="${SEEDV_CHANNEL_MANIFEST:-$REPO_DIR/docs/seedv_channel_manifest_provisional.json}"
SEED=42
BATCH_SIZE=16
EPOCHS=40
WEIGHT_DECAY=0.05
LAYER_DECAY=0.65
DROP_PATH=0.1
SMOOTHING=0.1
INPUT_SCALE_DIVISOR=100.0

submit_channel() {
  local run_id="$1" lr="$2" warmup="$3" backbone_scale="$4"
  sbatch \
    --export=ALL,EXPECTED_COMMIT="$COMMIT",REQUIRE_CLEAN_TREE=1,ALLOW_EXISTING_OUTPUT=0,\
SEED="$SEED",RUN_ID="$run_id",SEEDV_CHANNEL_MANIFEST="$SEEDV_CHANNEL_MANIFEST",\
LABRAM_BACKBONE_MODE=trainable,LABRAM_ADAPTER_TYPE=channel,LABRAM_ADAPTER_VARIANT=full,\
BACKBONE_LR_SCALE="$backbone_scale",HEAD_LR_SCALE=1.0,FROZEN_BACKBONE_EVAL_MODE=0,\
LABRAM_ADAPTER_INIT_ALPHA=0.01,LABRAM_ADAPTER_GAMMA=1.0,LABRAM_ADAPTER_LR_SCALE=1.0,\
LABRAM_ADAPTER_ALPHA_LR_SCALE=0.5,LABRAM_ADAPTER_DEPTH_MODE=none,\
BATCH_SIZE="$BATCH_SIZE",LR="$lr",EPOCHS="$EPOCHS",WARMUP_EPOCHS="$warmup",\
WEIGHT_DECAY="$WEIGHT_DECAY",LAYER_DECAY="$LAYER_DECAY",DROP_PATH="$DROP_PATH",\
SMOOTHING="$SMOOTHING",INPUT_SCALE_DIVISOR="$INPUT_SCALE_DIVISOR",NUM_WORKERS=0,\
SKIP_FINAL_TEST=0 \
    scripts/submit_seedv_finetune_accre.slurm
}

# A: isolate the proposed modest alpha-LR increase.
submit_channel seedv_labram_channel_alpha05_full_s42_lr3e-4_b16_e40_test 3e-4 5 1.0

# B: partial-plasticity bridge; same global recipe, backbone updates reduced.
submit_channel seedv_labram_channel_alpha05_bls0.1_s42_lr3e-4_b16_e40_test 3e-4 5 0.1

# C: conservative ISRUC-inspired bridge; lower LR and shorter warmup.
submit_channel seedv_labram_channel_alpha05_bls0.1_conservative_s42_lr2e-4_b16_e40_test 2e-4 1 0.1
