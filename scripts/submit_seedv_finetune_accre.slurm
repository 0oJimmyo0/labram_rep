#!/bin/bash
#SBATCH --job-name=labram_seedv_native
#SBATCH --account=neurogroup_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/logs/out/%x_%j.out
#SBATCH --error=/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/logs/err/%x_%j.err

set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM"
ENV_DIR="/data/neurogroup/mingyangjiang/venvs/labram-accre"
DATA_DIR="${DATA_DIR:-/data/neurogroup/mingyangjiang/data/SEED-V_processed_lmdb}"
SEED="${SEED:-1024}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
REQUIRE_CLEAN_TREE="${REQUIRE_CLEAN_TREE:-1}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
SEEDV_CHANNEL_MANIFEST="${SEEDV_CHANNEL_MANIFEST:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-40}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.03}"
LAYER_DECAY="${LAYER_DECAY:-0.65}"
DROP_PATH="${DROP_PATH:-0.1}"
SMOOTHING="${SMOOTHING:-0.1}"
INPUT_SCALE_DIVISOR="${INPUT_SCALE_DIVISOR:-100.0}"
LABRAM_ADAPTER_TYPE="${LABRAM_ADAPTER_TYPE:-none}"
LABRAM_ADAPTER_BOTTLENECK="${LABRAM_ADAPTER_BOTTLENECK:-64}"
LABRAM_ADAPTER_HEADS="${LABRAM_ADAPTER_HEADS:-4}"
LABRAM_ADAPTER_DROPOUT="${LABRAM_ADAPTER_DROPOUT:-0.0}"
LABRAM_ADAPTER_INIT_ALPHA="${LABRAM_ADAPTER_INIT_ALPHA:-0.01}"
LABRAM_ADAPTER_GAMMA="${LABRAM_ADAPTER_GAMMA:-1.0}"
LABRAM_ADAPTER_LR_SCALE="${LABRAM_ADAPTER_LR_SCALE:-0.1}"
LABRAM_ADAPTER_ALPHA_LR_SCALE="${LABRAM_ADAPTER_ALPHA_LR_SCALE:-0.1}"
LABRAM_ADAPTER_SEED="${LABRAM_ADAPTER_SEED:-12345}"
LABRAM_ADAPTER_DEPTH_MODE="${LABRAM_ADAPTER_DEPTH_MODE:-none}"
LABRAM_ADAPTER_DEPTH_K="${LABRAM_ADAPTER_DEPTH_K:-2}"
SKIP_FINAL_TEST="${SKIP_FINAL_TEST:-1}"
DETERMINISTIC="${DETERMINISTIC:-0}"
RUN_ID="${RUN_ID:-seedv_labram_${LABRAM_ADAPTER_TYPE}_s${SEED}_b${BATCH_SIZE}_lr${LR}_e${EPOCHS}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs/tensorboard/$RUN_ID}"

mkdir -p "$REPO_DIR/logs/out" "$REPO_DIR/logs/err" "$LOG_DIR"

unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
module --force purge >/dev/null 2>&1
module load gentoo/2023 gcc/12.3 python/3.11.5 scipy-stack/2026a >/dev/null 2>&1
export PYTHONNOUSERSITE=1

source "$ENV_DIR/bin/activate"
cd "$REPO_DIR"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ -n "$EXPECTED_COMMIT" && "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "ERROR: expected commit $EXPECTED_COMMIT but found $ACTUAL_COMMIT" >&2
  exit 1
fi
if [[ "$REQUIRE_CLEAN_TREE" == "1" && -n "$(git status --porcelain)" ]]; then
  echo "ERROR: repository is dirty; set REQUIRE_CLEAN_TREE=0 only for exploratory runs" >&2
  git status --short >&2
  exit 1
fi
if [[ "$ALLOW_EXISTING_OUTPUT" != "1" && -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "ERROR: output directory is not empty: $OUTPUT_DIR" >&2
  echo "Use a new RUN_ID or set ALLOW_EXISTING_OUTPUT=1 for an intentional resume." >&2
  exit 1
fi
if [[ -z "$SEEDV_CHANNEL_MANIFEST" || ! -f "$SEEDV_CHANNEL_MANIFEST" ]]; then
  echo "ERROR: SEEDV_CHANNEL_MANIFEST must point to a validated channel manifest" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"

python - <<'PY'
import sys
import warnings
import numpy
import pandas
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
print("gpu0:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print("gpu1:", torch.cuda.get_device_name(1))
try:
    import lmdb
    print("lmdb:", lmdb.__file__)
except Exception as exc:
    raise SystemExit(f"lmdb import failed before launch: {exc}")
PY
nvidia-smi || true
echo "NPROC_PER_NODE=$NPROC_PER_NODE BATCH_SIZE(per-gpu)=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS LR=$LR" >&2
echo "RUN_ID=$RUN_ID SEED=$SEED ADAPTER=$LABRAM_ADAPTER_TYPE DEPTH_MODE=$LABRAM_ADAPTER_DEPTH_MODE DEPTH_K=$LABRAM_ADAPTER_DEPTH_K SCALE=$INPUT_SCALE_DIVISOR SKIP_FINAL_TEST=$SKIP_FINAL_TEST" >&2
echo "git_commit: $ACTUAL_COMMIT" >&2
echo "checkpoint_sha256: $(sha256sum ./checkpoints/labram-base.pth | awk '{print $1}')" >&2

PYTHON_ARGS=(
  --labram_adapter_type "$LABRAM_ADAPTER_TYPE"
  --labram_adapter_bottleneck "$LABRAM_ADAPTER_BOTTLENECK"
  --labram_adapter_heads "$LABRAM_ADAPTER_HEADS"
  --labram_adapter_dropout "$LABRAM_ADAPTER_DROPOUT"
  --labram_adapter_init_alpha "$LABRAM_ADAPTER_INIT_ALPHA"
  --labram_adapter_gamma "$LABRAM_ADAPTER_GAMMA"
  --labram_adapter_lr_scale "$LABRAM_ADAPTER_LR_SCALE"
  --labram_adapter_alpha_lr_scale "$LABRAM_ADAPTER_ALPHA_LR_SCALE"
  --labram_adapter_seed "$LABRAM_ADAPTER_SEED"
  --labram_adapter_depth_mode "$LABRAM_ADAPTER_DEPTH_MODE"
  --labram_adapter_depth_k "$LABRAM_ADAPTER_DEPTH_K"
  --input_scale_divisor "$INPUT_SCALE_DIVISOR"
  --seedv_channel_manifest "$SEEDV_CHANNEL_MANIFEST"
  --smoothing "$SMOOTHING"
  --no_auto_resume
)
if [[ "$SKIP_FINAL_TEST" == "1" ]]; then
  PYTHON_ARGS+=(--skip_final_test)
fi
if [[ "$DETERMINISTIC" == "1" ]]; then
  export CUBLAS_WORKSPACE_CONFIG=":4096:8"
  PYTHON_ARGS+=(--deterministic)
fi

OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" run_class_finetuning.py \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  --model labram_base_patch200_200 \
  --finetune ./checkpoints/labram-base.pth \
  --data_path "$DATA_DIR" \
  --weight_decay "$WEIGHT_DECAY" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --update_freq 1 \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --epochs "$EPOCHS" \
  --layer_decay "$LAYER_DECAY" \
  --drop_path "$DROP_PATH" \
  --save_ckpt_freq 5 \
  --disable_rel_pos_bias \
  --abs_pos_emb \
  --dataset SEED-V \
  --disable_qkv_bias \
  --seed "$SEED" \
  --num_workers "$NUM_WORKERS" \
  "${PYTHON_ARGS[@]}"
