#!/bin/bash
#SBATCH --job-name=labram_seedv_2gpu
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
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/finetune_seedv_base}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs/tensorboard/finetune_seedv_base}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EPOCHS="${EPOCHS:-50}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
LR="${LR:-5e-4}"

mkdir -p "$REPO_DIR/logs/out" "$REPO_DIR/logs/err" "$LOG_DIR" "$OUTPUT_DIR"

unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
module --force purge >/dev/null 2>&1
module load gentoo/2023 gcc/12.3 python/3.11.5 scipy-stack/2026a >/dev/null 2>&1
export PYTHONNOUSERSITE=1

source "$ENV_DIR/bin/activate"
cd "$REPO_DIR"

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

OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" run_class_finetuning.py \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  --model labram_base_patch200_200 \
  --finetune ./checkpoints/labram-base.pth \
  --data_path "$DATA_DIR" \
  --weight_decay 0.05 \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --update_freq 1 \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --epochs "$EPOCHS" \
  --layer_decay 0.65 \
  --drop_path 0.1 \
  --save_ckpt_freq 5 \
  --disable_rel_pos_bias \
  --abs_pos_emb \
  --dataset SEED-V \
  --disable_qkv_bias \
  --seed 0 \
  --num_workers "$NUM_WORKERS"
