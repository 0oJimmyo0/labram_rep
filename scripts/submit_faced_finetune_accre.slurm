#!/bin/bash
#SBATCH --job-name=labram_faced
#SBATCH --account=neurogroup_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/logs/out/%x_%j.out
#SBATCH --error=/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/logs/err/%x_%j.err

set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM"
ENV_DIR="/data/neurogroup/mingyangjiang/venvs/labram-accre"
DATA_DIR="${DATA_DIR:-/data/neurogroup/mingyangjiang/data/FACED}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/finetune_faced_base}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs/tensorboard/finetune_faced_base}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-40}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LAYER_DECAY="${LAYER_DECAY:-0.65}"
DROP_PATH="${DROP_PATH:-0.1}"

mkdir -p "$REPO_DIR/logs/out" "$REPO_DIR/logs/err" "$LOG_DIR" "$OUTPUT_DIR"

unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
module --force purge >/dev/null 2>&1
module load gentoo/2023 gcc/12.3 python/3.11.5 scipy-stack/2026a >/dev/null 2>&1
export PYTHONNOUSERSITE=1

source "$ENV_DIR/bin/activate"
cd "$REPO_DIR"

python - <<'PY'
import sys, torch, lmdb
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
print("lmdb:", lmdb.__file__)
PY
nvidia-smi || true
echo "FACED baseline launch: BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS LR=$LR WD=$WEIGHT_DECAY EPOCHS=$EPOCHS" >&2

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
  --dataset FACED \
  --disable_qkv_bias \
  --seed 0 \
  --num_workers "$NUM_WORKERS"
