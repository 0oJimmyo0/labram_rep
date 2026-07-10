#!/bin/bash
#SBATCH --job-name=labram_faced
#SBATCH --account=neurogroup_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/logs/out/%x/%j.out
#SBATCH --error=/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/logs/err/%x/%j.err

set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM"
ENV_DIR="/data/neurogroup/mingyangjiang/venvs/labram-accre"
DATA_DIR="${DATA_DIR:-/data/neurogroup/mingyangjiang/data/FACED}"
SEED="${SEED:-0}"
DETERMINISTIC="${DETERMINISTIC:-0}"
LABRAM_ADAPTER_TYPE="${LABRAM_ADAPTER_TYPE:-none}"
RUN_ID="${RUN_ID:-faced_labram_${LABRAM_ADAPTER_TYPE}_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs/tensorboard/$RUN_ID}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-40}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LAYER_DECAY="${LAYER_DECAY:-0.65}"
DROP_PATH="${DROP_PATH:-0.1}"
LABRAM_ADAPTER_BOTTLENECK="${LABRAM_ADAPTER_BOTTLENECK:-64}"
LABRAM_ADAPTER_HEADS="${LABRAM_ADAPTER_HEADS:-4}"
LABRAM_ADAPTER_DROPOUT="${LABRAM_ADAPTER_DROPOUT:-0.0}"
LABRAM_ADAPTER_INIT_ALPHA="${LABRAM_ADAPTER_INIT_ALPHA:-0.01}"
LABRAM_ADAPTER_GAMMA="${LABRAM_ADAPTER_GAMMA:-1.0}"
LABRAM_ADAPTER_SEED="${LABRAM_ADAPTER_SEED:-12345}"
LABRAM_ADAPTER_USE_TOKEN_MLP="${LABRAM_ADAPTER_USE_TOKEN_MLP:-0}"
LABRAM_ADAPTER_DEPTH_MODE="${LABRAM_ADAPTER_DEPTH_MODE:-none}"
LABRAM_ADAPTER_DEPTH_K="${LABRAM_ADAPTER_DEPTH_K:-4}"
LABRAM_ADAPTER_GAMMA_ZERO_SKIP_BRANCH="${LABRAM_ADAPTER_GAMMA_ZERO_SKIP_BRANCH:-0}"

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
echo "FACED LaBraM launch: BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS LR=$LR WD=$WEIGHT_DECAY EPOCHS=$EPOCHS ADAPTER=$LABRAM_ADAPTER_TYPE" >&2
echo "git_commit: $(git rev-parse HEAD)" >&2
if [[ -f ./checkpoints/labram-base.pth ]]; then
  echo "checkpoint_sha256: $(sha256sum ./checkpoints/labram-base.pth | awk '{print $1}')" >&2
fi
echo "run_id: $RUN_ID seed: $SEED" >&2

PYTHON_ARGS=()
if [[ "$LABRAM_ADAPTER_USE_TOKEN_MLP" == "1" ]]; then
  PYTHON_ARGS+=(--labram_adapter_use_token_mlp)
fi
if [[ "$LABRAM_ADAPTER_GAMMA_ZERO_SKIP_BRANCH" == "1" ]]; then
  PYTHON_ARGS+=(--labram_adapter_gamma_zero_skip_branch)
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
  --labram_adapter_type "$LABRAM_ADAPTER_TYPE" \
  --labram_adapter_bottleneck "$LABRAM_ADAPTER_BOTTLENECK" \
  --labram_adapter_heads "$LABRAM_ADAPTER_HEADS" \
  --labram_adapter_dropout "$LABRAM_ADAPTER_DROPOUT" \
  --labram_adapter_init_alpha "$LABRAM_ADAPTER_INIT_ALPHA" \
  --labram_adapter_gamma "$LABRAM_ADAPTER_GAMMA" \
  --labram_adapter_seed "$LABRAM_ADAPTER_SEED" \
  --labram_adapter_depth_mode "$LABRAM_ADAPTER_DEPTH_MODE" \
  --labram_adapter_depth_k "$LABRAM_ADAPTER_DEPTH_K" \
  --save_ckpt_freq 5 \
  --disable_rel_pos_bias \
  --abs_pos_emb \
  --dataset FACED \
  --disable_qkv_bias \
  --seed "$SEED" \
  --num_workers "$NUM_WORKERS" \
  "${PYTHON_ARGS[@]}"
