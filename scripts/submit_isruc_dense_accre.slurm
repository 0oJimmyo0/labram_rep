#!/bin/bash
#SBATCH --job-name=labram_isruc_dense
#SBATCH --account=neurogroup_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/logs/out/%x_%j.out
#SBATCH --error=/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/logs/err/%x_%j.err

set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM"
ENV_DIR="/data/neurogroup/mingyangjiang/venvs/labram-accre"
DATA_DIR="${DATA_DIR:-/data/neurogroup/mingyangjiang/data/ISRUC}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LAYER_DECAY="${LAYER_DECAY:-0.65}"
BACKBONE_LR_SCALE="${BACKBONE_LR_SCALE:-1.0}"
HEAD_LR_SCALE="${HEAD_LR_SCALE:-1.0}"
SEQUENCE_DROPOUT="${SEQUENCE_DROPOUT:-0.1}"
INPUT_SCALE_DIVISOR="${INPUT_SCALE_DIVISOR:-1.0}"
SKIP_FINAL_TEST="${SKIP_FINAL_TEST:-0}"
RUN_ID="${RUN_ID:-isruc_labram_dense_lr${LR}_s${SEED}_b${BATCH_SIZE}_e${EPOCHS}_scale${INPUT_SCALE_DIVISOR}_bls${BACKBONE_LR_SCALE}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs/tensorboard/$RUN_ID}"

if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to reuse nonempty output directory: $OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$REPO_DIR/logs/out" "$REPO_DIR/logs/err" "$LOG_DIR" "$OUTPUT_DIR"
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
module --force purge >/dev/null 2>&1
module load gentoo/2023 gcc/12.3 python/3.11.5 scipy-stack/2026a >/dev/null 2>&1
export PYTHONNOUSERSITE=1
source "$ENV_DIR/bin/activate"
cd "$REPO_DIR"

python - <<'PY'
import sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("ISRUC dense run requires a CUDA GPU")
PY
nvidia-smi

echo "ISRUC dense launch: seed=$SEED batch=$BATCH_SIZE workers=$NUM_WORKERS epochs=$EPOCHS lr=$LR input_scale_divisor=$INPUT_SCALE_DIVISOR adapter=none depth=none backbone_lr_scale=$BACKBONE_LR_SCALE head_lr_scale=$HEAD_LR_SCALE sequence_dropout=$SEQUENCE_DROPOUT skip_final_test=$SKIP_FINAL_TEST strict_checkpoint_load=1" >&2
echo "git_commit: $(git rev-parse HEAD)" >&2
echo "checkpoint_sha256: $(sha256sum ./checkpoints/labram-base.pth | awk '{print $1}')" >&2
echo "signal_stats: $(sha256sum ./signal_stats_train.json | awk '{print $1}')" >&2

OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 run_class_finetuning.py \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  --model labram_base_patch200_200 \
  --finetune ./checkpoints/labram-base.pth \
  --data_path "$DATA_DIR" \
  --dataset ISRUC \
  --input_scale_divisor "$INPUT_SCALE_DIVISOR" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr "$LR" \
  --requested_lr_string "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --layer_decay "$LAYER_DECAY" \
  --backbone_lr_scale "$BACKBONE_LR_SCALE" \
  --head_lr_scale "$HEAD_LR_SCALE" \
  --update_freq 1 \
  --warmup_epochs 1 \
  --epochs "$EPOCHS" \
  --isruc_sequence_dropout "$SEQUENCE_DROPOUT" \
  --labram_adapter_type none \
  --labram_adapter_depth_mode none \
  --disable_rel_pos_bias \
  --abs_pos_emb \
  --disable_qkv_bias \
  --save_ckpt_freq 1 \
  --seed "$SEED" \
  --strict_checkpoint_load \
  --no_auto_resume \
  $(if [[ "$SKIP_FINAL_TEST" == "1" ]]; then echo --skip_final_test; fi)
