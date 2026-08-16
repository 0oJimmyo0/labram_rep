#!/bin/bash
#SBATCH --job-name=labram_isruc_native
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
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LAYER_DECAY="${LAYER_DECAY:-0.65}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
SEQUENCE_DROPOUT="${SEQUENCE_DROPOUT:-0.1}"
BACKBONE_LR_SCALE="${BACKBONE_LR_SCALE:-1.0}"
HEAD_LR_SCALE="${HEAD_LR_SCALE:-1.0}"
HEAD_WEIGHT_DECAY="${HEAD_WEIGHT_DECAY:-}"
ADAPTER_TYPE="${ADAPTER_TYPE:-none}"
ADAPTER_VARIANT="${ADAPTER_VARIANT:-full}"
ADAPTER_BOTTLENECK="${ADAPTER_BOTTLENECK:-64}"
ADAPTER_HEADS="${ADAPTER_HEADS:-4}"
ADAPTER_DROPOUT="${ADAPTER_DROPOUT:-0.0}"
ADAPTER_PATCH_OUTPUT_DROPOUT="${ADAPTER_PATCH_OUTPUT_DROPOUT:-0.0}"
ADAPTER_INIT_ALPHA="${ADAPTER_INIT_ALPHA:-0.01}"
ADAPTER_GAMMA="${ADAPTER_GAMMA:-1.0}"
ADAPTER_LR_SCALE="${ADAPTER_LR_SCALE:-0.1}"
ADAPTER_ALPHA_LR_SCALE="${ADAPTER_ALPHA_LR_SCALE:-0.1}"
ADAPTER_WEIGHT_DECAY="${ADAPTER_WEIGHT_DECAY:-}"
ADAPTER_SEED="${ADAPTER_SEED:-12345}"
ADAPTER_DEPTH_MODE="${ADAPTER_DEPTH_MODE:-none}"
ADAPTER_DEPTH_K="${ADAPTER_DEPTH_K:-4}"
BACKBONE_MODE="${BACKBONE_MODE:-trainable}"
UPPER_K="${UPPER_K:-2}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET="${LORA_TARGET:-qkv}"
FROZEN_BACKBONE_EVAL_MODE="${FROZEN_BACKBONE_EVAL_MODE:-0}"
EXPERIMENT_METHOD="${EXPERIMENT_METHOD:-}"
TARGET_ADAPTER_PARAMS="${TARGET_ADAPTER_PARAMS:-}"
SKIP_FINAL_TEST="${SKIP_FINAL_TEST:-0}"
RUN_ID="${RUN_ID:-isruc_labram_${ADAPTER_TYPE}_current_s${SEED}_b${BATCH_SIZE}_lr${LR}_e${EPOCHS}_test}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs/tensorboard/$RUN_ID}"

case "$ADAPTER_TYPE" in
  none|generic|channel|patch|channel_patch) ;;
  *) echo "ERROR: unsupported ADAPTER_TYPE=$ADAPTER_TYPE" >&2; exit 1 ;;
esac
case "$BACKBONE_MODE" in
  trainable|frozen|upper_k|lora) ;;
  *) echo "ERROR: BACKBONE_MODE must be trainable, frozen, upper_k, or lora" >&2; exit 1 ;;
esac
case "$FROZEN_BACKBONE_EVAL_MODE" in
  0|1) ;;
  *) echo "ERROR: FROZEN_BACKBONE_EVAL_MODE must be 0 or 1" >&2; exit 1 ;;
esac

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
import sys
import torch
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("ISRUC run requires a CUDA GPU")
PY
nvidia-smi

echo "ISRUC native launch: run=$RUN_ID seed=$SEED adapter=$ADAPTER_TYPE backbone_mode=$BACKBONE_MODE upper_k=$UPPER_K lora_rank=$LORA_RANK lora_target=$LORA_TARGET batch=$BATCH_SIZE epochs=$EPOCHS lr=$LR adapter_lr_scale=$ADAPTER_LR_SCALE alpha_lr_scale=$ADAPTER_ALPHA_LR_SCALE skip_final_test=$SKIP_FINAL_TEST" >&2
echo "git_commit: $(git rev-parse HEAD)" >&2
echo "checkpoint_sha256: $(sha256sum ./checkpoints/labram-base.pth | awk '{print $1}')" >&2

PYTHON_ARGS=(
  --labram_adapter_type "$ADAPTER_TYPE"
  --labram_adapter_variant "$ADAPTER_VARIANT"
  --labram_adapter_bottleneck "$ADAPTER_BOTTLENECK"
  --labram_adapter_heads "$ADAPTER_HEADS"
  --labram_adapter_dropout "$ADAPTER_DROPOUT"
  --labram_adapter_patch_output_dropout "$ADAPTER_PATCH_OUTPUT_DROPOUT"
  --labram_adapter_init_alpha "$ADAPTER_INIT_ALPHA"
  --labram_adapter_gamma "$ADAPTER_GAMMA"
  --labram_adapter_lr_scale "$ADAPTER_LR_SCALE"
  --labram_adapter_alpha_lr_scale "$ADAPTER_ALPHA_LR_SCALE"
  --labram_adapter_seed "$ADAPTER_SEED"
  --labram_adapter_depth_mode "$ADAPTER_DEPTH_MODE"
  --labram_adapter_depth_k "$ADAPTER_DEPTH_K"
  --labram_backbone_mode "$BACKBONE_MODE"
  --labram_upper_k "$UPPER_K"
  --labram_lora_rank "$LORA_RANK"
  --labram_lora_alpha "$LORA_ALPHA"
  --labram_lora_dropout "$LORA_DROPOUT"
  --labram_lora_target "$LORA_TARGET"
  --experiment_method "$EXPERIMENT_METHOD"
  --isruc_sequence_dropout "$SEQUENCE_DROPOUT"
  --input_scale_divisor 1.0
  --strict_checkpoint_load
  --no_auto_resume
)
if [[ "$FROZEN_BACKBONE_EVAL_MODE" == "1" ]]; then
  PYTHON_ARGS+=(--frozen_backbone_eval_mode)
fi
if [[ "$SKIP_FINAL_TEST" == "1" ]]; then
  PYTHON_ARGS+=(--skip_final_test)
fi
if [[ -n "$ADAPTER_WEIGHT_DECAY" ]]; then
  PYTHON_ARGS+=(--labram_adapter_weight_decay "$ADAPTER_WEIGHT_DECAY")
fi
if [[ -n "$HEAD_WEIGHT_DECAY" ]]; then
  PYTHON_ARGS+=(--head_weight_decay "$HEAD_WEIGHT_DECAY")
fi
if [[ -n "$TARGET_ADAPTER_PARAMS" ]]; then
  PYTHON_ARGS+=(--target_adapter_params "$TARGET_ADAPTER_PARAMS")
fi

OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 run_class_finetuning.py \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  --model labram_base_patch200_200 \
  --finetune ./checkpoints/labram-base.pth \
  --data_path "$DATA_DIR" \
  --dataset ISRUC \
  --input_scale_divisor 1.0 \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr "$LR" \
  --requested_lr_string "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --layer_decay "$LAYER_DECAY" \
  --backbone_lr_scale "$BACKBONE_LR_SCALE" \
  --head_lr_scale "$HEAD_LR_SCALE" \
  --update_freq 1 \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --epochs "$EPOCHS" \
  --disable_rel_pos_bias \
  --abs_pos_emb \
  --disable_qkv_bias \
  --save_ckpt_freq 1 \
  --seed "$SEED" \
  "${PYTHON_ARGS[@]}"
