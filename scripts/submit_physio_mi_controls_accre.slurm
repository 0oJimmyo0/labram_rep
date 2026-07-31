#!/bin/bash
# LaBraM-only PhysioNet-MI independent controls.
# METHOD=generic: frozen generic bottleneck/token-wise residual + head.
# METHOD=lora: independent frozen-backbone LoRA qkv-r8 + head.
# METHOD=upper2: trainable upper-2 blocks + head.
# These controls never enable the native channel/patch adapter.
#SBATCH --job-name=labram_physio_control
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
DATA_DIR="${DATA_DIR:-/data/neurogroup/mingyangjiang/data/PHYSIO_MI}"
METHOD="${METHOD:-generic}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LAYER_DECAY="${LAYER_DECAY:-0.65}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
INPUT_SCALE_DIVISOR="${INPUT_SCALE_DIVISOR:-100.0}"
SMOOTHING="${SMOOTHING:-0.1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
DROP_PATH="${DROP_PATH:-0.1}"
ADAPTER_BOTTLENECK="${ADAPTER_BOTTLENECK:-64}"
ADAPTER_HEADS="${ADAPTER_HEADS:-4}"
ADAPTER_DROPOUT="${ADAPTER_DROPOUT:-0.0}"
ADAPTER_INIT_ALPHA="${ADAPTER_INIT_ALPHA:-0.01}"
ADAPTER_LR_SCALE="${ADAPTER_LR_SCALE:-1.0}"
ADAPTER_ALPHA_LR_SCALE="${ADAPTER_ALPHA_LR_SCALE:-1.0}"
ADAPTER_SEED="${ADAPTER_SEED:-12345}"
UPPER_K="${UPPER_K:-2}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16.0}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET="${LORA_TARGET:-qkv}"

case "$METHOD" in
  generic)
    ADAPTER_TYPE=generic
    ADAPTER_VARIANT=full
    BACKBONE_MODE=frozen
    BACKBONE_LR_SCALE=0.0
    FROZEN_BACKBONE_EVAL_MODE=1
    USE_TOKEN_MLP=1
    ;;
  lora)
    ADAPTER_TYPE=none
    ADAPTER_VARIANT=full
    BACKBONE_MODE=lora
    BACKBONE_LR_SCALE=0.0
    FROZEN_BACKBONE_EVAL_MODE=0
    USE_TOKEN_MLP=0
    ;;
  upper2)
    ADAPTER_TYPE=none
    ADAPTER_VARIANT=full
    BACKBONE_MODE=upper_k
    BACKBONE_LR_SCALE=1.0
    FROZEN_BACKBONE_EVAL_MODE=0
    USE_TOKEN_MLP=0
    ;;
  *) echo "ERROR: METHOD must be generic, lora, or upper2; got $METHOD" >&2; exit 2 ;;
esac

RUN_ID="${RUN_ID:-physio_mi_${METHOD}_locked_lr5e4_s${SEED}_b${BATCH_SIZE}_e${EPOCHS}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs/tensorboard/$RUN_ID}"

if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Refusing to reuse nonempty output directory: $OUTPUT_DIR" >&2
  exit 2
fi
if [[ ! -f "$DATA_DIR/data.mdb" ]]; then
  echo "Missing PhysioNet-MI data.mdb under: $DATA_DIR" >&2
  exit 2
fi

mkdir -p "$REPO_DIR/logs/out" "$REPO_DIR/logs/err" "$LOG_DIR" "$OUTPUT_DIR"
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
module --force purge >/dev/null 2>&1
module load gentoo/2023 gcc/12.3 python/3.11.5 scipy-stack/2026a >/dev/null 2>&1
export PYTHONNOUSERSITE=1
source "$ENV_DIR/bin/activate"
cd "$REPO_DIR"
export DATA_DIR

python - <<'PY'
import sys
import torch
from utils import PHYSIONET_MI_LABRAM_CH, PhysioNetMILoader

root = __import__('os').environ['DATA_DIR']
train = PhysioNetMILoader(root, mode='train')
x, y = train[0]
assert tuple(x.shape) == (64, 4, 200), tuple(x.shape)
assert x.dtype == torch.float32
assert len(PHYSIONET_MI_LABRAM_CH) == 64
print('python:', sys.executable)
print('torch:', torch.__version__)
print('physio control preflight:', tuple(x.shape), y)
PY
nvidia-smi || true

echo "PhysioNet-MI LaBraM control launch: method=$METHOD seed=$SEED mode=$BACKBONE_MODE adapter=$ADAPTER_TYPE batch=$BATCH_SIZE epochs=$EPOCHS lr=$LR scale=$INPUT_SCALE_DIVISOR" >&2
echo "git_commit: $(git rev-parse HEAD)" >&2
git status --short >&2
echo "checkpoint_sha256: $(sha256sum ./checkpoints/labram-base.pth | awk '{print $1}')" >&2
echo "run_id: $RUN_ID" >&2

PYTHON_ARGS=(
  --labram_adapter_type "$ADAPTER_TYPE"
  --labram_adapter_variant "$ADAPTER_VARIANT"
  --labram_adapter_bottleneck "$ADAPTER_BOTTLENECK"
  --labram_adapter_heads "$ADAPTER_HEADS"
  --labram_adapter_dropout "$ADAPTER_DROPOUT"
  --labram_adapter_init_alpha "$ADAPTER_INIT_ALPHA"
  --labram_adapter_lr_scale "$ADAPTER_LR_SCALE"
  --labram_adapter_alpha_lr_scale "$ADAPTER_ALPHA_LR_SCALE"
  --labram_adapter_seed "$ADAPTER_SEED"
  --labram_backbone_mode "$BACKBONE_MODE"
  --labram_upper_k "$UPPER_K"
  --labram_lora_rank "$LORA_RANK"
  --labram_lora_alpha "$LORA_ALPHA"
  --labram_lora_dropout "$LORA_DROPOUT"
  --labram_lora_target "$LORA_TARGET"
  --input_scale_divisor "$INPUT_SCALE_DIVISOR"
  --strict_checkpoint_load
  --no_auto_resume
)
if [[ "$USE_TOKEN_MLP" == "1" ]]; then
  PYTHON_ARGS+=(--labram_adapter_use_token_mlp)
fi
if [[ "$FROZEN_BACKBONE_EVAL_MODE" == "1" ]]; then
  PYTHON_ARGS+=(--frozen_backbone_eval_mode)
fi

OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 run_class_finetuning.py \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  --model labram_base_patch200_200 \
  --finetune ./checkpoints/labram-base.pth \
  --data_path "$DATA_DIR" \
  --dataset PhysioNet-MI \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr "$LR" \
  --requested_lr_string "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --layer_decay "$LAYER_DECAY" \
  --backbone_lr_scale "$BACKBONE_LR_SCALE" \
  --head_lr_scale 1.0 \
  --clip_grad "$CLIP_GRAD" \
  --update_freq 1 \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --epochs "$EPOCHS" \
  --drop_path "$DROP_PATH" \
  --smoothing "$SMOOTHING" \
  --disable_rel_pos_bias \
  --abs_pos_emb \
  --disable_qkv_bias \
  --save_ckpt_freq 1 \
  --seed "$SEED" \
  "${PYTHON_ARGS[@]}"
