#!/bin/bash
# LaBraM-only PhysioNet-MI native-axis adaptation.
# This launcher inherits the locked dense PhysioNet-MI contract and changes
# only the trainability/adaptor condition. Native low-rank adaptation is
# separate from the generic LoRA control.
#SBATCH --job-name=labram_physio_native
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
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LAYER_DECAY="${LAYER_DECAY:-0.65}"
BACKBONE_LR_SCALE="${BACKBONE_LR_SCALE:-1.0}"
HEAD_LR_SCALE="${HEAD_LR_SCALE:-1.0}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
INPUT_SCALE_DIVISOR="${INPUT_SCALE_DIVISOR:-100.0}"
SMOOTHING="${SMOOTHING:-0.1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
DROP_PATH="${DROP_PATH:-0.1}"
ADAPTER_TYPE="${ADAPTER_TYPE:-channel_patch}"
ADAPTER_VARIANT="${ADAPTER_VARIANT:-low_rank}"
ADAPTER_BOTTLENECK="${ADAPTER_BOTTLENECK:-64}"
ADAPTER_HEADS="${ADAPTER_HEADS:-4}"
ADAPTER_DROPOUT="${ADAPTER_DROPOUT:-0.0}"
ADAPTER_INIT_ALPHA="${ADAPTER_INIT_ALPHA:-0.01}"
ADAPTER_GAMMA="${ADAPTER_GAMMA:-1.0}"
ADAPTER_LR_SCALE="${ADAPTER_LR_SCALE:-1.0}"
ADAPTER_ALPHA_LR_SCALE="${ADAPTER_ALPHA_LR_SCALE:-1.0}"
ADAPTER_WEIGHT_DECAY="${ADAPTER_WEIGHT_DECAY:-}"
ADAPTER_SEED="${ADAPTER_SEED:-12345}"
ADAPTER_DEPTH_MODE="${ADAPTER_DEPTH_MODE:-none}"
ADAPTER_DEPTH_K="${ADAPTER_DEPTH_K:-4}"
BACKBONE_MODE="${BACKBONE_MODE:-trainable}"
FROZEN_BACKBONE_EVAL_MODE="${FROZEN_BACKBONE_EVAL_MODE:-0}"
RUN_ID="${RUN_ID:-physio_mi_labram_${ADAPTER_TYPE}_${BACKBONE_MODE}_${ADAPTER_VARIANT}_lr${LR}_s${SEED}_b${BATCH_SIZE}_e${EPOCHS}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs/tensorboard/$RUN_ID}"

case "$ADAPTER_TYPE" in
  none|channel|patch|channel_patch) ;;
  *) echo "ERROR: launcher requires none, channel, patch, or channel_patch; got $ADAPTER_TYPE" >&2; exit 2 ;;
esac
if [[ "$ADAPTER_TYPE" != "none" ]]; then
  case "$ADAPTER_VARIANT" in
    low_rank) ;;
    *) echo "ERROR: PhysioNet native primary contract requires ADAPTER_VARIANT=low_rank; got $ADAPTER_VARIANT" >&2; exit 2 ;;
  esac
fi
case "$BACKBONE_MODE" in
  trainable|frozen) ;;
  *) echo "ERROR: native launcher supports BACKBONE_MODE=trainable or frozen; got $BACKBONE_MODE" >&2; exit 2 ;;
esac
if [[ "$FROZEN_BACKBONE_EVAL_MODE" == "1" && "$BACKBONE_MODE" != "frozen" ]]; then
  echo "ERROR: FROZEN_BACKBONE_EVAL_MODE=1 requires BACKBONE_MODE=frozen" >&2
  exit 2
fi
if [[ "$BACKBONE_MODE" == "frozen" && "$BACKBONE_LR_SCALE" != "0.0" ]]; then
  echo "ERROR: frozen native runs require BACKBONE_LR_SCALE=0.0" >&2
  exit 2
fi
if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
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
export DATA_DIR ADAPTER_TYPE

python - <<'PY'
import sys
import torch
from modeling_finetune import labram_base_patch200_200
from utils import PHYSIONET_MI_LABRAM_CH, PhysioNetMILoader, get_input_chans

root = __import__('os').environ['DATA_DIR']
train = PhysioNetMILoader(root, mode='train')
x, y = train[0]
assert tuple(x.shape) == (64, 4, 200), tuple(x.shape)
assert x.dtype == torch.float32
assert len(PHYSIONET_MI_LABRAM_CH) == 64

adapter_type = __import__('os').environ.get('ADAPTER_TYPE', 'channel_patch')
model = labram_base_patch200_200(
    num_classes=4,
    qkv_bias=False,
    use_abs_pos_emb=True,
    use_rel_pos_bias=False,
    init_values=0.1,
    adapter_type=adapter_type,
    adapter_variant='low_rank',
    adapter_bottleneck=64,
    adapter_num_heads=4,
    adapter_init_alpha=0.01,
)
with torch.no_grad():
    model.forward_features(x.unsqueeze(0), input_chans=get_input_chans(PHYSIONET_MI_LABRAM_CH))
diag = model.get_adapter_diagnostics()
if adapter_type != 'none':
    assert diag['adapter_channel_count'] == 64
    assert diag['adapter_patch_count'] == 4
    if adapter_type in ('channel', 'channel_patch'):
        assert diag['channel_spatial_interactions_active'] == 1
    else:
        assert diag['channel_spatial_interactions_active'] == 0
    if adapter_type in ('patch', 'channel_patch'):
        assert diag['patch_temporal_interactions_active'] == 1
    else:
        assert diag['patch_temporal_interactions_active'] == 0
print('python:', sys.executable)
print('torch:', torch.__version__)
print('physio native geometry preflight:', tuple(x.shape), 'adapter=', adapter_type, diag)
PY
nvidia-smi || true

echo "PhysioNet-MI LaBraM native launch: seed=$SEED adapter=$ADAPTER_TYPE variant=$ADAPTER_VARIANT backbone_mode=$BACKBONE_MODE batch=$BATCH_SIZE epochs=$EPOCHS lr=$LR adapter_lr_scale=$ADAPTER_LR_SCALE alpha_lr_scale=$ADAPTER_ALPHA_LR_SCALE scale=$INPUT_SCALE_DIVISOR" >&2
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
  --labram_adapter_gamma "$ADAPTER_GAMMA"
  --labram_adapter_lr_scale "$ADAPTER_LR_SCALE"
  --labram_adapter_alpha_lr_scale "$ADAPTER_ALPHA_LR_SCALE"
  --labram_adapter_seed "$ADAPTER_SEED"
  --labram_adapter_depth_mode "$ADAPTER_DEPTH_MODE"
  --labram_adapter_depth_k "$ADAPTER_DEPTH_K"
  --labram_backbone_mode "$BACKBONE_MODE"
  --input_scale_divisor "$INPUT_SCALE_DIVISOR"
  --strict_checkpoint_load
  --no_auto_resume
)
if [[ "$FROZEN_BACKBONE_EVAL_MODE" == "1" ]]; then
  PYTHON_ARGS+=(--frozen_backbone_eval_mode)
fi
if [[ -n "$ADAPTER_WEIGHT_DECAY" ]]; then
  PYTHON_ARGS+=(--labram_adapter_weight_decay "$ADAPTER_WEIGHT_DECAY")
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
  --head_lr_scale "$HEAD_LR_SCALE" \
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
