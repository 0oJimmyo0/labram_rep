#!/bin/bash
# LaBraM-only PhysioNet-MI dense baseline.
# Locked development contract after the seed-42 LR sweep: LR=5e-4.
# Override explicitly only for a documented sensitivity experiment.
# The LMDB was produced by the EEGxPlore preprocessing contract, but this
# launcher never imports or executes the EEGxPlore CBraMod model path.
#SBATCH --job-name=labram_physio_dense
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
RUN_ID="${RUN_ID:-physio_mi_labram_dense_lr${LR}_s${SEED}_b${BATCH_SIZE}_e${EPOCHS}_scale${INPUT_SCALE_DIVISOR}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs/tensorboard/$RUN_ID}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing PhysioNet-MI LMDB directory: $DATA_DIR" >&2
  exit 2
fi
if [[ ! -f "$DATA_DIR/data.mdb" ]]; then
  echo "Missing PhysioNet-MI data.mdb under: $DATA_DIR" >&2
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
export DATA_DIR

python - <<'PY'
import sys
import torch
from utils import PHYSIONET_MI_LABRAM_CH, PhysioNetMILoader

root = __import__('os').environ.get('DATA_DIR')
if not root:
    raise SystemExit('DATA_DIR was not exported for the preflight')
train = PhysioNetMILoader(root, mode='train')
val = PhysioNetMILoader(root, mode='val')
test = PhysioNetMILoader(root, mode='test')
x, y = train[0]
assert tuple(x.shape) == (64, 4, 200), tuple(x.shape)
assert x.dtype == torch.float32, x.dtype
assert len(PHYSIONET_MI_LABRAM_CH) == 64
assert all(isinstance(name, str) for name in PHYSIONET_MI_LABRAM_CH)
print('python:', sys.executable)
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('gpu count:', torch.cuda.device_count())
print('physio preflight:', len(train), len(val), len(test), tuple(x.shape), y)
PY
nvidia-smi || true

echo "PhysioNet-MI LaBraM dense launch: seed=$SEED batch=$BATCH_SIZE workers=$NUM_WORKERS epochs=$EPOCHS lr=$LR wd=$WEIGHT_DECAY clip_grad=$CLIP_GRAD scale=$INPUT_SCALE_DIVISOR adapter=none backbone_mode=trainable" >&2
echo "git_commit: $(git rev-parse HEAD)" >&2
echo "git_dirty_files:" >&2
git status --short >&2
echo "checkpoint_sha256: $(sha256sum ./checkpoints/labram-base.pth | awk '{print $1}')" >&2
echo "data_mdb: $(stat -c '%s bytes %y' "$DATA_DIR/data.mdb")" >&2
echo "run_id: $RUN_ID" >&2

OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 run_class_finetuning.py \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  --model labram_base_patch200_200 \
  --finetune ./checkpoints/labram-base.pth \
  --data_path "$DATA_DIR" \
  --dataset PhysioNet-MI \
  --input_scale_divisor "$INPUT_SCALE_DIVISOR" \
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
  --smoothing "$SMOOTHING" \
  --labram_adapter_type none \
  --labram_adapter_depth_mode none \
  --disable_rel_pos_bias \
  --abs_pos_emb \
  --disable_qkv_bias \
  --save_ckpt_freq 1 \
  --seed "$SEED" \
  --strict_checkpoint_load \
  --no_auto_resume
