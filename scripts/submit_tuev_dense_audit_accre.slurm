#!/bin/bash
# LaBraM-only TUEV faithful dense baseline.
# The source TUEV contract stores raw [16,1000] bipolar events. The loader
# reshapes them to [16,5,200] and the engine applies exactly one /100 scale.
# This launcher supersedes the historical submit_tuev_finetune_accre.slurm;
# it adds provenance/geometry preflight, strict checkpoint loading, sorted
# file order, explicit run IDs, and disables accidental auto-resume.
#
# Faithful baseline defaults preserve the historical TUEV optimizer/model
# recipe while using a practical fixed contract for this much larger
# serialized split: batch=64, epochs=25, lr=5e-4, warmup=5,
# layer_decay=.65, wd=.05, drop_path=.1, no gradient clipping, smoothing=.1,
# and the fixed bipolar 16-channel montage. NUM_WORKERS=4 is an operational
# choice; it does not change the ordered split or model computation.
#
# Override EPOCHS=1 for a smoke/preflight run. Do not use its metrics as a
# scientific baseline.
#SBATCH --job-name=labram_tuev_dense
#SBATCH --account=neurogroup_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/logs/out/%x_%j.out
#SBATCH --error=/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/logs/err/%x_%j.err

set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM"
ENV_DIR="/data/neurogroup/mingyangjiang/venvs/labram-accre"
DATA_DIR="${DATA_DIR:-/data/neurogroup/mingyangjiang/data/TUEV}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-25}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LAYER_DECAY="${LAYER_DECAY:-0.65}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
DROP_PATH="${DROP_PATH:-0.1}"
SMOOTHING="${SMOOTHING:-0.1}"
INPUT_SCALE_DIVISOR="${INPUT_SCALE_DIVISOR:-100.0}"
SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-5}"
RUN_ID="${RUN_ID:-tuev_labram_dense_faithful_lr5e4_s${SEED}_b${BATCH_SIZE}_e${EPOCHS}_scale100}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs/tensorboard/$RUN_ID}"

if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Refusing to reuse nonempty output directory: $OUTPUT_DIR" >&2
  exit 2
fi
for split in processed_train processed_eval processed_test; do
  if [[ ! -d "$DATA_DIR/$split" && ! -d "$DATA_DIR/processed/$split" ]]; then
    echo "Missing TUEV split $split under $DATA_DIR or $DATA_DIR/processed" >&2
    exit 2
  fi
done

mkdir -p "$REPO_DIR/logs/out" "$REPO_DIR/logs/err" "$LOG_DIR" "$OUTPUT_DIR"
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
module --force purge >/dev/null 2>&1
module load gentoo/2023 gcc/12.3 python/3.11.5 scipy-stack/2026a >/dev/null 2>&1
export PYTHONNOUSERSITE=1
source "$ENV_DIR/bin/activate"
cd "$REPO_DIR"
export DATA_DIR

python - <<'PY'
import os
import sys
import torch
from modeling_finetune import labram_base_patch200_200
from utils import TUEVLoader, TUEV_BIPOLAR_16_CH, prepare_TUEV_dataset

root = os.environ['DATA_DIR']
train, test, val = prepare_TUEV_dataset(root)
x, y = train[0]
assert tuple(x.shape) == TUEVLoader.expected_token_shape, tuple(x.shape)
assert x.dtype == torch.float32
assert int(y) in range(6)
assert len(TUEV_BIPOLAR_16_CH) == 16
model = labram_base_patch200_200(
    num_classes=6,
    qkv_bias=False,
    use_abs_pos_emb=True,
    use_rel_pos_bias=False,
    init_values=0.1,
    adapter_type='none',
)
with torch.no_grad():
    features = model.forward_features(
        x.unsqueeze(0), input_chans=None, return_all_tokens=True
    )
assert tuple(features.shape) == (1, 1 + 16 * 5, 200), tuple(features.shape)
print('python:', sys.executable)
print('torch:', torch.__version__)
print('TUEV dense preflight:', {
    'train': len(train), 'val': len(val), 'test': len(test),
    'sample_shape': tuple(x.shape), 'label': int(y),
    'token_grid': tuple(features.shape), 'channels': TUEV_BIPOLAR_16_CH,
    'input_position_policy': 'sequential_pos_embed_slots_1_to_16',
    'input_scale_divisor': 100.0,
})
PY
nvidia-smi || true

echo "TUEV LaBraM dense launch: seed=$SEED batch=$BATCH_SIZE epochs=$EPOCHS lr=$LR warmup=$WARMUP_EPOCHS workers=$NUM_WORKERS scale=$INPUT_SCALE_DIVISOR" >&2
echo "git_commit: $(git rev-parse HEAD)" >&2
git status --short >&2
echo "checkpoint_sha256: $(sha256sum ./checkpoints/labram-base.pth | awk '{print $1}')" >&2
echo "run_id: $RUN_ID" >&2

OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 run_class_finetuning.py \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  --model labram_base_patch200_200 \
  --finetune ./checkpoints/labram-base.pth \
  --data_path "$DATA_DIR" \
  --dataset TUEV \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr "$LR" \
  --requested_lr_string "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --layer_decay "$LAYER_DECAY" \
  --update_freq 1 \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --epochs "$EPOCHS" \
  --drop_path "$DROP_PATH" \
  --smoothing "$SMOOTHING" \
  --input_scale_divisor "$INPUT_SCALE_DIVISOR" \
  --disable_rel_pos_bias \
  --abs_pos_emb \
  --disable_qkv_bias \
  --save_ckpt_freq "$SAVE_CKPT_FREQ" \
  --seed "$SEED" \
  --strict_checkpoint_load \
  --no_auto_resume
