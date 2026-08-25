#!/bin/bash
# LaBraM-only TUEV registry launcher.
#
# This launcher is deliberately separate from the current LR sweep launcher:
# the sweep selects LR on seed 42 first. After LR is locked, this script runs
# the same 15-epoch TUEV contract for dense, frozen, native-axis, and generic
# comparator conditions. It never mixes CBraMod/EEGxPlore code.
#
# METHOD values:
#   full_dense, frozen_dense,
#   frozen_channel, frozen_patch, frozen_channel_patch,
#   native_channel, native_patch, native_channel_patch,
#   generic, axis_blind, lora, upper2, native_depth.
#
# Native channel/patch adaptation uses the LaBraM-native low-rank branch. This
# is distinct from METHOD=lora, which is the independent generic LoRA control.
# No test metric is used during training; final_test.json is produced only
# after validation-kappa and validation-BA checkpoint selection.
#
#SBATCH --job-name=labram_tuev_registry
#SBATCH --account=neurogroup_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/logs/out/%x_%j.out
#SBATCH --error=/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/logs/err/%x_%j.err

set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM"
ENV_DIR="/data/neurogroup/mingyangjiang/venvs/labram-accre"
DATA_DIR="${DATA_DIR:-/data/neurogroup/mingyangjiang/data/TUEV}"
METHOD="${METHOD:-full_dense}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-15}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LAYER_DECAY="${LAYER_DECAY:-0.65}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
DROP_PATH="${DROP_PATH:-0.2}"
SMOOTHING="${SMOOTHING:-0.1}"
INPUT_SCALE_DIVISOR="${INPUT_SCALE_DIVISOR:-100.0}"
SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-5}"
ADAPTER_BOTTLENECK="${ADAPTER_BOTTLENECK:-64}"
ADAPTER_HEADS="${ADAPTER_HEADS:-4}"
ADAPTER_DROPOUT="${ADAPTER_DROPOUT:-0.0}"
ADAPTER_INIT_ALPHA="${ADAPTER_INIT_ALPHA:-0.01}"
ADAPTER_GAMMA="${ADAPTER_GAMMA:-1.0}"
ADAPTER_ZERO_INIT_OUTPUT="${ADAPTER_ZERO_INIT_OUTPUT:-0}"
ADAPTER_LR_SCALE="${ADAPTER_LR_SCALE:-1.0}"
ADAPTER_ALPHA_LR_SCALE="${ADAPTER_ALPHA_LR_SCALE:-1.0}"
ADAPTER_WEIGHT_DECAY="${ADAPTER_WEIGHT_DECAY:-}"
ADAPTER_SEED="${ADAPTER_SEED:-12345}"
ADAPTER_DEPTH_K="${ADAPTER_DEPTH_K:-4}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16.0}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET="${LORA_TARGET:-qkv}"
TARGET_ADAPTER_PARAMS="${TARGET_ADAPTER_PARAMS:-}"

ADAPTER_TYPE=none
ADAPTER_VARIANT=full
ADAPTER_OPERATOR=attention
BACKBONE_MODE=trainable
BACKBONE_LR_SCALE=1.0
FROZEN_BACKBONE_EVAL_MODE=0
ADAPTER_DEPTH_MODE=none

case "$METHOD" in
  full_dense) ;;
  frozen_dense)
    BACKBONE_MODE=frozen; BACKBONE_LR_SCALE=0.0; FROZEN_BACKBONE_EVAL_MODE=1
    ;;
  frozen_channel|frozen_patch|frozen_channel_patch)
    ADAPTER_TYPE="${METHOD#frozen_}"; ADAPTER_VARIANT=low_rank
    BACKBONE_MODE=frozen; BACKBONE_LR_SCALE=0.0; FROZEN_BACKBONE_EVAL_MODE=1
    ;;
  native_channel|native_patch|native_channel_patch)
    ADAPTER_TYPE="${METHOD#native_}"; ADAPTER_VARIANT=low_rank
    ;;
  native_depth)
    ADAPTER_TYPE=channel_patch; ADAPTER_VARIANT=low_rank; ADAPTER_DEPTH_MODE=lastk_delta
    ;;
  generic)
    ADAPTER_TYPE=generic; BACKBONE_MODE=frozen; BACKBONE_LR_SCALE=0.0
    FROZEN_BACKBONE_EVAL_MODE=1
    ;;
  axis_blind)
    if [[ -z "$TARGET_ADAPTER_PARAMS" ]]; then
      echo "ERROR: axis_blind requires TARGET_ADAPTER_PARAMS from the native run." >&2
      exit 2
    fi
    ADAPTER_TYPE=generic; BACKBONE_MODE=frozen; BACKBONE_LR_SCALE=0.0
    FROZEN_BACKBONE_EVAL_MODE=1
    ;;
  axis_decomposed_mlp)
    if [[ -z "$TARGET_ADAPTER_PARAMS" ]]; then
      echo "ERROR: axis_decomposed_mlp requires TARGET_ADAPTER_PARAMS from the native run." >&2
      exit 2
    fi
    ADAPTER_TYPE=channel_patch; ADAPTER_VARIANT=low_rank; ADAPTER_OPERATOR=mlp
    BACKBONE_MODE=frozen; BACKBONE_LR_SCALE=0.0
    FROZEN_BACKBONE_EVAL_MODE=1
    ;;
  lora)
    BACKBONE_MODE=lora; BACKBONE_LR_SCALE=0.0
    ;;
  upper2)
    BACKBONE_MODE=upper_k; BACKBONE_LR_SCALE=1.0
    ;;
  *)
    echo "ERROR: unsupported TUEV registry METHOD=$METHOD" >&2
    exit 2
    ;;
esac

if [[ "$BACKBONE_MODE" == frozen && "$BACKBONE_LR_SCALE" != 0.0 ]]; then
  echo "ERROR: frozen TUEV method must use BACKBONE_LR_SCALE=0.0" >&2
  exit 2
fi
if [[ "$FROZEN_BACKBONE_EVAL_MODE" == 1 && "$BACKBONE_MODE" != frozen ]]; then
  echo "ERROR: frozen eval mode requires BACKBONE_MODE=frozen" >&2
  exit 2
fi

RUN_ID="${RUN_ID:-tuev_labram_${METHOD}_lr${LR}_s${SEED}_b${BATCH_SIZE}_e${EPOCHS}_scale100}"
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
export DATA_DIR METHOD ADAPTER_TYPE ADAPTER_VARIANT ADAPTER_OPERATOR BACKBONE_MODE ADAPTER_DEPTH_MODE \
  ADAPTER_BOTTLENECK ADAPTER_HEADS ADAPTER_DROPOUT ADAPTER_INIT_ALPHA ADAPTER_GAMMA \
  ADAPTER_DEPTH_K LORA_RANK LORA_ALPHA LORA_DROPOUT LORA_TARGET
export TARGET_ADAPTER_PARAMS

python - <<'PY'
import os
import sys
import torch
import modeling_finetune
from modeling_finetune import labram_base_patch200_200
from utils import TUEVLoader, TUEV_BIPOLAR_16_CH, prepare_TUEV_dataset

root = os.environ['DATA_DIR']
method = os.environ['METHOD']
adapter_type = os.environ['ADAPTER_TYPE']
adapter_variant = os.environ['ADAPTER_VARIANT']
adapter_operator = os.environ['ADAPTER_OPERATOR']
backbone_mode = os.environ['BACKBONE_MODE']
depth_mode = os.environ['ADAPTER_DEPTH_MODE']
train, test, val = prepare_TUEV_dataset(root)
x, y = train[0]
assert tuple(x.shape) == TUEVLoader.expected_token_shape, tuple(x.shape)
assert x.dtype == torch.float32
assert int(y) in range(6)
assert len(TUEV_BIPOLAR_16_CH) == 16

model = labram_base_patch200_200(
    num_classes=6, qkv_bias=False, use_abs_pos_emb=True,
    use_rel_pos_bias=False, init_values=0.1,
    adapter_type=adapter_type, adapter_variant=adapter_variant,
    adapter_operator=adapter_operator,
    adapter_bottleneck=int(os.environ['ADAPTER_BOTTLENECK']),
    adapter_num_heads=int(os.environ['ADAPTER_HEADS']),
    adapter_dropout=float(os.environ['ADAPTER_DROPOUT']),
    adapter_init_alpha=float(os.environ['ADAPTER_INIT_ALPHA']),
    adapter_gamma=float(os.environ['ADAPTER_GAMMA']),
    adapter_depth_mode=depth_mode,
    adapter_depth_k=int(os.environ['ADAPTER_DEPTH_K']),
)
if backbone_mode == 'lora':
    modeling_finetune.inject_lora(
        model, rank=int(os.environ['LORA_RANK']),
        alpha=float(os.environ['LORA_ALPHA']),
        dropout=float(os.environ['LORA_DROPOUT']),
        target=os.environ['LORA_TARGET'])

with torch.no_grad():
    features = model.forward_features(x.unsqueeze(0), input_chans=None, return_all_tokens=True)
assert tuple(features.shape) == (1, 81, 200), tuple(features.shape)
diag = model.get_adapter_diagnostics()
if adapter_type != 'none':
    assert diag['adapter_channel_count'] == 16, diag
    assert diag['adapter_patch_count'] == 5, diag
    if adapter_operator == 'attention':
        if adapter_type in ('channel', 'channel_patch'):
            assert diag['channel_spatial_interactions_active'] == 1, diag
        if adapter_type in ('patch', 'channel_patch'):
            assert diag['patch_temporal_interactions_active'] == 1, diag
    else:
        if adapter_type in ('channel', 'channel_patch'):
            assert diag['channel_axis_branch_active'] == 1, diag
        if adapter_type in ('patch', 'channel_patch'):
            assert diag['patch_axis_branch_active'] == 1, diag
if depth_mode != 'none':
    assert adapter_type == 'channel_patch'
if method == 'axis_blind':
    target = int(os.environ['TARGET_ADAPTER_PARAMS'])
    actual = sum(p.numel() for p in model.native_axis_adapter.parameters())
    relative_error = abs(actual - target) / max(1, target)
    assert relative_error <= 0.05, (actual, target, relative_error)
    # Axis-blind is a capacity-matched token MLP control.  Make the
    # exclusion of native channel/patch interactions executable rather than
    # relying on the adapter_type string alone.
    assert not hasattr(model.native_axis_adapter, 'channel_attn')
    assert not hasattr(model.native_axis_adapter, 'patch_attn')
    assert hasattr(model.native_axis_adapter, 'token_mlp')
if method == 'axis_decomposed_mlp':
    target = int(os.environ['TARGET_ADAPTER_PARAMS'])
    actual = sum(p.numel() for p in model.native_axis_adapter.parameters())
    relative_error = abs(actual - target) / max(1, target)
    assert relative_error <= 0.05, (actual, target, relative_error)
    assert adapter_type == 'channel_patch'
    assert adapter_operator == 'mlp'
    assert not hasattr(model.native_axis_adapter, 'channel_attn')
    assert not hasattr(model.native_axis_adapter, 'patch_attn')
    assert hasattr(model.native_axis_adapter, 'channel_down')
    assert hasattr(model.native_axis_adapter, 'patch_down')
print('python:', sys.executable)
print('torch:', torch.__version__)
print('TUEV registry preflight:', {
    'method': method, 'backbone_mode': backbone_mode,
    'adapter_type': adapter_type, 'adapter_variant': adapter_variant, 'adapter_operator': adapter_operator,
    'depth_mode': depth_mode, 'train': len(train), 'val': len(val),
    'test': len(test), 'sample_shape': tuple(x.shape),
    'token_grid': tuple(features.shape),
    'input_position_policy': 'sequential_pos_embed_slots_1_to_16',
    'input_scale_divisor': 100.0, 'adapter_diagnostics': diag,
    'target_adapter_parameters': os.environ.get('TARGET_ADAPTER_PARAMS', ''),
})
PY
nvidia-smi || true

echo "TUEV LaBraM registry launch: method=$METHOD seed=$SEED backbone=$BACKBONE_MODE adapter=$ADAPTER_TYPE variant=$ADAPTER_VARIANT operator=$ADAPTER_OPERATOR depth=$ADAPTER_DEPTH_MODE batch=$BATCH_SIZE epochs=$EPOCHS lr=$LR workers=$NUM_WORKERS scale=$INPUT_SCALE_DIVISOR" >&2
echo "git_commit: $(git rev-parse HEAD)" >&2
git status --short >&2
echo "checkpoint_sha256: $(sha256sum ./checkpoints/labram-base.pth | awk '{print $1}')" >&2
echo "run_id: $RUN_ID" >&2

PYTHON_ARGS=(
  --labram_adapter_type "$ADAPTER_TYPE"
  --labram_adapter_variant "$ADAPTER_VARIANT"
  --labram_adapter_operator "$ADAPTER_OPERATOR"
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
  --labram_upper_k 2
  --labram_lora_rank "$LORA_RANK"
  --labram_lora_alpha "$LORA_ALPHA"
  --labram_lora_dropout "$LORA_DROPOUT"
  --labram_lora_target "$LORA_TARGET"
  --input_scale_divisor "$INPUT_SCALE_DIVISOR"
  --disable_rel_pos_bias --abs_pos_emb --disable_qkv_bias
  --save_ckpt_freq "$SAVE_CKPT_FREQ"
  --seed "$SEED"
  --experiment_method "$METHOD"
  --strict_checkpoint_load --no_auto_resume
)
if [[ -n "$TARGET_ADAPTER_PARAMS" ]]; then
  PYTHON_ARGS+=(--target_adapter_params "$TARGET_ADAPTER_PARAMS")
fi
if [[ "$ADAPTER_ZERO_INIT_OUTPUT" == 1 ]]; then
  PYTHON_ARGS+=(--labram_adapter_zero_init_output)
fi
if [[ "$FROZEN_BACKBONE_EVAL_MODE" == 1 ]]; then
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
  --dataset TUEV \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr "$LR" \
  --requested_lr_string "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --layer_decay "$LAYER_DECAY" \
  --backbone_lr_scale "$BACKBONE_LR_SCALE" \
  --head_lr_scale 1.0 \
  --update_freq 1 \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --epochs "$EPOCHS" \
  --drop_path "$DROP_PATH" \
  --smoothing "$SMOOTHING" \
  "${PYTHON_ARGS[@]}"
