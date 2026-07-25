#!/bin/bash
#SBATCH --job-name=labram_isruc_eval
#SBATCH --account=neurogroup_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/logs/out/%x_%j.out
#SBATCH --error=/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/logs/err/%x_%j.err

set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM"
ENV_DIR="/data/neurogroup/mingyangjiang/venvs/labram-accre"
DATA_DIR="${DATA_DIR:-/data/neurogroup/mingyangjiang/data/ISRUC}"
RUN_DIR="${RUN_DIR:?RUN_DIR must point to an existing ISRUC checkpoint directory}"
SEED="${SEED:?SEED is required}"
BACKBONE_LR_SCALE="${BACKBONE_LR_SCALE:-1.0}"
BACKBONE_MODE="${BACKBONE_MODE:-trainable}"
ADAPTER_TYPE="${ADAPTER_TYPE:-none}"
ADAPTER_VARIANT="${ADAPTER_VARIANT:-full}"
ADAPTER_BOTTLENECK="${ADAPTER_BOTTLENECK:-64}"
ADAPTER_HEADS="${ADAPTER_HEADS:-4}"
ADAPTER_DROPOUT="${ADAPTER_DROPOUT:-0.0}"
ADAPTER_PATCH_OUTPUT_DROPOUT="${ADAPTER_PATCH_OUTPUT_DROPOUT:-0.0}"
ADAPTER_INIT_ALPHA="${ADAPTER_INIT_ALPHA:-0.01}"
ADAPTER_GAMMA="${ADAPTER_GAMMA:-1.0}"
ADAPTER_LR_SCALE="${ADAPTER_LR_SCALE:-1.0}"
ADAPTER_ALPHA_LR_SCALE="${ADAPTER_ALPHA_LR_SCALE:-1.0}"
ADAPTER_DEPTH_MODE="${ADAPTER_DEPTH_MODE:-none}"

if [[ ! -d "$RUN_DIR" || ! -f "$RUN_DIR/checkpoint-best.pth" || ! -f "$RUN_DIR/checkpoint-best-ba.pth" ]]; then
  echo "Missing selected checkpoints in RUN_DIR=$RUN_DIR" >&2
  exit 2
fi

unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
module --force purge >/dev/null 2>&1
module load gentoo/2023 gcc/12.3 python/3.11.5 scipy-stack/2026a >/dev/null 2>&1
export PYTHONNOUSERSITE=1
source "$ENV_DIR/bin/activate"
cd "$REPO_DIR"

echo "ISRUC checkpoint-only evaluation: run_dir=$RUN_DIR seed=$SEED input_scale_divisor=1.0 backbone_lr_scale=$BACKBONE_LR_SCALE backbone_mode=$BACKBONE_MODE adapter_type=$ADAPTER_TYPE" >&2
echo "git_commit: $(git rev-parse HEAD)" >&2

OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 run_class_finetuning.py \
  --output_dir "$RUN_DIR" \
  --log_dir "$RUN_DIR/eval_tensorboard" \
  --model labram_base_patch200_200 \
  --finetune ./checkpoints/labram-base.pth \
  --data_path "$DATA_DIR" \
  --dataset ISRUC \
  --input_scale_divisor 1.0 \
  --batch_size 16 \
  --num_workers 0 \
  --lr 2e-4 \
  --requested_lr_string 2e-4 \
  --weight_decay 0.05 \
  --layer_decay 0.65 \
  --backbone_lr_scale "$BACKBONE_LR_SCALE" \
  --head_lr_scale 1.0 \
  --update_freq 1 \
  --warmup_epochs 1 \
  --epochs 30 \
  --isruc_sequence_dropout 0.1 \
  --labram_adapter_type "$ADAPTER_TYPE" \
  --labram_adapter_variant "$ADAPTER_VARIANT" \
  --labram_adapter_bottleneck "$ADAPTER_BOTTLENECK" \
  --labram_adapter_heads "$ADAPTER_HEADS" \
  --labram_adapter_dropout "$ADAPTER_DROPOUT" \
  --labram_adapter_patch_output_dropout "$ADAPTER_PATCH_OUTPUT_DROPOUT" \
  --labram_adapter_init_alpha "$ADAPTER_INIT_ALPHA" \
  --labram_adapter_gamma "$ADAPTER_GAMMA" \
  --labram_adapter_lr_scale "$ADAPTER_LR_SCALE" \
  --labram_adapter_alpha_lr_scale "$ADAPTER_ALPHA_LR_SCALE" \
  --labram_adapter_depth_mode "$ADAPTER_DEPTH_MODE" \
  --labram_backbone_mode "$BACKBONE_MODE" \
  --disable_rel_pos_bias \
  --abs_pos_emb \
  --disable_qkv_bias \
  --seed "$SEED" \
  --strict_checkpoint_load \
  --no_auto_resume \
  --eval_selected_checkpoints
