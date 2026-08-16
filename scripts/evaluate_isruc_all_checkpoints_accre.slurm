#!/bin/bash
#SBATCH --job-name=labram_isruc_all_eval
#SBATCH --account=neurogroup_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/logs/out/%x_%j.out
#SBATCH --error=/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/logs/err/%x_%j.err

set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM"
ENV_DIR="/data/neurogroup/mingyangjiang/venvs/labram-accre"
DATA_DIR="${DATA_DIR:-/data/neurogroup/mingyangjiang/data/ISRUC}"
RUN_DIR="${RUN_DIR:?RUN_DIR must point to an existing ISRUC checkpoint directory}"
SEED="${SEED:?SEED is required}"
EVAL_OUTPUT="${EVAL_OUTPUT:-$RUN_DIR/all_epoch_test_metrics.jsonl}"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Missing RUN_DIR=$RUN_DIR" >&2
  exit 2
fi

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
if not torch.cuda.is_available():
    raise SystemExit("All-checkpoint evaluation requires CUDA")
PY

echo "ISRUC all-checkpoint evaluation: run_dir=$RUN_DIR seed=$SEED output=$EVAL_OUTPUT" >&2
echo "git_commit: $(git rev-parse HEAD)" >&2

OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 run_class_finetuning.py \
  --output_dir "$RUN_DIR" \
  --log_dir "$RUN_DIR/all_epoch_eval_tensorboard" \
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
  --backbone_lr_scale 0.0 \
  --head_lr_scale 1.0 \
  --update_freq 1 \
  --warmup_epochs 1 \
  --epochs 30 \
  --isruc_sequence_dropout 0.1 \
  --labram_backbone_mode frozen \
  --frozen_backbone_eval_mode \
  --labram_adapter_type generic \
  --labram_adapter_variant full \
  --labram_adapter_bottleneck 400 \
  --labram_adapter_heads 4 \
  --labram_adapter_dropout 0.0 \
  --labram_adapter_init_alpha 0.01 \
  --labram_adapter_gamma 1.0 \
  --labram_adapter_lr_scale 1.0 \
  --labram_adapter_alpha_lr_scale 1.0 \
  --labram_adapter_seed 12345 \
  --labram_adapter_depth_mode none \
  --experiment_method axis_blind \
  --target_adapter_params 161201 \
  --disable_rel_pos_bias \
  --abs_pos_emb \
  --disable_qkv_bias \
  --seed "$SEED" \
  --strict_checkpoint_load \
  --no_auto_resume \
  --eval_all_checkpoints \
  --all_checkpoint_eval_output "$EVAL_OUTPUT"
