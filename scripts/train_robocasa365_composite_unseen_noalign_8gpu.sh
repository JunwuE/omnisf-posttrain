#!/usr/bin/env bash
# Composite Unseen only: same 8-GPU / 120k / batch-512 action-only recipe as mixed.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$PROJECT_ROOT/examples/robocasa365/composite_unseen.env.sh"
CONFIG=robocasa365_target_composite_unseen
EXP_NAME=${EXP_NAME:-robocasa365_target_composite_unseen_noalign_120k_bs512_8gpu}
CONDA_BASE=${CONDA_BASE:-$HOME/miniconda3}
CONDA_ENV=${CONDA_ENV:-omnisf}
cd "$PROJECT_ROOT"

train_command=(
    python -u scripts/train.py "$CONFIG"
    --exp-name "$EXP_NAME"
    --num-train-steps 120000
    --batch-size 512 --gradient-accumulation-steps 1
    --fsdp-devices 8
    --no-align-enabled --align-loss-coeff 0 --vggt-weight-path None
    --num-workers 32
    --data.fast-local-loading
    --lr-schedule.warmup-steps 1000
    --lr-schedule.peak-lr 2.5e-5
    --lr-schedule.decay-steps 120000
    --lr-schedule.decay-lr 2.5e-6
    --log-interval 10
    --save-interval 5000 --keep-period 20000
    --no-wandb-enabled
)
case ${1:-} in
    --dry-run)
        shift
        printf '%q ' "${train_command[@]}" "$@"
        printf '\n'
        exit 0
        ;;
    --help|-h)
        cat <<'HELP'
Usage: bash scripts/train_robocasa365_composite_unseen_noalign_8gpu.sh [MODE] [TRAIN_ARGS...]

Modes (first argument):
  --dry-run        Print the training command without loading Conda or using GPUs.
  --check-only     Validate this dataset, its own norms, weights, tokenizer and GPUs.
  --prepare-stats  Compute full composite_unseen norms on CPU, without training.
  (no mode)        Validate and start training with persistent log + metrics CSV.

Defaults: 8 GPUs, 120000 updates, batch 512, FSDP 8, no alignment, accumulation 1.
Paths load from examples/robocasa365/composite_unseen.env.sh. Exported values win.
Initialization uses the same original OmniSF 35999 params as the mixed launcher.
Append --resume to restore this composite_unseen experiment; --overwrite is refused.

Outputs default to /share-mixed/project/robot-rfm/robot-exp/usr/ejw/posttrain/outputs/omnisf_composite_unseen:
  assets/robocasa365_target_composite_unseen/norm_stats.json
  checkpoints/robocasa365_target_composite_unseen/<EXP_NAME>/
  training_runs/<EXP_NAME>/<UTC timestamp>_<PID>/{train.log,metrics.csv,launch.json,status.json}
Each restart gets a new log directory. LOG_DIR overrides that directory (must be new).
CSV records the trainer's printed loss values, rounded to four decimal places.
HELP
        exit 0
        ;;
esac
mode=train
if [[ ${1:-} == --check-only || ${1:-} == --prepare-stats ]]; then
    mode=${1#--}
    shift
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-default}
export JAX_PLATFORMS=cuda OPENPI_SHOW_TQDM=0
export JAX_PROCESS_COUNT=1 JAX_PROCESS_INDEX=0
export JAX_LOCAL_PROCESS_COUNT=1 JAX_LOCAL_PROCESS_INDEX=0 JAX_NODE_COUNT=1
unset JAX_LOCAL_DEVICE_IDS JAX_COORDINATOR_ADDRESS

if [[ $mode == prepare-stats ]]; then
    mkdir -p "$ROBOCASA365_OUTPUT_ROOT/norm_stats_runs"
    stats_log="$ROBOCASA365_OUTPUT_ROOT/norm_stats_runs/compute_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
    CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu python -u scripts/compute_robocasa365_composite_seen_norm_stats.py --config-name "$CONFIG" "$@" 2>&1 | tee "$stats_log"
    exit 0
fi

# Parse the final effective arguments before consuming GPU memory.
if [[ $mode == check-only ]]; then
    exec python scripts/launch_robocasa365_composite_seen.py --check-only -- "${train_command[@]}" "$@"
fi
LOG_DIR=${LOG_DIR:-$ROBOCASA365_OUTPUT_ROOT/training_runs/$EXP_NAME/$(date -u +%Y%m%dT%H%M%SZ)_$$}
exec python -u scripts/launch_robocasa365_composite_seen.py --output-dir "$LOG_DIR" -- "${train_command[@]}" "$@"
