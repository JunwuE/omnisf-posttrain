#!/usr/bin/env bash
#SBATCH --job-name=omnisf-rc365
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=omnisf-robocasa365-%j.out
#SBATCH --error=omnisf-robocasa365-%j.err

# Submit from the repository root. Set the cluster partition/account with sbatch:
#   sbatch --partition=<partition> [--account=<account>] slurm_scripts/train_robocasa365_8gpu.sbatch
# Optional overrides are passed directly to train_align.py, e.g. --resume.
# Preview without submitting a job or activating Conda:
#   bash slurm_scripts/train_robocasa365_8gpu.sbatch --dry-run
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
CONFIG=${CONFIG:-robocasa365_target_atomic}
EXP_NAME=${EXP_NAME:-"${CONFIG}_60k_bs256"}
CONDA_ENV=${CONDA_ENV:-omnisf}

dry_run=0
if [[ ${1:-} == --dry-run ]]; then
    dry_run=1
    shift
fi

cd "$PROJECT_ROOT"
if [[ ! -f scripts/train_align.py ]]; then
    echo "Set PROJECT_ROOT to the omnisf checkout: scripts/train_align.py was not found." >&2
    exit 1
fi

train_command=(
    python -u scripts/train_align.py "$CONFIG"
    --exp-name "$EXP_NAME"
    --num-train-steps 60000
    --batch-size 256
    --gradient-accumulation-steps 1
    --fsdp-devices 8
    --align-target-model omnivggt
    --align-device-mode shared
    --align-vggt-devices 8
    --align-feature-batch-size 16
    --num-workers 16
    --lr-schedule.decay-steps 60000
    --save-interval 5000
    --keep-period None
    --no-wandb-enabled
    "$@"
)

printf 'PROJECT_ROOT=%s\nCONFIG=%s\nEXP_NAME=%s\n' "$PROJECT_ROOT" "$CONFIG" "$EXP_NAME"
printf 'Training command: '
printf '%q ' "${train_command[@]}"
printf '\n'
if (( dry_run )); then
    exit 0
fi

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "Submit this script with sbatch. Use --dry-run to preview it locally." >&2
    exit 1
fi
if [[ ${SLURM_JOB_NUM_NODES:-1} != 1 || ${SLURM_NTASKS:-1} != 1 ]]; then
    echo "This launcher requires one node and one task with all 8 GPUs visible." >&2
    exit 1
fi

# Honor the server's Conda installation; do not assume /root/miniconda3 exists.
if [[ -z ${CONDA_BASE:-} ]]; then
    if [[ -n ${CONDA_EXE:-} && -x $CONDA_EXE ]]; then
        CONDA_BASE=$("$CONDA_EXE" info --base)
    elif command -v conda >/dev/null 2>&1; then
        CONDA_BASE=$(conda info --base)
    elif [[ -f $HOME/miniconda3/etc/profile.d/conda.sh ]]; then
        CONDA_BASE="$HOME/miniconda3"
    else
        echo "Export CONDA_BASE=/path/to/miniconda3 before submitting the job." >&2
        exit 1
    fi
fi
if [[ ! -f $CONDA_BASE/etc/profile.d/conda.sh ]]; then
    echo "Conda activation script not found: $CONDA_BASE/etc/profile.d/conda.sh" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_PLATFORMS=cuda
export OPENPI_SHOW_TQDM=0

# One Python process owns all eight allocated GPUs. Preserve Slurm's device mask.
export JAX_PROCESS_COUNT=1
export JAX_PROCESS_INDEX=0
export JAX_LOCAL_PROCESS_COUNT=1
export JAX_LOCAL_PROCESS_INDEX=0
export JAX_NODE_COUNT=1
unset JAX_LOCAL_DEVICE_IDS

printf 'Job=%s Node=%s Conda=%s Python=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$CONDA_PREFIX" "$(command -v python)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<set by Slurm>}"
nvidia-smi -L

exec srun --nodes=1 --ntasks=1 --gres=gpu:8 --gpu-bind=none --export=ALL "${train_command[@]}"
