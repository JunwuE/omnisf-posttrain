#!/usr/bin/env bash
# Atomic action-only post-training through the standard JAX trainer.
# Source your path settings and choose a separate EXP_NAME before launching.
# --dry-run prints the command; --check-only validates paths and GPU visibility.
# Append --resume only when resuming a checkpoint from this action-only trainer.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CONFIG=robocasa365_target_atomic
EXP_NAME=${EXP_NAME:-robocasa365_target_atomic_noalign_60k_bs256_4gpu}
CONDA_BASE=${CONDA_BASE:-$HOME/miniconda3}
CONDA_ENV=${CONDA_ENV:-omnisf}
cd "$PROJECT_ROOT"

# Global batch 256 = 64 samples per GPU, one optimizer update per batch.
# train.py creates Pi0 directly; it does not instantiate Pi0Align or a teacher.
train_command=(
    python -u scripts/train.py "$CONFIG"
    --exp-name "$EXP_NAME"
    --num-train-steps 60000
    --batch-size 256 --gradient-accumulation-steps 1
    --fsdp-devices 4
    --no-align-enabled --align-loss-coeff 0 --vggt-weight-path None
    --num-workers 16
    --data.fast-local-loading
    --lr-schedule.decay-steps 60000
    --log-interval 10
    --save-interval 5000 --keep-period None
    --no-wandb-enabled
)
if [[ ${1:-} == --dry-run ]]; then
    shift
    printf '%q ' "${train_command[@]}" "$@"
    printf '\n'
    exit 0
fi
check_only=false
if [[ ${1:-} == --check-only ]]; then
    check_only=true
    shift
fi

# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# No GPU teacher shares this process. Allow JAX to reuse allocations while
# avoiding an up-front reservation of most GPU memory.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-default}
export JAX_PLATFORMS=cuda OPENPI_SHOW_TQDM=0
export JAX_PROCESS_COUNT=1 JAX_PROCESS_INDEX=0
export JAX_LOCAL_PROCESS_COUNT=1 JAX_LOCAL_PROCESS_INDEX=0 JAX_NODE_COUNT=1
unset JAX_LOCAL_DEVICE_IDS

python - <<'PY'
import json
import os
from pathlib import Path

import torch

required = [
    "ROBOCASA365_DATA_ROOT", "ROBOCASA365_ASSETS_DIR",
    "ROBOCASA365_PRETRAIN_PARAMS", "ROBOCASA365_CHECKPOINT_DIR",
]
missing = [name for name in required if not os.environ.get(name)]
if missing:
    raise SystemExit("Set the following paths first: " + ", ".join(missing))
dataset = Path(os.environ["ROBOCASA365_DATA_ROOT"]).expanduser() / "robocasa365-target-atomic"
info = json.loads((dataset / "meta/info.json").read_text())
if info["codebase_version"] != "v2.1":
    raise SystemExit("Expected the converted LeRobot v2.1 dataset.")
stats = Path(os.environ["ROBOCASA365_ASSETS_DIR"]).expanduser() / "robocasa365_target_atomic/norm_stats.json"
norm = json.loads(stats.read_text())["norm_stats"]
for name in ["state", "actions"]:
    if any(norm[name].get(key) is None or len(norm[name][key]) != 32 for key in ["mean", "std", "q01", "q99"]):
        raise SystemExit("Incomplete normalization statistics: " + str(stats))
params = Path(os.environ["ROBOCASA365_PRETRAIN_PARAMS"]).expanduser()
if not params.is_dir() or not (params / "_METADATA").is_file():
    raise SystemExit("Expected an Orbax JAX params checkpoint: " + str(params))
visible = torch.cuda.device_count()
if visible != 4:
    raise SystemExit(f"Expected 4 visible GPUs, found {visible}. Set CUDA_VISIBLE_DEVICES if needed.")
print("Preflight passed: local atomic data, norm stats, JAX weights, 4 visible GPUs. No VGGT required.", flush=True)
PY

printf 'Training command: '
printf '%q ' "${train_command[@]}" "$@"
printf '\n'
if [[ $check_only == true ]]; then
    exit 0
fi
exec "${train_command[@]}" "$@"
