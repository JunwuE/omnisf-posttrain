#!/usr/bin/env bash
# Local, single-process JAX training; no Slurm or torchrun is needed.
# Source your environment/path settings first. Append --resume to resume.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CONFIG=robocasa365_target_atomic
EXP_NAME=${EXP_NAME:-robocasa365_target_atomic_60k_bs256_4gpu}
CONDA_BASE=${CONDA_BASE:-$HOME/miniconda3}
CONDA_ENV=${CONDA_ENV:-omnisf}
cd "$PROJECT_ROOT"

# Keep the same 32 samples/GPU as the eight-GPU recipe. Two microsteps
# per optimizer update give an effective global batch of 128 * 2 = 256.
train_command=(
    python -u scripts/train_align.py "$CONFIG"
    --exp-name "$EXP_NAME"
    --num-train-steps 60000
    --batch-size 128 --gradient-accumulation-steps 2
    --fsdp-devices 4
    --align-target-model omnivggt --align-device-mode shared
    --align-vggt-devices 4 --align-feature-batch-size 16
    --num-workers 16
    --lr-schedule.decay-steps 60000
    --save-interval 5000 --keep-period None
    --no-wandb-enabled
)
if [[ ${1:-} == --dry-run ]]; then
    shift
    printf '%q ' "${train_command[@]}" "$@"
    printf '\n'
    exit 0
fi

# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
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
    'ROBOCASA365_DATA_ROOT', 'ROBOCASA365_ASSETS_DIR',
    'ROBOCASA365_PRETRAIN_PARAMS', 'ROBOCASA365_VGGT_PATH',
    'ROBOCASA365_CHECKPOINT_DIR',
]
missing = [name for name in required if not os.environ.get(name)]
if missing:
    raise SystemExit('Set the following paths first: ' + ', '.join(missing))
dataset = Path(os.environ['ROBOCASA365_DATA_ROOT']) / 'robocasa365-target-atomic'
info = json.loads((dataset / 'meta/info.json').read_text())
if info['codebase_version'] != 'v2.1':
    raise SystemExit('Expected the converted LeRobot v2.1 dataset.')
stats = Path(os.environ['ROBOCASA365_ASSETS_DIR']) / 'robocasa365_target_atomic/norm_stats.json'
norm = json.loads(stats.read_text())['norm_stats']
for name in ['state', 'actions']:
    if len(norm[name]['mean']) != 32 or norm[name]['q01'] is None or norm[name]['q99'] is None:
        raise SystemExit('Incomplete normalization statistics: ' + str(stats))
params = Path(os.environ['ROBOCASA365_PRETRAIN_PARAMS'])
if not params.is_dir() or not (params / '_METADATA').is_file():
    raise SystemExit('Expected an Orbax JAX params checkpoint: ' + str(params))
vggt = Path(os.environ['ROBOCASA365_VGGT_PATH']) / 'OmniVGGT.safetensors'
if not vggt.is_file():
    raise SystemExit('Missing OmniVGGT weights: ' + str(vggt))
if torch.cuda.device_count() != 4:
    raise SystemExit(f'Expected 4 visible GPUs, found {torch.cuda.device_count()}. Set CUDA_VISIBLE_DEVICES if needed.')
print('Ready: 4 GPUs, 60,000 optimizer updates, effective batch 256 (128 x 2).', flush=True)
PY

exec "${train_command[@]}" "$@"
