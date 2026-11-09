#!/usr/bin/env bash
# Single-node, single-process JAX action-only training on all three target splits.
# Shared paths load automatically; exported path settings override the defaults.
# --dry-run prints the command; --check-only validates inputs without training.
# --resume restores this trainer's checkpoints, or retries an empty experiment.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# shellcheck source=../examples/robocasa365/mixed_noalign_8gpu.env.sh
source "$PROJECT_ROOT/examples/robocasa365/mixed_noalign_8gpu.env.sh"
CONFIG=robocasa365_target_mixed
EXP_NAME=${EXP_NAME:-robocasa365_target_mixed_noalign_120k_bs512_8gpu}
CONDA_BASE=${CONDA_BASE:-$HOME/miniconda3}
CONDA_ENV=${CONDA_ENV:-omnisf}
export ROBOCASA365_MIXED_WEIGHTS=${ROBOCASA365_MIXED_WEIGHTS:-1,1,1}
cd "$PROJECT_ROOT"

# Global batch 512 = 64 samples/GPU; one microstep per optimizer update.
# train.py instantiates Pi0, without an alignment head or a VGGT teacher.
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
Usage: bash scripts/train_robocasa365_mixed_noalign_8gpu.sh [MODE] [TRAIN_ARGS...]

Modes (must be the first argument):
  --dry-run     Print the training command without activating Conda.
  --check-only  Check datasets, norm stats, cached tokenizer, weights and GPUs.
  --help        Show this launcher help.
  (no mode)     Validate inputs, then start training in the foreground.

Shared storage paths load from examples/robocasa365/mixed_noalign_8gpu.env.sh.
Export ROBOCASA365_DATA_ROOT, ROBOCASA365_ASSETS_DIR,
  ROBOCASA365_PRETRAIN_PARAMS or ROBOCASA365_CHECKPOINT_DIR to override them.
Optional: EXP_NAME, CONDA_BASE, CONDA_ENV, PROJECT_ROOT,
  CUDA_VISIBLE_DEVICES, ROBOCASA365_MIXED_WEIGHTS (default: 1,1,1),
  OPENPI_DATA_HOME (default: /share/project/robot-rfm/cache/openpi).
The PaliGemma tokenizer must already be present in that cache; startup checks it
locally so a new node does not silently wait for a remote tokenizer download.

Mixture order: atomic, composite_seen, composite_unseen.
Sampling probability is proportional to weight * frames, not weight alone.
Compute robocasa365_target_mixed norm stats using the same data and weights.
The loader samples the three directories directly; no merged dataset is needed.
Append --resume to resume this mixed action-only experiment. An existing empty
experiment directory is retried from step 0 using the pretrained parameters.
See examples/robocasa365/README.md, section 5.3, for data preparation.
HELP
        exit 0
        ;;
esac
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
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-default}
export JAX_PLATFORMS=cuda OPENPI_SHOW_TQDM=0
# One process owns all eight GPUs; do not launch eight copies with torchrun.
export JAX_PROCESS_COUNT=1 JAX_PROCESS_INDEX=0
export JAX_LOCAL_PROCESS_COUNT=1 JAX_LOCAL_PROCESS_INDEX=0 JAX_NODE_COUNT=1
unset JAX_LOCAL_DEVICE_IDS

# Parse the effective CLI overrides, but never initialize JAX on a GPU here.
JAX_PLATFORMS=cpu python - "${train_command[@]:3}" "$@" <<'PY'
import json
import math
import os
from pathlib import Path

required = [
    "ROBOCASA365_DATA_ROOT", "ROBOCASA365_ASSETS_DIR",
    "ROBOCASA365_PRETRAIN_PARAMS", "ROBOCASA365_CHECKPOINT_DIR",
]
missing = [name for name in required if not os.environ.get(name)]
if missing:
    raise SystemExit("Set the following paths first: " + ", ".join(missing))

from openpi.training import config as config_lib
import torch

config = config_lib.cli()
errors = []
if config.align_enabled or config.align_loss_coeff != 0 or config.vggt_weight_path is not None:
    errors.append("This launcher requires alignment disabled and no VGGT weights.")
if config.fsdp_devices != 8 or config.batch_size % 8:
    errors.append("Expected FSDP 8 and a global batch divisible by 8.")
if config.overwrite:
    errors.append("Use a new EXP_NAME instead of --overwrite; use --resume for an existing run.")

sources = config.data.sources
frame_weights = []
for source in sources:
    root = Path(source.data_factory.repo_id)
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        errors.append(f"Missing converted dataset: {info_path}. Convert this split to LeRobot v2.1 first.")
        continue
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v2.1":
        errors.append(f"Expected LeRobot v2.1: {root}")
        continue
    if info.get("total_frames", 0) <= 0 or info.get("total_episodes", 0) <= 0:
        errors.append(f"Empty dataset: {root}")
        continue
    for name in ["tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"]:
        if not (root / "meta" / name).is_file():
            errors.append(f"Missing dataset metadata: {root / 'meta' / name}")
    # Fast local loading skips LeRobot's validation: check the file inventory here.
    episodes_path = root / "meta/episodes.jsonl"
    if episodes_path.is_file():
        episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
        if len(episodes) != info["total_episodes"] or sum(e["length"] for e in episodes) != info["total_frames"]:
            errors.append(f"Episode/frame counts disagree with meta/info.json: {root}")
        video_keys = [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]
        expected = set()
        for episode in episodes:
            index = episode["episode_index"]
            fields = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
            expected.add(info["data_path"].format(**fields))
            expected.update(info["video_path"].format(video_key=key, **fields) for key in video_keys)
        actual = set()
        for directory in [root / "data", root / "videos"]:
            for parent, _, files in os.walk(directory):
                actual.update(str((Path(parent) / name).relative_to(root)) for name in files)
        absent = sorted(expected - actual)
        if absent:
            errors.append(f"{root}: {len(absent)} missing data/video files; first: {absent[0]}")
    frame_weights.append((source.name, info["total_frames"] * source.weight))
    print(f"Dataset {source.name}: {info['total_frames']:,} frames, weight={source.weight:g}, root={root}")

stats_path = Path(config.data.assets.assets_dir) / config.data.assets.asset_id / "norm_stats.json"
if not stats_path.is_file():
    errors.append(f"Missing mixed norm stats: {stats_path}. Run compute_norm_stats.py --config-name robocasa365_target_mixed.")
else:
    norm = json.loads(stats_path.read_text()).get("norm_stats", {})
    for name in ["state", "actions"]:
        for key in ["mean", "std", "q01", "q99"]:
            values = norm.get(name, {}).get(key)
            if not isinstance(values, list) or len(values) != 32 or not all(math.isfinite(v) for v in values):
                errors.append(f"Invalid 32-dimensional {name}/{key} in {stats_path}")

tokenizer_path = Path(os.environ["OPENPI_DATA_HOME"]).expanduser() / "big_vision/paligemma_tokenizer.model"
if not tokenizer_path.is_file():
    errors.append(f"Missing cached PaliGemma tokenizer: {tokenizer_path}. "
                  "Use the prepared shared cache: OPENPI_DATA_HOME=/share/project/robot-rfm/cache/openpi.")
else:
    import sentencepiece

    try:
        tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
        if tokenizer.get_piece_size() <= 0:
            raise ValueError("Tokenizer vocabulary is empty.")
        print(f"Tokenizer cache: {tokenizer_path} ({tokenizer.get_piece_size():,} tokens; local load OK)")
    except (OSError, RuntimeError, ValueError) as exc:
        errors.append(f"Invalid cached PaliGemma tokenizer {tokenizer_path}: {exc}")

params = Path(os.environ["ROBOCASA365_PRETRAIN_PARAMS"]).expanduser()
if not (params / "_METADATA").is_file():
    errors.append(f"Expected an Orbax JAX params checkpoint: {params}")
visible = torch.cuda.device_count()
if visible != 8:
    errors.append(f"Expected 8 visible GPUs, found {visible}. Run on one 8-GPU node; check CUDA_VISIBLE_DEVICES.")
checkpoint = config.checkpoint_dir
if config.resume:
    steps = [p for p in checkpoint.iterdir() if p.is_dir() and p.name.isdigit()] if checkpoint.is_dir() else []
    latest = max(steps, key=lambda p: int(p.name)) if steps else None
    if latest is None and checkpoint.is_dir() and not any(checkpoint.iterdir()):
        # The native checkpoint manager turns resuming off for an empty directory.
        # Keep --resume so it can reuse the directory without deleting anything.
        print("Experiment directory is empty: retrying from step 0 with pretrained parameters; "
              "there is no training state to restore.")
    elif latest is None or int(latest.name) == 0 or not all(
        (latest / item).is_file() for item in ["_CHECKPOINT_METADATA", "params/_METADATA", "train_state/_METADATA"]
    ):
        errors.append(f"No complete training checkpoint to resume in {checkpoint}")
elif checkpoint.exists():
    errors.append(f"Experiment already exists: {checkpoint}. Choose a new EXP_NAME or append --resume.")
if errors:
    raise SystemExit("Preflight failed:\n- " + "\n- ".join(errors))
total = sum(weight for _, weight in frame_weights)
print("Sampling probabilities: " + ", ".join(f"{name}={weight / total:.2%}" for name, weight in frame_weights))
print(f"Ready: {config.num_train_steps:,} updates, global batch {config.batch_size}, "
      f"accumulation {config.gradient_accumulation_steps}, FSDP 8, action loss only.")
print(f"Checkpoint directory: {checkpoint}")
PY

printf 'Training command: '
printf '%q ' "${train_command[@]}" "$@"
printf '\n'
if [[ $check_only == true ]]; then
    exit 0
fi
exec "${train_command[@]}" "$@"
