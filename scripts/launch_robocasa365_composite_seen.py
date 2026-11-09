"""Validate a composite subset recipe and persist its console output and losses."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


CONFIGS = {
    "robocasa365_target_composite_seen": "robocasa365-target-composite-seen",
    "robocasa365_target_composite_unseen": "robocasa365-target-composite-unseen",
}
METRICS = ["loss", "action_loss", "grad_norm", "learning_rate", "param_norm", "train_step_s", "data_wait_s", "samples_per_s", "steps_per_h"]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, default=str) + "\n")
    pending.replace(path)


def preflight(command, *, require_idle_gpus=True):
    if len(command) < 4 or command[1:3] != ["-u", "scripts/train.py"] or command[3] not in CONFIGS:
        raise ValueError("Expected a composite subset action-only scripts/train.py command")
    config_name = command[3]
    # Config parsing must not allocate JAX memory on any GPU.
    os.environ["JAX_PLATFORMS"] = "cpu"
    from openpi.training import config as config_lib
    from openpi.training.weight_loaders import CheckpointWeightLoader
    import torch

    saved_argv = sys.argv
    try:
        sys.argv = ["train.py", *command[3:]]
        cfg = config_lib.cli()
    finally:
        sys.argv = saved_argv
    errors = []
    if cfg.name != config_name or not isinstance(cfg.data, config_lib.LeRobotRoboCasa365DataConfig):
        errors.append("Expected a single composite subset dataset config")
    if cfg.align_enabled or cfg.align_loss_coeff != 0 or cfg.vggt_weight_path is not None:
        errors.append("Alignment must be disabled with no VGGT weights")
    if cfg.fsdp_devices != 8 or cfg.batch_size <= 0 or cfg.batch_size % 8:
        errors.append("Expected FSDP 8 and a positive global batch divisible by 8")
    if cfg.num_train_steps < 1 or cfg.log_interval < 1 or cfg.save_interval < 1:
        errors.append("Training/log/save intervals must be positive")
    if cfg.overwrite:
        errors.append("--overwrite is refused; use a new EXP_NAME or --resume")
    root = Path(cfg.data.repo_id)
    if root.name != CONFIGS[config_name]:
        errors.append(f"Unexpected dataset directory: {root}")
    info = json.loads((root / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    if info["codebase_version"] != "v2.1" or len(episodes) != info["total_episodes"] or sum(e["length"] for e in episodes) != info["total_frames"]:
        errors.append("Dataset format or episode/frame counts disagree")
    if not episodes or info["total_frames"] <= 0:
        errors.append("Dataset is empty")
    for name in ["tasks.jsonl", "episodes_stats.jsonl"]:
        if not (root / "meta" / name).is_file():
            errors.append(f"Missing metadata: {name}")
    expected = set()
    cameras = [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]
    required_cameras = {"observation.images.robot0_agentview_left", "observation.images.robot0_agentview_right", "observation.images.robot0_eye_in_hand"}
    if set(cameras) != required_cameras:
        errors.append("Unexpected camera keys")
    for ep in episodes:
        idx = ep["episode_index"]
        fields = {"episode_index": idx, "episode_chunk": idx // info["chunks_size"]}
        expected.add(info["data_path"].format(**fields))
        expected.update(info["video_path"].format(video_key=key, **fields) for key in cameras)
    actual = set()
    for directory in [root / "data", root / "videos"]:
        for parent, _, files in os.walk(directory):
            actual.update(str((Path(parent) / name).relative_to(root)) for name in files)
    absent = sorted(expected - actual)
    if absent:
        errors.append(f"{len(absent)} missing episode files; first: {absent[0]}")
    stats_path = Path(cfg.data.assets.assets_dir) / cfg.data.assets.asset_id / "norm_stats.json"
    if not stats_path.is_file():
        errors.append(f"Missing norms: {stats_path}; run the shell launcher with --prepare-stats")
        norm_hash = None
    else:
        norm_hash = hashlib.sha256(stats_path.read_bytes()).hexdigest()
        norms = json.loads(stats_path.read_text())["norm_stats"]
        for key in ["state", "actions"]:
            for stat in ["mean", "std", "q01", "q99"]:
                values = norms.get(key, {}).get(stat)
                if not isinstance(values, list) or len(values) != 32 or not all(math.isfinite(v) for v in values):
                    errors.append(f"Invalid {key}/{stat} norm stats")
            if all(isinstance(norms.get(key, {}).get(k), list) for k in ["std", "q01", "q99"]):
                if any(v < 0 for v in norms[key]["std"]) or any(a > b for a, b in zip(norms[key]["q01"], norms[key]["q99"])):
                    errors.append(f"Invalid {key} standard deviation or quantile order")
    if not isinstance(cfg.weight_loader, CheckpointWeightLoader):
        raise ValueError("Expected initialization from an OmniSF JAX params checkpoint")
    params = Path(cfg.weight_loader.params_path).expanduser()
    if not (params / "_METADATA").is_file():
        errors.append(f"Missing pretrained JAX params: {params}")
    tokenizer = Path(os.environ["OPENPI_DATA_HOME"]) / "big_vision/paligemma_tokenizer.model"
    if not tokenizer.is_file():
        errors.append(f"Missing cached tokenizer: {tokenizer}")
    else:
        import sentencepiece
        if sentencepiece.SentencePieceProcessor(model_file=str(tokenizer)).get_piece_size() <= 0:
            errors.append("Empty tokenizer vocabulary")
    visible = torch.cuda.device_count()
    if visible != 8:
        errors.append(f"Expected 8 visible GPUs, found {visible}")
    # This launcher owns all eight GPUs on this node; prevent accidental overlap.
    gpu_processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True).strip()
    if gpu_processes and require_idle_gpus:
        errors.append("GPUs already have compute processes: " + gpu_processes.replace("\n", "; "))
    elif gpu_processes:
        print("GPUs are occupied; this is a validation-only check. Actual launch requires idle GPUs.", flush=True)
    checkpoint = cfg.checkpoint_dir
    if cfg.resume:
        steps = [p for p in checkpoint.iterdir() if p.is_dir() and p.name.isdigit()] if checkpoint.is_dir() else []
        latest = max(steps, key=lambda p: int(p.name)) if steps else None
        if latest is None and checkpoint.is_dir() and not any(checkpoint.iterdir()):
            print("Empty checkpoint directory: --resume will retry from initialization", flush=True)
        elif latest is None or not all((latest / item).is_file() for item in ["_CHECKPOINT_METADATA", "params/_METADATA", "train_state/_METADATA"]):
            errors.append(f"No complete checkpoint to resume in {checkpoint}")
        elif not json.loads((latest / "_CHECKPOINT_METADATA").read_text()).get("commit_timestamp_nsecs"):
            errors.append(f"Latest checkpoint has not committed: {latest}")
        elif int(latest.name) >= cfg.num_train_steps - 1:
            errors.append("This experiment already reached the requested number of steps")
    elif checkpoint.exists():
        errors.append(f"Experiment already exists: {checkpoint}; use --resume or a new EXP_NAME")
    if errors:
        raise ValueError("Preflight failed:\n- " + "\n- ".join(errors))
    print(f"Ready: {info['total_episodes']:,} {config_name} episodes / {info['total_frames']:,} frames; {cfg.num_train_steps:,} steps; batch {cfg.batch_size}; FSDP 8; no alignment", flush=True)
    return {"config": cfg.name, "exp_name": cfg.exp_name, "resolved_config_repr": repr(cfg), "dataset": str(root), "episodes": info["total_episodes"], "frames": info["total_frames"], "norm_stats": str(stats_path), "norm_sha256": norm_hash, "initialization_params": str(params), "checkpoint_dir": str(checkpoint), "num_train_steps": cfg.num_train_steps, "batch_size": cfg.batch_size, "gradient_accumulation_steps": cfg.gradient_accumulation_steps, "fsdp_devices": cfg.fsdp_devices, "align_enabled": cfg.align_enabled, "resume": cfg.resume, "log_interval": cfg.log_interval}


def run_logged(command, output, metadata):
    """Flush logs and CSV as they arrive; preserve the trainer's failure exit code."""
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, JAX_PLATFORMS="cuda")
    # Match the shell's activated interpreter instead of relying on a later PATH lookup.
    command = [sys.executable, *command[1:]]
    metadata.update(started_at_utc=now(), command=command, cwd=str(Path.cwd()), log_dir=str(output), wrapper_pid=os.getpid(), environment={key:env[key] for key in ["CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_PYTHON_CLIENT_PREALLOCATE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENPI_DATA_HOME"] if key in env})
    write_json(output / "launch.json", metadata)
    write_json(output / "status.json", {"status":"starting", "updated_at_utc":now()})
    print(f"Training log: {output / 'train.log'}\nMetrics CSV: {output / 'metrics.csv'}", flush=True)
    process = None
    previous_handlers = {}
    interrupted_by = None

    def forward(signum, _frame):
        nonlocal interrupted_by
        interrupted_by = signum
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass  # The child may have exited between poll() and killpg().

    try:
        with (output / "train.log").open("x", buffering=1) as log, (output / "metrics.csv").open("x", newline="", buffering=1) as metrics:
            writer = csv.DictWriter(metrics, fieldnames=["recorded_at_utc", "step", *METRICS])
            writer.writeheader()
            log.write("# Launch metadata: " + json.dumps(metadata, default=str) + "\n")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1, env=env, start_new_session=True)
            metadata["training_pid"] = process.pid
            write_json(output / "launch.json", metadata)
            (output / "training.pid").write_text(str(process.pid) + "\n")
            for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
                previous_handlers[sig] = signal.signal(sig, forward)
            status = {"status":"running", "training_pid":process.pid, "updated_at_utc":now()}
            write_json(output / "status.json", status)
            for line in process.stdout:
                log.write(line)
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except BrokenPipeError:
                    # A vanished viewer must not discard the persistent training log.
                    pass
                match = re.search(r"\bStep (\d+):", line)
                if match:
                    parsed = dict(re.findall(r"\b(" + "|".join(METRICS) + r")=([^,\s]+)", line))
                    if "loss" in parsed:
                        # The action-only trainer labels action loss simply 'loss'.
                        parsed.setdefault("action_loss", parsed["loss"])
                        row = {"recorded_at_utc":now(), "step":int(match[1]), **parsed}
                        writer.writerow(row)
                        metrics.flush()
                        log.flush()
                        os.fsync(metrics.fileno())
                        os.fsync(log.fileno())
                        status.update(updated_at_utc=now(), last_metrics=row)
                        write_json(output / "status.json", status)
            code = process.wait()
            status.update(status="completed" if code == 0 else "failed", exit_code=code, interrupted_by=interrupted_by, updated_at_utc=now())
            write_json(output / "status.json", status)
            return code if code >= 0 else 128 - code
    except BaseException as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        write_json(output / "status.json", {"status":"failed", "error":str(exc), "updated_at_utc":now()})
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not args.check_only and args.output_dir is None:
        parser.error("--output-dir is required when starting training")
    metadata = preflight(command, require_idle_gpus=not args.check_only)
    if args.check_only:
        print(json.dumps(metadata, indent=2), flush=True)
        return
    raise SystemExit(run_logged(command, args.output_dir.expanduser().resolve(), metadata))


if __name__ == "__main__":
    main()
