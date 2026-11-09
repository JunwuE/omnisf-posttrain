"""Wait for this composite_seen run to finish, then exec the matching unseen launcher."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def seen_complete(run):
    """Do not release on a crash, missing final checkpoint, or uncommitted save."""
    launch = json.loads((run / "launch.json").read_text())
    expected = {"config": "robocasa365_target_composite_seen", "num_train_steps": 120000,
                "batch_size": 512, "gradient_accumulation_steps": 1, "fsdp_devices": 8,
                "align_enabled": False}
    if any(launch.get(key) != value for key, value in expected.items()):
        raise ValueError("The seen run does not match the requested 8-GPU/120k/batch-512 recipe")
    status = json.loads((run / "status.json").read_text())
    if status["status"] == "failed":
        raise RuntimeError(f"Seen training failed: {status}")
    if status["status"] != "completed":
        pid = status.get("training_pid", launch.get("training_pid"))
        if not pid:
            raise RuntimeError("Seen training has no recorded PID")
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes()
            if b"scripts/train.py" not in command or b"robocasa365_target_composite_seen" not in command:
                raise RuntimeError("Recorded seen PID no longer belongs to this training process")
        except FileNotFoundError as exc:
            # The logger may still be draining stdout after the trainer exits.
            wrapper = launch.get("wrapper_pid")
            wrapper_command = Path(f"/proc/{wrapper}/cmdline") if wrapper else None
            if wrapper_command is None or not wrapper_command.exists() or b"launch_robocasa365_composite_seen.py" not in wrapper_command.read_bytes():
                raise RuntimeError("Seen process exited without a completed status") from exc
        return False, status
    if status.get("exit_code") != 0:
        raise RuntimeError("Seen did not exit successfully")
    final = Path(launch["checkpoint_dir"]) / "119999"
    for name in ["_CHECKPOINT_METADATA", "params/_METADATA", "train_state/_METADATA"]:
        if not (final / name).is_file():
            raise RuntimeError(f"Missing final seen checkpoint component: {final / name}")
    if not json.loads((final / "_CHECKPOINT_METADATA").read_text()).get("commit_timestamp_nsecs"):
        raise RuntimeError("Final seen checkpoint has not committed")
    return True, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    directory = args.queue_dir.resolve()
    queue = json.loads((directory / "queue.json").read_text())
    seen_run = Path(queue["seen_log_dir"])
    if args.check_only:
        ready, status = seen_complete(seen_run)
        print(json.dumps({"seen_complete": ready, "seen_status": status, "queue": queue}, indent=2))
        return
    # Keep the lock until exec; its close-on-exec descriptor is then released.
    with (directory / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = directory / "status.json"
        if status_path.exists() and json.loads(status_path.read_text()).get("status") == "released_to_unseen":
            raise RuntimeError("This queue already released unseen; refusing a duplicate launch")

        def cancelled(signum, _frame):
            write_json(status_path, {"status": "cancelled", "signal": signum, "updated_at_utc": now()})
            raise SystemExit(128 + signum)

        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
            signal.signal(sig, cancelled)
        try:
            last_report = 0
            while True:
                complete, seen_status = seen_complete(seen_run)
                gpu_processes = ""
                if complete:
                    gpu_processes = subprocess.check_output(
                        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True
                    ).strip()
                state = {"status": "waiting_for_idle_gpus" if complete else "waiting_for_seen",
                         "updated_at_utc": now(), "queue_pid": os.getpid(), "seen_log_dir": str(seen_run),
                         "seen_last_step": seen_status.get("last_metrics", {}).get("step"),
                         "unseen_log_dir": queue["unseen_log_dir"]}
                if complete and not gpu_processes:
                    break
                write_json(status_path, state)
                if time.monotonic() - last_report >= 300:
                    print(json.dumps(state), flush=True)
                    last_report = time.monotonic()
                time.sleep(args.poll_seconds)

            if Path(queue["unseen_log_dir"]).exists():
                raise FileExistsError("The queued unseen log directory already exists")
            env = dict(os.environ)
            env.update(queue["environment"])
            state.update(status="released_to_unseen", updated_at_utc=now())
            write_json(status_path, state)
            print(f"Seen completed and checkpoint 119999 committed. Starting unseen: {queue['unseen_launcher']}", flush=True)
            os.chdir(queue["project_root"])
            os.execvpe("bash", ["bash", queue["unseen_launcher"]], env)
        except Exception as exc:
            write_json(status_path, {"status": "blocked", "error": str(exc), "updated_at_utc": now()})
            raise


if __name__ == "__main__":
    main()
