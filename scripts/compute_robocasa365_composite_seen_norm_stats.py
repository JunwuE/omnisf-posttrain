"""Compute full composite subset norms, including exact action-window multiplicities.

Uses the same full-data method as the mixed run: float32 inputs, float64
moments, exact weighted empirical quantiles, and native final-action padding.
Only numeric Parquet columns are read. Existing statistics are never replaced.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.policies.robocasa365_policy import RoboCasa365Inputs
from openpi.shared import normalize
from openpi.training import config as config_lib


def multiplicities(length, horizon):
    counts = np.minimum(np.arange(1, length + 1, dtype=np.int64), horizon)
    counts[-1] = length * horizon - int(counts[:-1].sum())
    assert counts.sum() == length * horizon and np.all(counts > 0)
    return counts


def native_indices(length, horizon):
    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.episode_data_index = {"from": torch.tensor([0]), "to": torch.tensor([length])}
    dataset.delta_indices = {"action": list(range(horizon))}
    return np.asarray([dataset._get_query_indices(i, 0)[0]["action"] for i in range(length)])


def column_stats(values, weights):
    x = values.astype(np.float64)
    total = float(weights.sum())
    mean = float(np.dot(x, weights) / total)
    std = float(np.sqrt(np.dot((x - mean) ** 2, weights) / total))
    order = np.argsort(values, kind="stable")
    positions = np.searchsorted(np.cumsum(weights[order]), total * np.array([0.01, 0.99]), side="left")
    quantiles = values[order[np.minimum(positions, len(order) - 1)]]
    return mean, std, float(quantiles[0]), float(quantiles[1])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--config-name", choices=["robocasa365_target_composite_seen", "robocasa365_target_composite_unseen"], default="robocasa365_target_composite_seen")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    started = time.monotonic()
    cfg = config_lib.get_config(args.config_name)
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    target = Path(cfg.data.assets.assets_dir) / data.asset_id / "norm_stats.json"
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing statistics: {target}")
    assert data.action_sequence_keys == ("action",)
    assert len(data.data_transforms.inputs) == 1
    assert isinstance(data.data_transforms.inputs[0], RoboCasa365Inputs)
    horizon, dim = cfg.model.action_horizon, cfg.model.action_dim
    assert horizon == 50 and dim == 32
    root = Path(data.repo_id)
    info_file, episodes_file = root / "meta/info.json", root / "meta/episodes.jsonl"
    info = json.loads(info_file.read_text())
    episodes = [json.loads(line) for line in episodes_file.read_text().splitlines() if line.strip()]
    assert info["codebase_version"] == "v2.1"
    assert len(episodes) == info["total_episodes"]
    assert sum(e["length"] for e in episodes) == info["total_frames"]
    for length in (1, 2, 17, 49, 50, 51, 173):
        expected = np.bincount(native_indices(length, horizon).ravel(), minlength=length)
        np.testing.assert_array_equal(expected, multiplicities(length, horizon))
    records, offset = [], 0
    for index, episode in enumerate(episodes):
        assert episode["episode_index"] == index and episode["length"] > 0
        path = root / info["data_path"].format(episode_index=index, episode_chunk=index // info["chunks_size"])
        records.append((path, index, episode["length"], offset))
        offset += episode["length"]
    print(f"Reading all {offset:,} frames / {len(records):,} episodes", flush=True)
    states = np.empty((offset, 16), dtype=np.float32, order="F")
    actions = np.empty((offset, 12), dtype=np.float32, order="F")
    state_weights = np.ones(offset, dtype=np.float64)
    action_weights = np.empty(offset, dtype=np.float64)

    def read_episode(record):
        path, index, length, start = record
        table = pq.read_table(path, columns=["observation.state", "action", "episode_index", "frame_index"], use_threads=False)
        assert len(table) == length and np.all(table["episode_index"].to_numpy() == index), path
        np.testing.assert_array_equal(table["frame_index"].to_numpy(), np.arange(length))
        for key, dest, width in [("observation.state", states, 16), ("action", actions, 12)]:
            column = table[key].combine_chunks()
            assert column.null_count == 0
            values = np.asarray(column.flatten().to_numpy(zero_copy_only=False), dtype=np.float32).reshape(length, width)
            assert np.isfinite(values).all(), path
            dest[start:start + length] = values
        action_weights[start:start + length] = multiplicities(length, horizon)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count, _ in enumerate(pool.map(read_episode, records), 1):
            if count % 1000 == 0 or count == len(records):
                print(f"Read {count}/{len(records)} episodes ({time.monotonic() - started:.1f}s)", flush=True)
    validation = []
    for position in (0, len(records) // 2, len(records) - 1):
        path, _, length, start = records[position]
        native = native_indices(length, horizon)
        counts = multiplicities(length, horizon)
        np.testing.assert_array_equal(np.bincount(native.ravel(), minlength=length), counts)
        aa = actions[start:start + length]
        expanded = aa[native].reshape(-1, 12).astype(np.float64)
        mean = np.sum(aa.astype(np.float64) * counts[:, None], axis=0) / (length * horizon)
        np.testing.assert_allclose(mean, expanded.mean(axis=0), rtol=1e-12, atol=1e-12)
        variance = np.sum((aa.astype(np.float64) - mean) ** 2 * counts[:, None], axis=0) / (length * horizon)
        np.testing.assert_allclose(variance, expanded.var(axis=0), rtol=1e-10, atol=1e-12)
        for idx in (0, length // 2, length - 1):
            obs = {key: np.zeros((3, 1, 1), dtype=np.uint8) for key in ["observation/image", "observation/right_image", "observation/wrist_image"]}
            obs.update({"observation/state": states[start + idx], "actions": aa[native[idx]]})
            transformed = data.data_transforms.inputs[0](obs)
            np.testing.assert_array_equal(transformed["state"], np.pad(states[start + idx], (0, dim - 16)))
            np.testing.assert_array_equal(transformed["actions"], np.pad(aa[native[idx]], ((0, 0), (0, dim - 12))))
        validation.append({"episode": str(path), "native_windows_and_transforms": "passed"})
    assert action_weights.sum() == offset * horizon
    stats = {}
    for name, values, weights in [("state", states, state_weights), ("actions", actions, action_weights)]:
        print(f"Computing full {name} statistics", flush=True)
        with ThreadPoolExecutor(max_workers=min(args.workers, 4)) as pool:
            columns = list(pool.map(lambda i: column_stats(values[:, i], weights), range(values.shape[1])))
        fields = np.zeros((4, dim), dtype=np.float64)
        fields[:, :values.shape[1]] = np.asarray(columns).T
        assert np.isfinite(fields).all() and np.all(fields[1] >= 0) and np.all(fields[2] <= fields[3])
        stats[name] = normalize.NormStats(mean=fields[0], std=fields[1], q01=fields[2], q99=fields[3])
    serialized = normalize.serialize_json(stats)
    restored = normalize.deserialize_json(serialized)
    for name in stats:
        for key in ("mean", "std", "q01", "q99"):
            np.testing.assert_array_equal(getattr(restored[name], key), getattr(stats[name], key))
    report = dict(config=cfg.name, created_at=datetime.now(timezone.utc).isoformat(), dataset=str(root), total_frames=offset, total_episodes=len(records), action_horizon=horizon, model_dim=dim, coverage="all frames; no sampling", method="float64 two-pass weighted moments; exact weighted empirical CDF quantiles", action_weighting="every 50-step forward window with final-action boundary repetition", difference_from_native_script="exact quantiles instead of 5000-bin histogram approximation", info_sha256=digest(info_file), episodes_sha256=digest(episodes_file), script_sha256=digest(Path(__file__)), sha256=hashlib.sha256(serialized.encode()).hexdigest(), validation=validation, elapsed_seconds=time.monotonic() - started)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(f"norm_stats.json.pending-{os.getpid()}")
    with pending.open("x") as f:
        f.write(serialized)
        f.flush()
        os.fsync(f.fileno())
    os.link(pending, target)
    pending.unlink()
    target.with_name("norm_stats.provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    loaded = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert loaded.norm_stats is not None
    for name in stats:
        np.testing.assert_array_equal(loaded.norm_stats[name].q01, stats[name].q01)
        np.testing.assert_array_equal(loaded.norm_stats[name].q99, stats[name].q99)
    print(f"DONE: {target} ({time.monotonic() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
