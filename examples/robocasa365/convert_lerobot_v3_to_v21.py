"""Losslessly split RoboCasa365 v3 files into the pinned LeRobot v2.1 layout.

Requires episode-aligned Parquet files and H.264 keyframes at episode boundaries.
Unsupported layouts fail explicitly. The source is never modified. An incomplete
conversion stays in <output>.incomplete and can be resumed with the same command.
Image statistics are not invented: episodes_stats contains numeric features only.
"""

import argparse
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import hashlib
import json
import math
from pathlib import Path
import time

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def episode_path(root, template, episode_index, **kwargs):
    return root / template.format(episode_chunk=episode_index // 1000, episode_index=episode_index, **kwargs)


def convert_data(args):
    source, output, relative, episodes, fps = args
    marker = output / ".conversion" / (relative.replace("/", "_") + ".json")
    if marker.exists():
        return json.loads(marker.read_text())
    table = pq.read_table(source / relative)
    if table.num_rows != sum(episode["length"] for episode in episodes):
        raise ValueError(f"Episodes cross Parquet file boundaries: {relative}")
    offset = 0
    statistics = []
    for episode in episodes:
        index, length = episode["episode_index"], episode["length"]
        part = table.slice(offset, length)
        if not np.all(part["episode_index"].to_numpy() == index):
            raise ValueError(f"Wrong episode rows: {index}")
        np.testing.assert_array_equal(part["frame_index"].to_numpy(), np.arange(length))
        np.testing.assert_array_equal(
            part["index"].to_numpy(), np.arange(episode["dataset_from_index"], episode["dataset_to_index"])
        )
        np.testing.assert_allclose(part["timestamp"].to_numpy(), np.arange(length) / fps, atol=1e-4, rtol=0)
        stats = {}
        for key in part.column_names:
            column = part[key].combine_chunks()
            values = (
                np.asarray(column.to_pylist())
                if pa.types.is_nested(column.type)
                else column.to_numpy(zero_copy_only=False)
            )
            values = values.reshape(length, -1).astype(np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Nonfinite feature {key}, episode {index}")
            stats[key] = {
                "min": values.min(axis=0).tolist(),
                "max": values.max(axis=0).tolist(),
                "mean": values.mean(axis=0).tolist(),
                "std": values.std(axis=0).tolist(),
                "count": [length],
            }
        path = episode_path(output, DATA_PATH, index)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(part, temporary, compression="zstd")
        # Check persisted values and schema, not just the row count.
        if not pq.read_table(temporary).equals(part):
            raise ValueError(f"Parquet roundtrip changed episode {index}")
        temporary.replace(path)
        statistics.append({"episode_index": index, "stats": stats})
        offset += length
    write_json(marker, statistics)
    return statistics


def convert_video(args):
    source, output, relative, key, episodes, fps = args
    marker = output / ".conversion" / (relative.replace("/", "_") + ".json")
    if marker.exists():
        return json.loads(marker.read_text())
    episodes = sorted(episodes, key=lambda e: e[f"videos/{key}/from_timestamp"])
    destination = None
    previous = -1
    seen = set()
    path = temporary = None

    def finish_episode():
        nonlocal destination
        if destination is None:
            return
        destination.close()
        destination = None
        episode = episodes[previous]
        if seen != set(range(episode["length"])):
            raise ValueError(f"Missing or extra video frames: {key}, episode {episode['episode_index']}")
        with av.open(str(temporary)) as check:
            stream = check.streams.video[0]
            first = next(check.decode(stream))
            if first.pts != 0 or stream.frames != episode["length"]:
                raise ValueError(f"Wrong output timestamps or frame count: {path}")
        temporary.replace(path)

    with av.open(str(source / relative)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        starts = [round(e[f"videos/{key}/from_timestamp"] / float(time_base)) for e in episodes]
        ends = [round(e[f"videos/{key}/to_timestamp"] / float(time_base)) for e in episodes]
        if starts[0] != 0 or starts[1:] != ends[:-1]:
            raise ValueError(f"Noncontiguous video metadata: {relative}")
        try:
            for packet in container.demux(stream):
                if packet.pts is None:
                    continue
                current = bisect_right(starts, packet.pts) - 1
                if current < 0 or packet.pts >= ends[current]:
                    raise ValueError(f"Video packet outside episode metadata: {relative}, {packet.pts}")
                if current != previous:
                    finish_episode()
                    if current != previous + 1 or packet.pts != starts[current] or not packet.is_keyframe:
                        raise ValueError(f"Episode does not start at an independent keyframe: {relative}, {current}")
                    previous = current
                    seen = set()
                    path = episode_path(output, VIDEO_PATH, episodes[current]["episode_index"], video_key=key)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".mp4.tmp")
                    destination = av.open(str(temporary), "w", format="mp4")
                    output_stream = destination.add_stream_from_template(stream)
                position = (packet.pts - starts[current]) * time_base * fps
                frame_index = round(position)
                if abs(float(position) - frame_index) > 1e-3 or frame_index in seen:
                    raise ValueError(f"Duplicate or irregular video timestamp: {relative}, {current}")
                seen.add(frame_index)
                packet.pts -= starts[current]
                packet.dts -= starts[current]
                packet.stream = output_stream
                destination.mux(packet)
            finish_episode()
            if previous != len(episodes) - 1:
                raise ValueError(f"Incomplete input video: {relative}")
        finally:
            if destination is not None:
                destination.close()
    result = {"video": relative, "episodes": len(episodes), "frames": sum(e["length"] for e in episodes)}
    write_json(marker, result)
    return result


def convert(source: Path, target: Path, workers: int):
    source, target = source.resolve(), target.resolve()
    if workers < 1 or source == target or source in target.parents:
        raise ValueError("Use a positive worker count and a separate output directory.")
    if target.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {target}")
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError("Expected LeRobot v3.0 source")
    episodes = [
        row for file in sorted((source / "meta/episodes").rglob("*.parquet")) for row in pq.read_table(file).to_pylist()
    ]
    episodes.sort(key=lambda e: e["episode_index"])
    if [e["episode_index"] for e in episodes] != list(range(info["total_episodes"])):
        raise ValueError("Episode IDs must be contiguous from zero")
    if sum(e["length"] for e in episodes) != info["total_frames"]:
        raise ValueError("Total frame count does not match metadata")
    data_groups, video_groups = defaultdict(list), defaultdict(list)
    cameras = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    for episode in episodes:
        relative = info["data_path"].format(
            chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"]
        )
        data_groups[relative].append(episode)
        for key in cameras:
            relative = info["video_path"].format(
                video_key=key,
                chunk_index=episode[f"videos/{key}/chunk_index"],
                file_index=episode[f"videos/{key}/file_index"],
            )
            video_groups[relative, key].append(episode)
    files = sorted({*data_groups, *(relative for relative, _ in video_groups), "meta/tasks.parquet"})
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "info": info,
                "episodes": episodes,
                "files": [(f, (source / f).stat().st_size, (source / f).stat().st_mtime_ns) for f in files],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    output = target.with_name(target.name + ".incomplete")
    manifest = {
        "source": str(source),
        "source_fingerprint": fingerprint,
        "format": "v3.0 -> v2.1",
        "video_conversion": "packet copy, timestamps rebased per episode",
        "episode_statistics": "numeric features only; no image normalization statistics",
    }
    manifest_path = output / ".conversion/manifest.json"
    if output.exists() and (not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest):
        raise ValueError(f"Unrecognized or changed partial conversion: {output}")
    write_json(manifest_path, manifest)
    start = time.monotonic()
    stats = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {
            pool.submit(convert_data, (source, output, path, group, info["fps"])): ("data", path)
            for path, group in data_groups.items()
        }
        jobs.update(
            {
                pool.submit(convert_video, (source, output, path, key, group, info["fps"])): ("video", path)
                for (path, key), group in video_groups.items()
            }
        )
        for count, future in enumerate(as_completed(jobs), 1):
            kind, path = jobs[future]
            result = future.result()
            if kind == "data":
                stats.extend(result)
            print(f"[{count}/{len(jobs)}] {kind}: {path} ({time.monotonic() - start:.0f}s)", flush=True)
    meta = output / "meta"
    meta.mkdir(exist_ok=True)
    info.update(
        codebase_version="v2.1",
        chunks_size=1000,
        total_chunks=math.ceil(len(episodes) / 1000),
        total_videos=len(episodes) * len(cameras),
        data_path=DATA_PATH,
        video_path=VIDEO_PATH,
    )
    for key in ("data_files_size_in_mb", "video_files_size_in_mb"):
        info.pop(key, None)
    for key in cameras:
        feature = info["features"][key]
        if "video_info" in feature:
            feature["info"] = feature.pop("video_info")
    tasks = pq.read_table(source / "meta/tasks.parquet").to_pylist()
    write_jsonl(meta / "tasks.jsonl", sorted(tasks, key=lambda row: row["task_index"]))
    write_jsonl(
        meta / "episodes.jsonl", [{key: e[key] for key in ("episode_index", "tasks", "length")} for e in episodes]
    )
    write_jsonl(meta / "episodes_stats.jsonl", sorted(stats, key=lambda row: row["episode_index"]))
    write_json(
        meta / "conversion.json",
        {
            **manifest,
            "episodes": len(episodes),
            "frames": info["total_frames"],
            "videos": info["total_videos"],
            "elapsed_seconds": time.monotonic() - start,
        },
    )
    write_json(meta / "info.json", info)
    output.rename(target)
    print(f"Complete: {target} ({len(episodes)} episodes, {info['total_frames']} frames)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    convert(arguments.source, arguments.output, arguments.workers)
