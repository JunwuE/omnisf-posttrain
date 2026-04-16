"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

# Monkey-patch to fix 'List' feature type error in old datasets
try:
    import datasets.features.features as features

    _OLD_GENERATE_FROM_DICT = features.generate_from_dict

    def _new_generate_from_dict(obj):
        if isinstance(obj, dict) and obj.get("_type") == "List":
            obj["_type"] = "Sequence"
        return _OLD_GENERATE_FROM_DICT(obj)

    features.generate_from_dict = _new_generate_from_dict
except (ImportError, AttributeError):
    # If datasets or the function doesn't exist, do nothing.
    pass
# End of monkey-patch

from collections.abc import Sequence
import dataclasses
import math
import multiprocessing
import pathlib

import numpy as np
import torch
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


@dataclasses.dataclass(frozen=True)
class ComponentSamplingPlan:
    quota: int
    per_sample_weight: float


class WeightedRunningStats:
    """Compute weighted statistics for variable-length vectors."""

    def __init__(self, num_quantile_bins: int = 5000):
        self._num_quantile_bins = num_quantile_bins
        self._weight_sum = np.zeros(0, dtype=np.float64)
        self._sum = np.zeros(0, dtype=np.float64)
        self._sum_sq = np.zeros(0, dtype=np.float64)
        self._min = np.full(0, np.inf, dtype=np.float64)
        self._max = np.full(0, -np.inf, dtype=np.float64)
        self._histograms: list[np.ndarray | None] = []
        self._bin_edges: list[np.ndarray | None] = []

    def update(self, batch: np.ndarray, sample_weight: float = 1.0) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 0:
            raise ValueError("Expected batch to have at least one dimension.")
        if sample_weight < 0:
            raise ValueError("Sample weights must be non-negative.")

        batch = batch.reshape(-1, batch.shape[-1])
        if batch.size == 0:
            return

        _, vector_length = batch.shape
        self._ensure_dim(vector_length)

        batch_min = np.min(batch, axis=0)
        batch_max = np.max(batch, axis=0)
        for i in range(vector_length):
            self._ensure_bin_edges(i, float(batch_min[i]), float(batch_max[i]))

        batch_weight = sample_weight * batch.shape[0]
        self._weight_sum[:vector_length] += batch_weight
        self._sum[:vector_length] += sample_weight * np.sum(batch, axis=0)
        self._sum_sq[:vector_length] += sample_weight * np.sum(batch**2, axis=0)
        self._min[:vector_length] = np.minimum(self._min[:vector_length], batch_min)
        self._max[:vector_length] = np.maximum(self._max[:vector_length], batch_max)

        hist_weights = np.full(batch.shape[0], sample_weight, dtype=np.float64)
        for i in range(vector_length):
            histogram, _ = np.histogram(batch[:, i], bins=self._bin_edges[i], weights=hist_weights)
            self._histograms[i] += histogram

    def get_statistics(self) -> normalize.NormStats:
        if not np.any(self._weight_sum > 0):
            raise ValueError("Cannot compute statistics for an empty dataset.")

        last_dim = int(np.max(np.flatnonzero(self._weight_sum > 0))) + 1
        mean = self._sum[:last_dim] / self._weight_sum[:last_dim]
        variance = self._sum_sq[:last_dim] / self._weight_sum[:last_dim] - mean**2
        stddev = np.sqrt(np.maximum(0.0, variance))
        q01, q99 = self._compute_quantiles(last_dim, [0.01, 0.99])
        return normalize.NormStats(mean=mean, std=stddev, q01=q01, q99=q99)

    def _ensure_dim(self, vector_length: int) -> None:
        if vector_length <= self._weight_sum.size:
            return

        extra_dims = vector_length - self._weight_sum.size
        self._weight_sum = np.pad(self._weight_sum, (0, extra_dims))
        self._sum = np.pad(self._sum, (0, extra_dims))
        self._sum_sq = np.pad(self._sum_sq, (0, extra_dims))
        self._min = np.pad(self._min, (0, extra_dims), constant_values=np.inf)
        self._max = np.pad(self._max, (0, extra_dims), constant_values=-np.inf)
        self._histograms.extend([None] * extra_dims)
        self._bin_edges.extend([None] * extra_dims)

    def _ensure_bin_edges(self, index: int, batch_min: float, batch_max: float) -> None:
        if self._bin_edges[index] is None:
            self._histograms[index] = np.zeros(self._num_quantile_bins, dtype=np.float64)
            self._bin_edges[index] = self._make_bin_edges(batch_min, batch_max)
            return

        current_min = float(self._min[index])
        current_max = float(self._max[index])
        if batch_min < current_min or batch_max > current_max:
            self._rebin(index, min(current_min, batch_min), max(current_max, batch_max))

    def _make_bin_edges(self, min_value: float, max_value: float) -> np.ndarray:
        if np.isclose(min_value, max_value):
            delta = max(abs(min_value), 1.0) * 1e-6
            min_value -= delta
            max_value += delta
        else:
            min_value -= 1e-10
            max_value += 1e-10
        return np.linspace(min_value, max_value, self._num_quantile_bins + 1, dtype=np.float64)

    def _rebin(self, index: int, new_min: float, new_max: float) -> None:
        assert self._histograms[index] is not None
        assert self._bin_edges[index] is not None

        new_edges = self._make_bin_edges(new_min, new_max)
        histogram, _ = np.histogram(self._bin_edges[index][:-1], bins=new_edges, weights=self._histograms[index])
        self._histograms[index] = histogram.astype(np.float64, copy=False)
        self._bin_edges[index] = new_edges

    def _compute_quantiles(self, vector_length: int, quantiles: Sequence[float]) -> list[np.ndarray]:
        results = []
        for quantile in quantiles:
            q_values = []
            for i in range(vector_length):
                assert self._histograms[i] is not None
                assert self._bin_edges[i] is not None
                cumsum = np.cumsum(self._histograms[i])
                idx = np.searchsorted(cumsum, quantile * self._weight_sum[i], side="left")
                idx = min(int(idx), len(self._bin_edges[i]) - 2)
                q_values.append(self._bin_edges[i][idx])
            results.append(np.asarray(q_values, dtype=np.float64))
        return results


def resolve_output_path(config: _config.TrainConfig, data_config: _config.DataConfig) -> pathlib.Path:
    if data_config.asset_id is None:
        raise ValueError("Data config must have an asset_id.")

    if config.data.assets.assets_dir is not None:
        return pathlib.Path(config.data.assets.assets_dir).expanduser() / data_config.asset_id
    return config.assets_dirs / data_config.asset_id


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("Weights must be a non-empty 1D sequence.")
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")

    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return weights / total


def allocate_component_sampling_plans(
    dataset_lengths: Sequence[int],
    weights: Sequence[float],
    max_frames: int | None,
) -> list[ComponentSamplingPlan]:
    dataset_lengths = np.asarray(dataset_lengths, dtype=int)
    normalized_weights = normalize_weights(weights)

    if np.any(dataset_lengths <= 0):
        raise ValueError("All mixed datasets must contain at least one sample.")

    if max_frames is None or max_frames >= int(np.sum(dataset_lengths)):
        quotas = dataset_lengths.copy()
    else:
        desired = normalized_weights * max_frames
        quotas = np.minimum(np.floor(desired).astype(int), dataset_lengths)
        remaining = max_frames - int(np.sum(quotas))

        while remaining > 0:
            available = quotas < dataset_lengths
            if not np.any(available):
                break

            scores = np.full_like(normalized_weights, -np.inf, dtype=np.float64)
            scores[available] = desired[available] - quotas[available]
            if np.all(scores[available] <= 0):
                scores[available] = normalized_weights[available]

            quotas[int(np.argmax(scores))] += 1
            remaining -= 1

    return [
        ComponentSamplingPlan(
            quota=int(quota),
            per_sample_weight=(0.0 if quota == 0 else float(weight / quota)),
        )
        for quota, weight in zip(quotas, normalized_weights, strict=True)
    ]


def select_sample_indices(length: int, quota: int, seed: int, component_index: int) -> np.ndarray:
    if quota <= 0:
        return np.zeros(0, dtype=int)
    if quota >= length:
        return np.arange(length, dtype=int)

    rng = np.random.default_rng(np.random.SeedSequence([seed, component_index]))
    indices = rng.choice(length, size=quota, replace=False)
    return np.sort(indices.astype(int, copy=False))


def create_mixed_component_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
) -> _data_loader.Dataset:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")

    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    return _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )


def create_mixed_component_dataloader(
    dataset: _data_loader.Dataset,
    batch_size: int,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=_data_loader._collate_fn,  # noqa: SLF001
        worker_init_fn=_data_loader._worker_init_fn,  # noqa: SLF001
        drop_last=False,
    )


def compute_mixed_torch_norm_stats(
    data_config: _config.MixedDataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
    seed: int = 0,
) -> dict[str, normalize.NormStats]:
    datasets = [
        create_mixed_component_dataset(component, action_horizon, model_config) for component in data_config.components
    ]
    plans = allocate_component_sampling_plans(
        [len(dataset) for dataset in datasets],
        data_config.weights,
        max_frames,
    )

    keys = ["state", "actions"]
    stats = {key: WeightedRunningStats() for key in keys}

    for component_index, (dataset, plan) in enumerate(zip(datasets, plans, strict=True)):
        if plan.quota == 0:
            continue

        selected_dataset = dataset
        if plan.quota < len(dataset):
            indices = select_sample_indices(len(dataset), plan.quota, seed, component_index)
            selected_dataset = torch.utils.data.Subset(dataset, indices.tolist())

        data_loader = create_mixed_component_dataloader(selected_dataset, batch_size, num_workers)
        num_batches = math.ceil(plan.quota / batch_size)
        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]), sample_weight=plan.per_sample_weight)

    return {key: stats[key].get_statistics() for key in keys}


def main(config_name: str, max_frames: int | None = None, seed: int = 0):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
        keys = ["state", "actions"]
        stats = {key: normalize.RunningStats() for key in keys}

        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))

        norm_stats = {key: stats[key].get_statistics() for key in keys}
    elif isinstance(data_config, _config.MixedDataConfig):
        norm_stats = compute_mixed_torch_norm_stats(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            max_frames=max_frames,
            seed=seed,
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

        keys = ["state", "actions"]
        stats = {key: normalize.RunningStats() for key in keys}

        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))

        norm_stats = {key: stats[key].get_statistics() for key in keys}

    output_path = resolve_output_path(config, data_config)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
