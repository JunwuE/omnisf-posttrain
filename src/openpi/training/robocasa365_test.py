"""Exercise native configs, LeRobot video loading, norm stats and model inputs.

Only the language tokenizer is stubbed to keep these regression tests offline.
Parquet metadata and videos are written/read with the installed LeRobot library.
"""

import dataclasses
import functools
import json

from lerobot.common.datasets import lerobot_dataset
import numpy as np
import pytest

from openpi import transforms
from openpi.shared import normalize
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training.misc import robocasa365_config
from scripts import compute_norm_stats


class _OfflineTokenizer:
    def __init__(self, max_len):
        self.max_len = max_len

    def tokenize(self, prompt, state=None, *, use_discrete_state_format=None):
        assert prompt == "open the cabinet"
        assert state is None or state.shape == (32,)
        assert use_discrete_state_format
        return np.ones(self.max_len, dtype=np.int32), np.ones(self.max_len, dtype=bool)


@pytest.fixture
def recipes(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOCASA365_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROBOCASA365_ASSETS_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("ROBOCASA365_MIXED_WEIGHTS", "2,1,1")
    monkeypatch.setattr(config_lib._tokenizer, "PaligemmaTokenizer", _OfflineTokenizer)  # noqa: SLF001
    return {
        cfg.name: dataclasses.replace(
            cfg, model=dataclasses.replace(cfg.model, action_horizon=2), batch_size=2, num_workers=0
        )
        for cfg in robocasa365_config.get_robocasa365_configs()
    }


@pytest.fixture
def local_datasets(recipes, monkeypatch):
    # H.264 makes tiny test videos cheap to encode and exercises real video decoding.
    monkeypatch.setattr(
        lerobot_dataset, "encode_video_frames", functools.partial(lerobot_dataset.encode_video_frames, vcodec="h264")
    )
    camera_keys = [
        "observation.images.robot0_agentview_left",
        "observation.images.robot0_eye_in_hand",
        "observation.images.robot0_agentview_right",
    ]
    features = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": None},
        "action": {"dtype": "float32", "shape": (12,), "names": None},
        **{key: {"dtype": "video", "shape": (3, 32, 48), "names": ["c", "h", "w"]} for key in camera_keys},
    }
    for split_index, split in enumerate(robocasa365_config.SPLIT_DATASET_NAMES):
        cfg = recipes[f"robocasa365_target_{split}"]
        dataset = lerobot_dataset.LeRobotDataset.create(
            repo_id=f"test/{split}", root=cfg.data.repo_id, fps=10, features=features
        )
        for frame in range(4):
            dataset.add_frame(
                {
                    "observation.state": np.arange(16, dtype=np.float32) + frame + split_index * 10,
                    "action": np.arange(12, dtype=np.float32) / 20 + frame / 10 + split_index,
                    "task": "open the cabinet",
                    **{
                        key: np.full((32, 48, 3), (index + 1) * 50, dtype=np.uint8)
                        for index, key in enumerate(camera_keys)
                    },
                }
            )
        dataset.save_episode()
    return recipes


@pytest.mark.parametrize("split", [*robocasa365_config.SPLIT_DATASET_NAMES, "mixed"])
def test_builtin_registry(split):
    cfg = config_lib.get_config(f"robocasa365_target_{split}")
    assert cfg.model.pi05
    assert cfg.model.max_token_len == 200
    assert cfg.align_enabled
    assert cfg.align_loss_coeff == 0.5


@pytest.mark.parametrize("split", ["atomic", "mixed"])
@pytest.mark.parametrize("fast", [False, True])
def test_native_norm_stats_and_training_batch(local_datasets, monkeypatch, split, fast):
    cfg = local_datasets[f"robocasa365_target_{split}"]
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, fast_local_loading=fast))
    monkeypatch.setitem(config_lib._CONFIGS_DICT, cfg.name, cfg)  # noqa: SLF001
    compute_norm_stats.main(cfg.name, batch_size=2, num_workers=0)
    stats = normalize.load(cfg.data.assets.assets_dir + "/" + cfg.name)
    # The mixed sampler and shared statistics both weight frames by (2, 1, 1).
    expected_offset = 1.5 if split == "atomic" else 9.0
    np.testing.assert_allclose(stats["state"].mean[:16], np.arange(16) + expected_offset)
    np.testing.assert_array_equal(stats["state"].mean[16:], 0)
    assert stats["actions"].q01.shape == (32,)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    if split == "mixed":
        assert tuple(data_config.weights) == (2.0, 1.0, 1.0)
        for component in data_config.components:
            np.testing.assert_array_equal(component.norm_stats["state"].mean, stats["state"].mean)

    loader = data_loader.create_data_loader(cfg, num_batches=1, shuffle=False, return_host_batch=True)
    (obs, actions), (host_obs, host_actions) = next(iter(loader))
    np.testing.assert_allclose(actions, host_actions, atol=1e-7)
    assert obs.state.shape == (2, 32)
    assert actions.shape == (2, 2, 32)
    assert obs.tokenized_prompt.shape == (2, 200)
    for camera, image in obs.images.items():
        assert image.shape == (2, 224, 224, 3)
        np.testing.assert_array_equal(image, host_obs.images[camera])
        assert np.asarray(obs.image_masks[camera]).all()
        assert obs.image_padding_mask[camera].shape == (2, 224, 224)
        assert not np.asarray(obs.image_padding_mask[camera]).all()  # Rectangular input is letterboxed.

    if split == "atomic":
        sample = data_loader.create_transformed_single_torch_dataset(data_config, 2, cfg.model)[0]
        restored = transforms.Unnormalize(stats, use_quantiles=True)(sample)
        result = transforms.compose(data_config.data_transforms.outputs)(restored)
        expected_actions = np.arange(12, dtype=np.float32)[None, :] / 20 + np.array([[0], [0.1]])
        np.testing.assert_allclose(result["actions"], expected_actions, atol=1e-6)


def test_local_relative_path_and_v3_rejection(local_datasets, monkeypatch, tmp_path):
    cfg = local_datasets["robocasa365_target_atomic"]
    monkeypatch.chdir(tmp_path)
    data = dataclasses.replace(cfg.data.create(cfg.assets_dirs, cfg.model), repo_id="./data/robocasa365-target-atomic")
    dataset = data_loader.create_torch_dataset(data, 2, cfg.model)
    assert dataset[0]["prompt"] == "open the cabinet"
    relative_data = dataclasses.replace(data, repo_id="data/robocasa365-target-atomic", fast_local_loading=True)
    assert data_loader.create_torch_dataset(relative_data, 2, cfg.model)[0]["prompt"] == "open the cabinet"
    info_path = tmp_path / "data/robocasa365-target-atomic/meta/info.json"
    info = json.loads(info_path.read_text())
    info["codebase_version"] = "v3.0"
    info_path.write_text(json.dumps(info))
    for fast in (False, True):
        with pytest.raises(ValueError, match=r"v?3\.0"):
            data_loader.create_torch_dataset(dataclasses.replace(data, fast_local_loading=fast), 2, cfg.model)


@pytest.mark.parametrize("weights", ["1,2", "1,0,1", "nan,1,1", "inf,1,1"])
def test_rejects_invalid_mixture_weights(monkeypatch, weights):
    monkeypatch.setenv("ROBOCASA365_MIXED_WEIGHTS", weights)
    with pytest.raises(ValueError, match="ROBOCASA365_MIXED_WEIGHTS"):
        robocasa365_config.get_robocasa365_configs()


def test_missing_local_dataset_and_remote_fast_mode_fail_early(tmp_path):
    cfg = config_lib.get_config("robocasa365_target_atomic")
    with pytest.raises(FileNotFoundError, match=r"meta/info\.json"):
        data_loader.create_torch_dataset(config_lib.DataConfig(repo_id=str(tmp_path / "missing")), 2, cfg.model)
    with pytest.raises(ValueError, match="requires a local"):
        data_loader.create_torch_dataset(
            config_lib.DataConfig(repo_id="test/remote", fast_local_loading=True), 2, cfg.model
        )
