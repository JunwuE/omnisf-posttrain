"""Built-in RoboCasa365 post-training recipes for scripts/train_align.py."""

import dataclasses
import math
import os
import pathlib
from typing import TYPE_CHECKING

from openpi.models import pi0_config
from openpi.training import optimizer
from openpi.training import weight_loaders

if TYPE_CHECKING:
    from openpi.training.config import TrainConfig


SPLIT_DATASET_NAMES = {
    "atomic": "robocasa365-target-atomic",
    "composite_seen": "robocasa365-target-composite-seen",
    "composite_unseen": "robocasa365-target-composite-unseen",
}


def get_robocasa365_configs() -> list["TrainConfig"]:
    # Config calls this after defining its dataclasses, like roboarena_config.
    from openpi.training import config as config_lib  # noqa: PLC0415

    root = pathlib.Path(__file__).resolve().parents[4]
    data_root = pathlib.Path(os.environ.get("ROBOCASA365_DATA_ROOT", root / "data/robocasa365_v21")).expanduser()
    assets_dir = os.environ.get("ROBOCASA365_ASSETS_DIR", str(root / "assets"))
    checkpoint_dir = os.environ.get("ROBOCASA365_CHECKPOINT_DIR", str(root / "checkpoints"))
    pretrained = os.environ.get(
        "ROBOCASA365_PRETRAIN_PARAMS", str(root / "checkpoints/omnisf_9.0_jax/experiments/35999/params")
    )
    vggt_path = os.environ.get("ROBOCASA365_VGGT_PATH", str(root / "checkpoints/omnivggt"))
    weights = tuple(float(value) for value in os.environ.get("ROBOCASA365_MIXED_WEIGHTS", "1,1,1").split(","))
    if len(weights) != 3 or any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("ROBOCASA365_MIXED_WEIGHTS must contain three finite, positive values (e.g. 2,1,1).")

    def data_factory(split: str) -> config_lib.LeRobotRoboCasa365DataConfig:
        return config_lib.LeRobotRoboCasa365DataConfig(
            repo_id=str((data_root / SPLIT_DATASET_NAMES[split]).resolve()),
            base_config=config_lib.DataConfig(prompt_from_task=True),
            use_quantile_norm=True,
        )

    configs = []
    for split in (*SPLIT_DATASET_NAMES, "mixed"):
        name = f"robocasa365_target_{split}"
        assets = config_lib.AssetsConfig(assets_dir=str(pathlib.Path(assets_dir).expanduser()), asset_id=name)
        if split == "mixed":
            data = config_lib.MixedLeRobotDataConfig(
                repo_id="robocasa365-target-mixed",
                assets=assets,
                sources=tuple(
                    config_lib.MixedDataSourceConfig(name=source, weight=weight, data_factory=data_factory(source))
                    for source, weight in zip(SPLIT_DATASET_NAMES, weights, strict=True)
                ),
                use_quantile_norm=True,
            )
        else:
            data = dataclasses.replace(data_factory(split), assets=assets)
        model = pi0_config.Pi0Config(pi05=True)
        configs.append(
            config_lib.TrainConfig(
                name=name,
                project_name="robocasa365_omnisf",
                model=model,
                data=data,
                state_dropout_prob=0.25,
                freeze_filter=model.get_freeze_filter(),
                weight_loader=weight_loaders.CheckpointWeightLoader(str(pathlib.Path(pretrained).expanduser())),
                assets_base_dir=assets_dir,
                checkpoint_base_dir=str(pathlib.Path(checkpoint_dir).expanduser()),
                num_train_steps=20_000,
                save_interval=20_000,
                keep_period=None,
                # Start with one GPU; the README gives the reference 8-GPU recipe.
                batch_size=1,
                gradient_accumulation_steps=2,
                num_workers=2,
                fsdp_devices=1,
                ema_decay=0.999,
                lr_schedule=optimizer.CosineDecaySchedule(
                    warmup_steps=1000, peak_lr=2.5e-5, decay_steps=100_000, decay_lr=2.5e-6
                ),
                optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
                align_enabled=True,
                align_target_model="auto",
                vggt_weight_path=str(pathlib.Path(vggt_path).expanduser()),
                vla_layers_align=12,
                vla_adapt_layers_align=False,
                vggt_layers_align=-1,
                vggt_adapt_layers_align=False,
                pooling_func="bilinear",
                use_vggt_pe=True,
                use_vlm_norm=True,
                align_loss_coeff=0.5,
                ignore_img_padding_area=True,
                use_camera_params=False,
                align_device_mode="shared",
                align_vggt_devices=1,
                align_feature_batch_size=1,
            )
        )
    return configs
