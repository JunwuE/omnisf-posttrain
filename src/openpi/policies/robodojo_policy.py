import dataclasses

import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms

_SOURCE_TO_TARGET = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
_TARGET_TO_SOURCE = _SOURCE_TO_TARGET.T
_ROBOT_BASE_MIDPOINT = np.array([0.0, -0.45, 0.765])  # The midpoint of the arm base in the source coordinate frame.


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def source_to_target_pose(pose: np.ndarray) -> np.ndarray:
    position = np.einsum("ij,...j->...i", _SOURCE_TO_TARGET, pose[..., :3] - _ROBOT_BASE_MIDPOINT)
    xyzw = np.concatenate([pose[..., 4:7], pose[..., 3:4]], axis=-1)
    source_rotation = Rotation.from_quat(xyzw.reshape(-1, 4)).as_matrix().reshape(*pose.shape[:-1], 3, 3)
    target_rotation = np.einsum("ij,...jk,kl->...il", _SOURCE_TO_TARGET, source_rotation, _TARGET_TO_SOURCE)
    rpy = Rotation.from_matrix(target_rotation.reshape(-1, 3, 3)).as_euler("xyz").reshape(*pose.shape[:-1], 3)
    return np.concatenate([position, rpy, pose[..., 7:8]], axis=-1)


def target_to_source_pose(pose: np.ndarray) -> np.ndarray:
    position = np.einsum("ij,...j->...i", _TARGET_TO_SOURCE, pose[..., :3]) + _ROBOT_BASE_MIDPOINT
    target_rotation = Rotation.from_euler(
        "xyz",
        pose[..., 3:6].reshape(-1, 3)
    ).as_matrix().reshape(*pose.shape[:-1], 3, 3)
    source_rotation = np.einsum("ij,...jk,kl->...il", _TARGET_TO_SOURCE, target_rotation, _SOURCE_TO_TARGET)
    xyzw = Rotation.from_matrix(source_rotation.reshape(-1, 3, 3)).as_quat().reshape(*pose.shape[:-1], 4)
    quaternion = np.concatenate([xyzw[..., 3:4], xyzw[..., :3]], axis=-1)
    return np.concatenate([position, quaternion, pose[..., 6:7]], axis=-1)


@dataclasses.dataclass(frozen=True)
class RobodojoInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        images = {
            "base_0_rgb": _parse_image(data["images"]["cam_high"]),
            "left_wrist_0_rgb": _parse_image(data["images"]["cam_left_wrist"]),
            "right_wrist_0_rgb": _parse_image(data["images"]["cam_right_wrist"]),
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }
        state = np.asarray(data["state"])
        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": np.concatenate(
                [
                    source_to_target_pose(state[..., :8]),
                    source_to_target_pose(state[..., 8:16]),
                ],
                axis=-1,
            ),
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = np.concatenate(
                [
                    source_to_target_pose(actions[..., :8]),
                    source_to_target_pose(actions[..., 8:16]),
                ],
                axis=-1,
            )

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RobodojoOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])[..., :14]
        return {
            "actions": np.concatenate(
                [
                    target_to_source_pose(actions[..., :7]),
                    target_to_source_pose(actions[..., 7:14]),
                ],
                axis=-1,
            )
        }
