"""RoboCasa365's mobile manipulation interface: 16-D state and 12-D actions.

Unlike RoboCasaNoMovement, this interface retains the base pose, base commands
and control mode. Controller deltas and gripper commands are used as stored;
there is no rotation conversion, rescaling or additional delta-action transform.
"""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    return image


@dataclasses.dataclass(frozen=True)
class RoboCasa365Inputs(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        if self.action_dim < 16:
            raise ValueError("RoboCasa365 requires model action_dim >= 16 to retain the full state.")
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape[-1] != 16:
            raise ValueError(f"RoboCasa365 expects a 16-D state, got {state.shape}.")

        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        if self.model_type == _model.ModelType.PI05:
            if "observation/right_image" not in data:
                raise ValueError("RoboCasa365 pi0.5 requires observation/right_image.")
            right_image = _parse_image(data["observation/right_image"])
            right_mask = np.True_
        else:
            right_image = np.zeros_like(base_image)
            right_mask = np.bool_(self.model_type != _model.ModelType.PI0)

        inputs = {
            "state": transforms.pad_to_dim(state, self.action_dim),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_mask,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != 12:
                raise ValueError(f"RoboCasa365 expects 12-D actions, got {actions.shape}.")
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)
        for key in ("prompt", "index"):
            if key in data:
                inputs[key] = data[key]
        return inputs


@dataclasses.dataclass(frozen=True)
class RoboCasa365Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :12]}
