import numpy as np
import pytest

from openpi.models import model
from openpi.policies import robocasa365_policy as policy


def example():
    return {
        "observation/state": np.arange(16, dtype=np.float32),
        "observation/image": np.full((8, 12, 3), 51, dtype=np.uint8),
        "observation/wrist_image": np.full((3, 8, 12), 0.4, dtype=np.float32),
        "observation/right_image": np.full((3, 8, 12), 153, dtype=np.uint8),
        "actions": np.arange(48, dtype=np.float32).reshape(4, 12) / 50 - 0.5,
        "prompt": "open the cabinet",
    }


def test_full_mobile_controls_and_camera_mapping():
    data = example()
    result = policy.RoboCasa365Inputs(32)(data)
    np.testing.assert_array_equal(result["state"][:16], data["observation/state"])
    np.testing.assert_array_equal(result["state"][16:], 0)
    np.testing.assert_array_equal(result["actions"][..., :12], data["actions"])
    np.testing.assert_array_equal(result["actions"][..., 12:], 0)
    for camera, pixel in zip(result["image"], (51, 102, 153), strict=True):
        assert result["image"][camera].shape == (8, 12, 3)
        assert result["image"][camera].dtype == np.uint8
        np.testing.assert_array_equal(result["image"][camera], pixel)
        assert result["image_mask"][camera]
    assert result["prompt"] == data["prompt"]
    np.testing.assert_array_equal(policy.RoboCasa365Outputs()(result)["actions"], data["actions"])
    batched = policy.RoboCasa365Outputs()({"actions": result["actions"][None]})
    assert batched["actions"].shape == (1, 4, 12)


def test_inference_without_actions_and_required_right_camera():
    data = example()
    del data["actions"]
    assert "actions" not in policy.RoboCasa365Inputs(32)(data)
    del data["observation/right_image"]
    with pytest.raises(ValueError, match="right_image"):
        policy.RoboCasa365Inputs(32)(data)
    result = policy.RoboCasa365Inputs(32, model.ModelType.PI0)(data)
    assert not result["image_mask"]["right_wrist_0_rgb"]


@pytest.mark.parametrize(("key", "width"), [("observation/state", 7), ("actions", 7)])
def test_rejects_no_movement_schema(key, width):
    data = example()
    data[key] = data[key][..., :width]
    with pytest.raises(ValueError, match="expects"):
        policy.RoboCasa365Inputs(32)(data)


def test_rejects_model_that_would_drop_state_dimensions():
    with pytest.raises(ValueError, match="action_dim >= 16"):
        policy.RoboCasa365Inputs(12)(example())
