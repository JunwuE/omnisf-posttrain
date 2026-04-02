import numpy as np
import torch._inductor.config as torch_inductor_cfg

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from openpi.policies import aloha_policy

torch_inductor_cfg.max_autotune = False


import debugpy
try:
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass


config = _config.get_config("pi0_aloha_place_one_floor_block")
checkpoint_dir = "/sharedata/lfh/openpi-main/checkpoints/pytorch_pi0_base"

# Create a trained policy (automatically detects PyTorch format)
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference (same API as JAX)
example = {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }

action_chunk = policy.infer(example)["actions"]