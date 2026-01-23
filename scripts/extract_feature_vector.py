# scripts/diff_ckpt.py
import dataclasses
import logging
from pathlib import Path
from typing import Any

import torch
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class CkptSpec:
    config: str
    dir: str


@dataclasses.dataclass
class Args:
    a: CkptSpec
    b: CkptSpec
    out: str = "checkpoints/diff/a_minus_b.pth"
    only_vlm: bool = False
    strict_keys: bool = False          # True: keys 必须一致，否则报错；False: 只对交集 keys 做差
    dtype: str = "fp32"               # fp32 / fp16 / bf16
    device: str = "cpu"               # cpu / cuda


def _extract_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    """
    Try best to get a torch state_dict from a Policy or Module-like object.
    """
    # Case 1: policy itself has state_dict()
    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        sd = obj.state_dict()
        if isinstance(sd, dict) and all(isinstance(v, torch.Tensor) for v in sd.values()):
            return sd

    # Case 2: common attributes that hold torch.nn.Module
    for attr in ["model", "_model", "module", "net", "_net", "policy", "_policy"]:
        if hasattr(obj, attr):
            m = getattr(obj, attr)
            if hasattr(m, "state_dict") and callable(m.state_dict):
                sd = m.state_dict()
                if isinstance(sd, dict) and all(isinstance(v, torch.Tensor) for v in sd.values()):
                    return sd

    raise RuntimeError(
        "Cannot extract state_dict. "
        "Please inspect Policy object and update attribute list in _extract_state_dict()."
    )


def _cast_tensor(t: torch.Tensor, dtype: str) -> torch.Tensor:
    if dtype == "fp32":
        return t.float()
    if dtype == "fp16":
        return t.half()
    if dtype == "bf16":
        return t.bfloat16()
    raise ValueError(f"Unknown dtype: {dtype}")


def load_policy(spec: CkptSpec) -> _policy.Policy:
    cfg = _config.get_config(spec.config)
    return _policy_config.create_trained_policy(cfg, spec.dir)


def main(args: Args) -> None:
    logging.info("Loading A: config=%s dir=%s", args.a.config, args.a.dir)
    pol_a = load_policy(args.a)
    logging.info("Loading B: config=%s dir=%s", args.b.config, args.b.dir)
    pol_b = load_policy(args.b)

    sd_a = _extract_state_dict(pol_a)
    sd_b = _extract_state_dict(pol_b)

    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())

    if args.strict_keys:
        if keys_a != keys_b:
            only_a = sorted(list(keys_a - keys_b))[:20]
            only_b = sorted(list(keys_b - keys_a))[:20]
            raise RuntimeError(
                f"State dict keys mismatch.\n"
                f"Only in A (show up to 20): {only_a}\n"
                f"Only in B (show up to 20): {only_b}\n"
                f"Set --strict-keys False to subtract intersection only."
            )
        keys = sorted(keys_a)
    else:
        keys = sorted(list(keys_a & keys_b))
        logging.warning("Non-strict mode: subtracting only intersection keys: %d", len(keys))

    device = torch.device(args.device)
    diff: dict[str, torch.Tensor] = {}

    if args.only_vlm:
        ZERO_PREFIXES = [
            "paligemma_with_expert.gemma_expert.",
            "action_in_proj.",
            "action_out_proj."
            "action_time_mlp_in",
            "action_time_mlp_oout",
            # 你还可以在这里加别的，比如：
            # "some_other_expert.",
        ]
    else:
        ZERO_PREFIXES = []

    for k in keys:
        ta = sd_a[k].to(device)
        tb = sd_b[k].to(device)

        if ta.shape != tb.shape:
            raise RuntimeError(f"Shape mismatch at key={k}: {ta.shape} vs {tb.shape}")

        # 是否需要置零
        zero_this = any(k.startswith(p) for p in ZERO_PREFIXES)

        if zero_this:
            out = torch.zeros_like(ta)
        else:
            # 只对浮点权重做差；非浮点（比如 int 的计数器）直接复制 A
            if ta.is_floating_point():
                out = _cast_tensor(ta, args.dtype) - _cast_tensor(tb, args.dtype)
            else:
                out = ta

        diff[k] = out.detach().cpu()



    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": diff, "a": dataclasses.asdict(args.a), "b": dataclasses.asdict(args.b)}, out_path)
    logging.info("Saved diff checkpoint to: %s", str(out_path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
