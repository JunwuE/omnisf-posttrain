import dataclasses
import logging
from pathlib import Path

import torch
import tyro
from safetensors.torch import load_file, save_file


@dataclasses.dataclass
class Args:
    # Base pretrained weights in safetensors
    base_safetensors: str

    # Diff checkpoint in .pth (either {"state_dict": ...} or raw state_dict)
    diff_pth: str

    # Output safetensors path
    out_safetensors: str = "model_merged.safetensors"

    # final = base + scale * diff
    scale: float = 1.0

    # whether keys must match exactly
    strict_keys: bool = True  # use --strict-keys / --no-strict-keys

    # arithmetic dtype
    dtype: str = "fp32"  # fp32/fp16/bf16

    # compute device
    device: str = "cpu"  # cpu/cuda


def cast(t: torch.Tensor, dtype: str) -> torch.Tensor:
    if dtype == "fp32":
        return t.float()
    if dtype == "fp16":
        return t.half()
    if dtype == "bf16":
        return t.bfloat16()
    raise ValueError(f"Unknown dtype: {dtype}")


def load_diff_state_dict(path: str) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise RuntimeError(f"Unexpected diff format: {type(obj)}")

    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            raise RuntimeError(f"Diff contains non-tensor at key={k}: {type(v)}")
    return sd


def main(args: Args) -> None:
    logging.info("Loading base safetensors: %s", args.base_safetensors)
    base_sd = load_file(args.base_safetensors, device="cpu")  # dict[str, Tensor]

    logging.info("Loading diff pth: %s", args.diff_pth)
    diff_sd = load_diff_state_dict(args.diff_pth)

    keys_base = set(base_sd.keys())
    keys_diff = set(diff_sd.keys())

    if args.strict_keys:
        if keys_base != keys_diff:
            only_base = sorted(list(keys_base - keys_diff))[:30]
            only_diff = sorted(list(keys_diff - keys_base))[:30]
            raise RuntimeError(
                "Keys mismatch between base safetensors and diff.\n"
                f"Only in base (up to 30): {only_base}\n"
                f"Only in diff (up to 30): {only_diff}\n"
                "Use --no-strict-keys to apply on intersection only."
            )
        keys_apply = keys_base
    else:
        keys_apply = keys_base & keys_diff
        logging.warning("Non-strict mode: applying on intersection keys: %d", len(keys_apply))

    dev = torch.device(args.device)

    merged_sd: dict[str, torch.Tensor] = {}
    applied_float = 0
    skipped_nonfloat = 0
    skipped_missing = 0

    for k, base_t_cpu in base_sd.items():
        base_t = base_t_cpu  # already on cpu

        if k not in keys_apply:
            merged_sd[k] = base_t
            skipped_missing += 1
            continue

        diff_t_cpu = diff_sd[k]

        if base_t.shape != diff_t_cpu.shape:
            raise RuntimeError(f"Shape mismatch at key={k}: base {base_t.shape} vs diff {diff_t_cpu.shape}")

        # only add for floating-point tensors
        if base_t.is_floating_point() and diff_t_cpu.is_floating_point():
            a = cast(base_t.to(dev), args.dtype)
            d = cast(diff_t_cpu.to(dev), args.dtype)
            out = a + args.scale * d
            merged_sd[k] = out.to(base_t.dtype).detach().cpu()
            applied_float += 1
        else:
            merged_sd[k] = base_t
            skipped_nonfloat += 1

    out_path = Path(args.out_safetensors)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # safetensors 需要所有 tensor 在 CPU
    for k, v in merged_sd.items():
        if v.device.type != "cpu":
            merged_sd[k] = v.cpu()

    logging.info(
        "Done. applied_float=%d, skipped_nonfloat=%d, skipped_missing=%d",
        applied_float,
        skipped_nonfloat,
        skipped_missing,
    )
    logging.info("Saving merged safetensors to: %s", str(out_path))
    save_file(merged_sd, str(out_path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
