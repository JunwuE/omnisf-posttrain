from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import dataclasses
import os
import time
from typing import Any, Literal

import numpy as np
from safetensors.torch import load_file
import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models import model as _model
from vggt.heads.utils import custom_pooling
from vggt.utils.load_fn import preprocess_images_from_openpi

AlignTargetModel = Literal["auto", "vggt", "omnivggt"]


@dataclasses.dataclass
class AlignFeatures:
    targets: np.ndarray
    mask: np.ndarray
    compute_s: float = 0.0


@dataclasses.dataclass(frozen=True)
class AlignDeviceLayout:
    pi_device_ids: list[int]
    vggt_device_ids: list[int]


def infer_target_model(config: Any) -> Literal["vggt", "omnivggt"]:
    target = config.align_target_model
    weight_path = config.vggt_weight_path
    if target in ("vggt", "omnivggt"):
        return target
    elif "omnivggt" in weight_path.lower() and target == "auto":
        return "omnivggt"
    elif "vggt" in weight_path.lower() and target == "auto":
        return "vggt"
    else:
        raise ValueError(f"Invalid align_target_model {target} or vggt_weight_path {weight_path}")


def resolve_align_devices(config: Any) -> AlignDeviceLayout:
    count = torch.cuda.device_count()
    vggt_devices = config.align_vggt_devices if count > 1 else 1
    vggt_devices = max(1, min(vggt_devices, count))
    mode = config.align_device_mode
    if mode == "shared" or count == 1:
        return AlignDeviceLayout(
            pi_device_ids=list(range(count)),
            vggt_device_ids=list(range(count - vggt_devices, count)),
        )
    elif mode == "reserved":
        return AlignDeviceLayout(
            pi_device_ids=list(range(count - vggt_devices)),
            vggt_device_ids=list(range(count - vggt_devices, count)),
        )
    else:
        raise ValueError(f"Invalid align_device_mode {mode}. Must be one of ['shared', 'reserved'].")


def _split_batch_ranges(batch_size: int, shard_count: int) -> list[tuple[int, int]]:
    boundaries = np.linspace(0, batch_size, shard_count + 1, dtype=int)
    return [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(shard_count)]


def _concat_align_features(chunks: list[AlignFeatures]) -> AlignFeatures:
    return AlignFeatures(
        targets=np.concatenate([chunk.targets for chunk in chunks], axis=0),
        mask=np.concatenate([chunk.mask for chunk in chunks], axis=0),
        compute_s=max(chunk.compute_s for chunk in chunks),
    )


class _TorchAlignFeatureShard:
    def __init__(self, config: Any, *, device_id: int):
        self.config = config
        self.target_model = infer_target_model(config)
        self.device = torch.device(f"cuda:{device_id}")

        self.model = self._load_model().to(self.device)
        self.model.eval()

    def _load_model(self):
        if self.target_model == "omnivggt":
            from omnivggt.models.omnivggt import OmniVGGT

            model = OmniVGGT(
                enable_camera=False,
                enable_point=False,
                enable_depth=False,
                feature_only=True,
            )
            vggt_ckpt_path = os.path.join(self.config.vggt_weight_path, "OmniVGGT.safetensors")
            if not os.path.exists(vggt_ckpt_path):
                raise FileNotFoundError(f"OmniVGGT weight file not found at {vggt_ckpt_path}")
            model.load_state_dict(load_file(vggt_ckpt_path), strict=False)
        else:
            from vggt.models.vggt import VGGT
            model = VGGT(
                enable_camera=False,
                enable_point=False,
                enable_depth=False,
                enable_track=False,
                feature_only=True,
            )
            vggt_ckpt_path = os.path.join(self.config.vggt_weight_path, "model.pt")
            if not os.path.exists(vggt_ckpt_path):
                raise FileNotFoundError(f"VGGT weight file not found at {vggt_ckpt_path}")
            model.load_state_dict(torch.load(vggt_ckpt_path), strict=False)

        return model

    def close(self) -> None:
        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def __call__(self, observation: _model.Observation, *, reference_tokens: int) -> AlignFeatures:
        start_s = time.perf_counter()
        chunk_size = self.config.align_feature_batch_size
        batch_size = next(iter(observation.images.values())).shape[0]
        if chunk_size is not None and chunk_size < batch_size:
            chunks = []
            for start in range(0, batch_size, chunk_size):
                stop = min(start + chunk_size, batch_size)
                chunks.append(self._extract_features(_slice_observation_batch(observation, start, stop), reference_tokens))
            features = _concat_align_features(chunks)
        else:
            features = self._extract_features(observation, reference_tokens)
        return dataclasses.replace(features, compute_s=time.perf_counter() - start_s)

    def _extract_features(self, observation: _model.Observation, reference_tokens: int) -> AlignFeatures:
        with torch.cuda.device(self.device) if self.device.type == "cuda" else contextlib.nullcontext():
            images, img_padding_mask, img_masks = self._prepare_images(observation)
            # from torchvision.utils import save_image; import matplotlib.cm as cm; import numpy as np; from PIL import Image
            # save_image(original_img[0,0], 'image.png')

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                if self.target_model == "omnivggt" and self.config.use_camera_params:
                    extrinsics, intrinsics, camera_param_mask = self._prepare_camera_params(
                        observation, list(observation.images)
                    )
                    vggt_images, intrinsics = preprocess_images_from_openpi(images, intrinsics)
                    output = self.model(
                        images=vggt_images,
                        extrinsics=extrinsics,
                        intrinsics=intrinsics,
                        camera_gt_index=camera_param_mask,
                    )
                else:
                    vggt_images = preprocess_images_from_openpi(images)
                    output = self.model(vggt_images)

            features = output["features"]
            patch_start_idx = output["patch_start_idx"]
            original_img = output["images"]
            vggt_hidden = [hidden[:, :, patch_start_idx:, :] for hidden in features]

            h, w = original_img.shape[-2:]
            patch_h, patch_w = h // self.model.patch_size, w // self.model.patch_size
            reference = torch.zeros(
                (vggt_images.shape[0], reference_tokens, 1),
                dtype=vggt_hidden[0].dtype,
                device=self.device,
            )
            pooled_hidden = [
                custom_pooling(
                    hidden,
                    (patch_h, patch_w),
                    (h, w),
                    reference,
                    self.config.pooling_func,
                    self.config.use_vggt_pe,
                )
                for hidden in vggt_hidden
            ]
            target = pooled_hidden[self.config.vggt_layers_align].float()
            mask = self._make_align_mask(img_masks, img_padding_mask, reference_tokens)
            return AlignFeatures(
                targets=target.detach().cpu().numpy().astype(np.float32),
                mask=mask.detach().cpu().numpy().astype(np.bool_),
            )

    def _prepare_images(self, observation: _model.Observation) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], list[torch.Tensor]]:
        images = []
        padding_masks = {}
        image_masks = []
        for key, image in observation.images.items():
            image_t = torch.as_tensor(image, device=self.device)
            if image_t.ndim != 4:
                raise ValueError(f"Expected image {key} to have shape [B,H,W,C], got {image_t.shape}")
            image_t = image_t.to(torch.float32)
            image_t = torch.clamp(image_t / 2.0 + 0.5, 0, 1)

            padding_mask = torch.as_tensor(observation.image_padding_mask[key], device=self.device).bool()
            image_t = image_t.clone() if not (image == 0).all() else torch.ones_like(image_t)  # TODO: 用仅双视角做检查
            image_t[~padding_mask] = 1.0
            images.append(image_t.permute(0, 3, 1, 2).contiguous())
            padding_masks[key] = padding_mask
            image_masks.append(torch.as_tensor(observation.image_masks[key], device=self.device).bool())
        return images, padding_masks, image_masks

    def _prepare_camera_params(self, observation: _model.Observation, image_keys: list[str]):
        if not observation.camera_param:
            return None, None, None
        extrinsics = torch.stack(
            [
                torch.as_tensor(observation.camera_param[f"{key}_extrinsics"], device=self.device).float()
                for key in image_keys
            ],
            dim=1,
        )
        intrinsics = torch.stack(
            [
                torch.as_tensor(observation.camera_param[f"{key}_intrinsics"], device=self.device).float()
                for key in image_keys
            ],
            dim=1,
        )
        camera_param_mask = [
            idx
            for idx, key in enumerate(image_keys)
            if observation.camera_param_mask
            and key in observation.camera_param_mask
            and bool(observation.camera_param_mask[key].all())
        ]
        return extrinsics, intrinsics, camera_param_mask

    def _make_align_mask(
        self,
        img_masks: list[torch.Tensor],
        img_padding_mask: dict[str, torch.Tensor],
        reference_tokens: int,
    ) -> torch.Tensor:
        # empty image feature masks for alignment loss
        tokens_per_img = reference_tokens // len(img_masks)
        mask = torch.repeat_interleave(torch.stack(img_masks, dim=1), repeats=tokens_per_img, dim=1)

        # useless image padding feature masks for alignment loss
        if self.config.ignore_img_padding_area:
            padding = torch.stack(list(img_padding_mask.values()), dim=1)
            target_size = padding.shape[-1] // 14
            mask_downsampled = F.interpolate(
                padding.float(),
                size=(target_size, target_size),
                mode="nearest",
            ).bool().flatten(start_dim=1)
            mask = mask & mask_downsampled
        return mask


class TorchAlignFeatureExtractor:
    def __init__(self, config: Any, *, devices: list[int]):
        self.config = config
        # if torch.cuda.is_available():
        #     torch.backends.cuda.matmul.allow_tf32 = True
        #     torch.backends.cudnn.allow_tf32 = True
        self.shards = [_TorchAlignFeatureShard(config, device_id=device_id) for device_id in devices]
        self.executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=len(self.shards)) if len(self.shards) > 1 else None
        )

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)
        for shard in self.shards:
            shard.close()

    def __call__(self, observation: _model.Observation, *, reference_tokens: int) -> AlignFeatures:
        start_s = time.perf_counter()
        if len(self.shards) == 1:
            features = self.shards[0](observation, reference_tokens=reference_tokens)
            return dataclasses.replace(features, compute_s=time.perf_counter() - start_s)

        batch_size = next(iter(observation.images.values())).shape[0]
        ranges = _split_batch_ranges(batch_size, len(self.shards))
        futures = []
        for shard, (start, stop) in zip(self.shards, ranges, strict=True):
            if start == stop:
                continue
            futures.append(
                self.executor.submit(
                    shard,
                    _slice_observation_batch(observation, start, stop),
                    reference_tokens=reference_tokens,
                )
            )
        features = _concat_align_features([future.result() for future in futures])
        return dataclasses.replace(features, compute_s=time.perf_counter() - start_s)


class AsyncAlignFeatureWorker:
    def __init__(self, extractor: TorchAlignFeatureExtractor):
        self.extractor = extractor
        # TODO max_workers=1 is a temporary workaround 
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.futures: collections.deque[concurrent.futures.Future[AlignFeatures]] = collections.deque()

    def submit(self, observation: _model.Observation, *, reference_tokens: int) -> None:
        self.futures.append(self.executor.submit(self.extractor, observation, reference_tokens=reference_tokens))

    def result(self) -> AlignFeatures:
        return self.futures.popleft().result()

    def pending_count(self) -> int:
        return len(self.futures)

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        self.extractor.close()


def _slice_observation_batch(observation: _model.Observation, start: int, stop: int) -> _model.Observation:
    return _model.Observation(
        images={key: value[start:stop] for key, value in observation.images.items()},
        image_masks={key: value[start:stop] for key, value in observation.image_masks.items()},
        state=observation.state[start:stop],
        image_padding_mask=None
        if observation.image_padding_mask is None
        else {key: value[start:stop] for key, value in observation.image_padding_mask.items()},
        tokenized_prompt=None if observation.tokenized_prompt is None else observation.tokenized_prompt[start:stop],
        tokenized_prompt_mask=None
        if observation.tokenized_prompt_mask is None
        else observation.tokenized_prompt_mask[start:stop],
        token_ar_mask=None if observation.token_ar_mask is None else observation.token_ar_mask[start:stop],
        token_loss_mask=None if observation.token_loss_mask is None else observation.token_loss_mask[start:stop],
        camera_param=None
        if observation.camera_param is None
        else {key: value[start:stop] for key, value in observation.camera_param.items()},
        camera_param_mask=None
        if observation.camera_param_mask is None
        else {key: value[start:stop] for key, value in observation.camera_param_mask.items()},
        index=None if observation.index is None else observation.index[start:stop],
    )
