"""Training-cloud adapter for Ruan's frozen budget-matched wrapper.

The engine and wrapper under ``ruan_vendor`` are verbatim copies pinned by
SHA256.  This adapter only converts the batch tensor/numpy boundary and shapes
Ruan's per-sample metadata into the mask/seasons contract used by pipeline.py.
Formal four-scene evaluation remains Ruan's fixed evaluator; this adapter is
training-only.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

_VENDOR_ROOT = Path(__file__).resolve().parent / "ruan_vendor"
_EXPECTED_VENDOR_SHA256 = {
    "src/training_clouds.py": "6214cb52a8d1be82e84d3a7f8221cc6a066d11f2039eb8c8c4647baa15d9763e",
    "src/protocol.py": "ec81bc76211ae611eef5950dc278374817757f343626e590244194e2a3f797ff",
    "vendor/cloud_haze.py": "a6170460e07998798fa1549dff6f9572b07c205a7b473bcf32bb868ac7bea210",
}
for relative, expected in _EXPECTED_VENDOR_SHA256.items():
    actual = hashlib.sha256((_VENDOR_ROOT / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Vendored Ruan cloud source differs: {relative}")

if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from src.training_clouds import VERSION as TRAINING_WRAPPER_VERSION  # noqa: E402
from src.training_clouds import augment_sample as _augment_sample  # noqa: E402

_ALLOWED_SETTINGS = {"variant", "probability"}


def _json_meta(meta):
    """Keep only scalar metadata; masks are returned separately."""
    per_season = meta.get("per_season") or {}
    return {
        "training_protocol": meta.get("training_protocol"),
        "effective_max_shift": meta.get("effective_max_shift"),
        "pair_iou_min": meta.get("pair_iou_min"),
        "mask_sha256": meta.get("mask_sha256"),
        "coverage_jitter": meta.get("coverage_jitter"),
        "per_season": {
            str(season): {
                "coverage_actual": value.get("coverage_actual"),
                "n_cloud_pixels": value.get("n_cloud_pixels"),
                "shift": list(value.get("shift", (0, 0))),
            }
            for season, value in per_season.items()
        },
    }


def augment(*, image, sample_ids, kind, scope, seed, epoch, settings):
    """Apply Ruan's frozen wrapper to a batch without touching labels.

    ``kind`` is ``independent`` or ``overlap``; ``scope`` must be ``training``.
    Ruan's wrapper draws 2-4 seasons per sample and may leave a sample clean
    when the frozen probability is below 1.
    """
    if scope != "training":
        raise ValueError("Ruan cloud adapter is training-only; use Ruan's fixed evaluator")
    if kind not in ("independent", "overlap"):
        raise ValueError("Training cloud kind must be independent or overlap")
    if not isinstance(image, torch.Tensor) or image.ndim != 5 or tuple(image.shape[1:]) != (4, 128, 128, 10):
        raise ValueError("Cloud adapter expects image [B,4,128,128,10]")
    if image.dtype != torch.float32:
        raise ValueError("Cloud adapter expects float32 reflectance")
    if len(sample_ids) != image.shape[0]:
        raise ValueError("sample_ids length must match batch size")
    unknown = set(settings or {}) - _ALLOWED_SETTINGS
    if unknown:
        raise ValueError(f"Unsupported cloud settings for Ruan wrapper: {sorted(unknown)}")
    variant = (settings or {}).get("variant", "additive")
    probability = float((settings or {}).get("probability", 0.5))
    if variant != "additive":
        raise ValueError("The frozen pilot protocol uses variant=additive")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0,1]")

    affected, masks, seasons_per_sample, applied, provider_meta = [], [], [], [], []
    for i, sample_id in enumerate(sample_ids):
        source = image[i].detach().cpu().numpy()
        output, meta = _augment_sample(
            source, int(sample_id), int(epoch), seed=int(seed), mode=kind,
            variant=variant, probability=probability,
        )
        if output.shape != source.shape or not np.isfinite(output).all():
            raise ValueError("Ruan wrapper returned an invalid image")
        affected.append(output.astype(np.float32, copy=False))

        is_applied = bool(meta.get("applied", False))
        binary_masks = meta.get("binary_masks") or {}
        seasons = sorted(int(s) for s in binary_masks)
        if is_applied:
            if not 2 <= len(seasons) <= 4:
                raise ValueError("Ruan wrapper must apply 2-4 seasons")
        elif seasons:
            raise ValueError("Ruan wrapper returned seasons for an unapplied sample")
        sample_mask = np.zeros((4, 128, 128), dtype=bool)
        for season in seasons:
            sample_mask[season] = np.asarray(binary_masks[season], dtype=bool)
        masks.append(sample_mask)
        seasons_per_sample.append(seasons)
        applied.append(is_applied)
        provider_meta.append(_json_meta(meta))

    result = torch.from_numpy(np.stack(affected, axis=0)).to(device=image.device, dtype=image.dtype)
    mask = torch.from_numpy(np.stack(masks, axis=0)).to(device=image.device, dtype=torch.bool)
    return result, mask, {
        "kind": kind,
        "scope": scope,
        "applied": applied,
        "seasons_per_sample": seasons_per_sample,
        "seasons": sorted({s for seasons in seasons_per_sample for s in seasons}),
        "training_wrapper_version": TRAINING_WRAPPER_VERSION,
        "samples": provider_meta,
    }
