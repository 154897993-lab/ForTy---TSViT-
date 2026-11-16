"""Training-cloud adapter v1.2: additive + multiplicative with one sampling stream.

The frozen v1.1 wrapper exposes only ``variant='additive'``.  For B_mul we
first ask that wrapper for its exact per-sample draws (probability, seasons,
coverage, level, cloud seed and overlap shift), then call the unchanged
vendored engine's ``multiplicative`` degradation with those same draws.  This
keeps B_ind and B_mul matched on every parameter except the declared variant.
The vendored engine/wrapper sources are never modified.
"""
import hashlib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
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

from src.training_clouds import ENGINE, VERSION as TRAINING_WRAPPER_VERSION  # noqa: E402
from src.training_clouds import augment_sample as _additive_sample  # noqa: E402

ADAPTER_VERSION = "ruan-cloud-adapter-v1.2-multiplicative"
_ALLOWED_SETTINGS = {"variant", "probability", "cpu_workers"}
_POOLS = {}


def _augment_job(arguments):
    source, sample_id, epoch, seed, kind, variant, probability = arguments
    # This call supplies the frozen v1.1 parameter stream and mask decisions.
    out, meta = _additive_sample(source, sample_id, epoch, seed=seed, mode=kind,
                                 variant="additive", probability=probability)
    if variant == "additive":
        return out, meta
    if not meta.get("applied", False):
        meta = dict(meta)
        meta.update(variant="multiplicative", training_protocol=ADAPTER_VERSION,
                    matched_sampling_from=TRAINING_WRAPPER_VERSION)
        return source.copy(), meta
    # Same cloud seed/level/seasons/coverage/shift as the additive draw; only
    # the engine variant changes.  The additive image itself is discarded.
    out, meta = ENGINE.degrade(
        source, seed=int(meta["seed"]), level=meta["level"], variant="multiplicative",
        seasons=meta["seasons_disturbed"], cloud_coverage=float(meta["cloud_coverage_target"]),
        overlap=(kind == "overlap"), overlap_shift=int(meta["effective_max_shift"]))
    meta.update(applied=True, mode=kind, epoch=int(epoch), sample_id=int(sample_id),
                pair_iou_min=meta.get("pair_iou_min"), mask_sha256=meta.get("mask_sha256"),
                training_protocol=ADAPTER_VERSION, coverage_jitter=meta.get("coverage_jitter"),
                effective_max_shift=meta.get("effective_max_shift"),
                matched_sampling_from=TRAINING_WRAPPER_VERSION)
    return out, meta


def _json_meta(meta):
    per_season = meta.get("per_season") or {}
    return {
        "variant": meta.get("variant"),
        "training_protocol": meta.get("training_protocol"),
        "matched_sampling_from": meta.get("matched_sampling_from"),
        "effective_max_shift": meta.get("effective_max_shift"),
        "pair_iou_min": meta.get("pair_iou_min"),
        "mask_sha256": meta.get("mask_sha256"),
        "coverage_jitter": meta.get("coverage_jitter"),
        "cloud_coverage_target": meta.get("cloud_coverage_target"),
        "level": meta.get("level"),
        "seasons_disturbed": list(meta.get("seasons_disturbed", [])),
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
    """Apply the frozen wrapper and optional multiplicative degradation.

    ``kind`` is independent/overlap; ``scope`` must be training.  The returned
    mask and season assignment are identical for additive and multiplicative
    draws because both use the same frozen v1.1 parameter stream.
    """
    if scope != "training":
        raise ValueError("Cloud adapter is training-only; use Ruan's fixed evaluator")
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
        raise ValueError(f"Unsupported cloud settings: {sorted(unknown)}")
    variant = (settings or {}).get("variant", "additive")
    if variant not in ("additive", "multiplicative"):
        raise ValueError("variant must be additive or multiplicative")
    probability = float((settings or {}).get("probability", 0.5))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0,1]")
    workers = (settings or {}).get("cpu_workers", 0)
    if isinstance(workers, bool) or not isinstance(workers, int) or not 0 <= workers <= 12:
        raise ValueError("cpu_workers must be an integer in [0,12]")

    sources = image.detach().cpu().numpy()
    jobs = [(sources[i], int(sid), int(epoch), int(seed), kind, variant, probability)
            for i, sid in enumerate(sample_ids)]
    if workers:
        if workers not in _POOLS:
            _POOLS[workers] = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        outputs = _POOLS[workers].map(_augment_job, jobs)
    else:
        outputs = map(_augment_job, jobs)

    affected, masks, seasons_per_sample, applied, provider_meta = [], [], [], [], []
    for source, (output, meta) in zip(sources, outputs):
        if output.shape != source.shape or not np.isfinite(output).all():
            raise ValueError("Cloud wrapper returned an invalid image")
        affected.append(output.astype(np.float32, copy=False))
        is_applied = bool(meta.get("applied", False))
        binary_masks = meta.get("binary_masks") or {}
        seasons = sorted(int(s) for s in binary_masks)
        if is_applied:
            if not 2 <= len(seasons) <= 4:
                raise ValueError("Cloud wrapper must apply 2-4 seasons")
        elif seasons:
            raise ValueError("Cloud wrapper returned seasons for an unapplied sample")
        sample_mask = np.zeros((4, 128, 128), dtype=bool)
        for season in seasons:
            sample_mask[season] = np.asarray(binary_masks[season], dtype=bool)
        masks.append(sample_mask)
        seasons_per_sample.append(seasons)
        applied.append(is_applied)
        provider_meta.append(_json_meta(meta))

    coverage_rows, targets, ious = [], [], []
    for seasons, is_applied, meta in zip(seasons_per_sample, applied, provider_meta):
        per = meta.get("per_season") or {}
        coverage_rows.append([per[str(s)]["coverage_actual"] for s in seasons] if is_applied else [])
        targets.append(meta.get("cloud_coverage_target") if is_applied else None)
        ious.append(meta.get("pair_iou_min") if len(seasons) >= 2 else None)
    result = torch.from_numpy(np.stack(affected, axis=0)).to(device=image.device, dtype=image.dtype)
    mask = torch.from_numpy(np.stack(masks, axis=0)).to(device=image.device, dtype=torch.bool)
    return result, mask, {
        "kind": kind,
        "scope": scope,
        "variant": variant,
        "adapter_version": ADAPTER_VERSION,
        "training_protocol": TRAINING_WRAPPER_VERSION if variant == "additive" else ADAPTER_VERSION,
        "training_wrapper_version": TRAINING_WRAPPER_VERSION,
        "applied": applied,
        "seasons_per_sample": seasons_per_sample,
        "seasons": sorted({s for seasons in seasons_per_sample for s in seasons}),
        "coverage_target_per_sample": targets,
        "coverage_per_sample_season": coverage_rows,
        "mask_iou_per_sample": ious,
        "samples": provider_meta,
    }
