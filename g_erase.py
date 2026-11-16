"""G: budget-matched rectangular erasing, training-only.

G shares the frozen B_ind parameter draws for probability, seasons and
coverage.  Rectangle geometry uses a separate G-only deterministic stream so
batch order or DataLoader workers cannot change a sample's augmentation.
Labels are never passed to this module and are never modified.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

_VENDOR_ROOT = Path(__file__).resolve().parent / "ruan_vendor"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from src.protocol import derived_seed  # noqa: E402
from src.training_clouds import VERSION as BUDGET_WRAPPER_VERSION  # noqa: E402

G_VERSION = "g-rect-erase-v1.0"
FILL_STATS_VERSION = "g-fill-stats-v1.0"
BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
SHAPE = (4, 128, 128, 10)


def _resolve(cfg, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(cfg["_directory"]) / path).resolve()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compute_fill_stats(dataset, manifest_sha256=None):
    """Ten per-band means over every pixel/season of the current train split.

    The fill statistic is image-level: unknown labels are not excluded.  It is
    computed once from the current training manifest and then frozen; G never
    recomputes it from validation or from a batch at run time.
    """
    total = torch.zeros(10, dtype=torch.float64)
    sample_count = 0
    pixel_count = 0
    for index in range(len(dataset)):
        item = dataset[index]
        s2 = torch.as_tensor(item["s2"], dtype=torch.float64)
        if tuple(s2.shape) != SHAPE or not bool(torch.isfinite(s2).all()):
            raise ValueError(f"Invalid training S2 for fill statistics: {tuple(s2.shape)}")
        total += s2.sum(dim=(0, 1, 2))
        sample_count += 1
        pixel_count += int(4 * 128 * 128)
    if not sample_count:
        raise ValueError("Cannot compute G fill statistics from an empty train split")
    mean = (total / pixel_count).tolist()
    return {
        "version": FILL_STATS_VERSION,
        "bands": list(BANDS),
        "mean": [float(v) for v in mean],
        "sample_count": int(sample_count),
        "pixel_count": int(pixel_count),
        "source": "current_training_manifest_only",
        "manifest_sha256": manifest_sha256,
        "definition": "mean over all four seasons and all 128x128 pixels of every training sample; unknown labels are not excluded",
    }


def write_fill_stats(path, stats):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def load_fill_stats(path, expected_sha256=None):
    path = Path(path)
    actual = sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("G fill-stats SHA256 mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("version") != FILL_STATS_VERSION:
        raise ValueError("Unexpected G fill-stats version")
    mean = value.get("mean")
    if (not isinstance(mean, list) or len(mean) != 10
            or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in mean)):
        raise ValueError("G fill-stats mean must be ten finite numbers")
    if not isinstance(value.get("sample_count"), int) or value["sample_count"] < 1:
        raise ValueError("G fill-stats sample_count must be positive")
    if not isinstance(value.get("pixel_count"), int) or value["pixel_count"] < 1:
        raise ValueError("G fill-stats pixel_count must be positive")
    if value.get("bands") != list(BANDS):
        raise ValueError("G fill-stats band order mismatch")
    value["_sha256"] = actual
    return value


class GErase:
    """Deterministic rectangular erasing for the G training group."""

    def __init__(self, cfg):
        spec = cfg.get("g")
        if not isinstance(spec, dict):
            raise ValueError("G group requires a frozen g block")
        stats_path = _resolve(cfg, spec["fill_stats"])
        self.fill_stats = load_fill_stats(stats_path, spec.get("fill_stats_sha256"))
        self.fill_stats_sha256 = self.fill_stats["_sha256"]
        self.fill_mean = np.asarray(self.fill_stats["mean"], dtype=np.float32)
        self.probability = float(spec.get("probability", 0.5))
        self.seasons_min = int(spec.get("seasons_min", 2))
        self.seasons_max = int(spec.get("seasons_max", 4))
        self.coverage_min = float(spec.get("coverage_min", 0.1))
        self.coverage_max = float(spec.get("coverage_max", 0.7))
        self.aspect_min = float(spec.get("aspect_ratio_min", 0.5))
        self.aspect_max = float(spec.get("aspect_ratio_max", 2.0))
        if self.probability != 0.5:
            raise ValueError("G probability is frozen at 0.5")
        if (self.seasons_min, self.seasons_max) != (2, 4):
            raise ValueError("G must disturb 2-4 seasons")
        if (self.coverage_min, self.coverage_max) != (0.1, 0.7):
            raise ValueError("G coverage is frozen at U(0.1,0.7)")
        if not (0 < self.aspect_min <= self.aspect_max):
            raise ValueError("Invalid G aspect-ratio range")

    def _rectangle(self, coverage, rng):
        h, w = 128, 128
        area = float(coverage) * h * w
        # A requested aspect ratio can be geometrically infeasible at high
        # coverage.  Clip only the feasible interval, then preserve the target
        # area; this keeps actual coverage close to B_ind's target while the
        # audit records the realized aspect ratio.
        low = max(self.aspect_min, area / (h * h))
        high = min(self.aspect_max, (w * w) / area)
        if low > high:
            low = high = math.sqrt(low * high)
        ratio = float(np.exp(rng.uniform(math.log(low), math.log(high))))
        height = max(1, min(h, int(round(math.sqrt(area / ratio)))))
        width = max(1, min(w, int(round(area / height))))
        if width > w:
            width = w
            height = max(1, min(h, int(round(area / width))))
        if height > h:
            height = h
            width = max(1, min(w, int(round(area / height))))
        y0 = int(rng.integers(0, h - height + 1))
        x0 = int(rng.integers(0, w - width + 1))
        return y0, x0, height, width

    def apply(self, image, ids, scope, seed, epoch):
        if scope != "training":
            raise ValueError("G erasing is training-only")
        if not isinstance(image, torch.Tensor) or image.ndim != 5 or tuple(image.shape[1:]) != SHAPE:
            raise ValueError("G expects image [B,4,128,128,10]")
        if not image.is_floating_point() or not bool(torch.isfinite(image).all()):
            raise ValueError("G expects finite floating-point reflectance")
        if len(ids) != image.shape[0]:
            raise ValueError("G sample_ids length must match batch size")
        source = image.detach().cpu().numpy()
        output = source.copy()
        mask = np.zeros((image.shape[0], 4, 128, 128), dtype=bool)
        applied, seasons_per_sample, targets, coverage_rows, samples = [], [], [], [], []
        for i, sid in enumerate(ids):
            sid = int(sid)
            rng = np.random.default_rng(derived_seed(BUDGET_WRAPPER_VERSION + "-parameters",
                                                     int(seed), int(epoch), sid))
            use = bool(rng.random() < self.probability)
            if not use:
                applied.append(False)
                seasons_per_sample.append([])
                targets.append(None)
                coverage_rows.append([])
                samples.append({"sample_id": sid, "applied": False, "seasons": [], "coverage_target": None,
                                "rectangles": []})
                continue
            seasons = sorted(int(v) for v in rng.choice(
                4, size=int(rng.integers(self.seasons_min, self.seasons_max + 1)), replace=False).tolist())
            coverage = float(rng.uniform(self.coverage_min, self.coverage_max))
            rectangles, actuals = [], []
            for season in seasons:
                rr = np.random.default_rng(derived_seed(G_VERSION + "-rect", int(seed), int(epoch), sid, season))
                y0, x0, height, width = self._rectangle(coverage, rr)
                output[i, season, y0:y0 + height, x0:x0 + width, :] = self.fill_mean
                mask[i, season, y0:y0 + height, x0:x0 + width] = True
                actual = float(height * width) / float(128 * 128)
                actuals.append(actual)
                rectangles.append({"season": season, "y0": y0, "x0": x0, "h": height, "w": width,
                                   "coverage_target": coverage, "coverage_actual": actual,
                                   "aspect_ratio": float(width) / float(height)})
            applied.append(True)
            seasons_per_sample.append(seasons)
            targets.append(coverage)
            coverage_rows.append(actuals)
            samples.append({"sample_id": sid, "applied": True, "seasons": seasons,
                            "coverage_target": coverage, "rectangles": rectangles})
        details = {
            "kind": "erase",
            "scope": scope,
            "seed": int(seed),
            "epoch": int(epoch),
            "applied": applied,
            "seasons_per_sample": seasons_per_sample,
            "coverage_target_per_sample": targets,
            "coverage_per_sample_season": coverage_rows,
            "mask_iou_per_sample": [None] * len(ids),
            "g_version": G_VERSION,
            "sampling_source": BUDGET_WRAPPER_VERSION,
            "fill_stats_version": self.fill_stats["version"],
            "fill_stats_sha256": self.fill_stats_sha256,
            "samples": samples,
        }
        json.dumps(details, allow_nan=False)
        return (torch.from_numpy(output).to(device=image.device, dtype=image.dtype),
                torch.from_numpy(mask).to(device=image.device), details)
