"""Independent protocol, augmentation boundary, metrics and evaluation engine."""
import csv
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data import ForTyDataset, read_manifest
from forty_adapter import BANDS, SEASONS, flatten_s2, encode_labels, decode_predictions
from metrics_reference import confusion_matrix, summarize

SCENES = ("clean", "single", "dual_indep", "dual_overlap")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def resolve(cfg, value):
    return (Path(cfg["_directory"]) / value).resolve()


def load_protocol(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    if cfg.get("frozen") is not True or not cfg.get("version"):
        raise ValueError("Protocol must have an explicit version and frozen=true")
    if cfg.get("purpose") not in ("research", "synthetic"):
        raise ValueError("Declare purpose=research or synthetic")
    if cfg.get("seasons") != list(SEASONS) or cfg.get("bands") != list(BANDS):
        raise ValueError("Confirm the four-season/ten-band ordering before running")
    train = cfg["training"]
    for key in ("epochs", "batch_size", "eval_batch_size", "base_channels", "cpu_threads"):
        if not isinstance(train[key], int) or isinstance(train[key], bool) or train[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if not isinstance(train["workers"], int) or isinstance(train["workers"], bool) or train["workers"] < 0:
        raise ValueError("workers must be a nonnegative integer")
    if (not math.isfinite(train["lr"]) or not math.isfinite(train["weight_decay"])
            or train["lr"] <= 0 or train["weight_decay"] < 0):
        raise ValueError("Invalid workers/lr/weight_decay")
    if train["precision"] not in ("float32", "bfloat16"):
        raise ValueError("precision must be float32 or bfloat16")
    if train["select_metric"] not in ("forest_macro_f1", "macro_f1", "miou"):
        raise ValueError("Freeze a valid checkpoint selection metric")
    width = cfg["evaluation"]["boundary_width"]
    if not isinstance(width, int) or isinstance(width, bool) or width < 0:
        raise ValueError("Freeze boundary_width (0 disables boundary metrics)")
    seeds = cfg["training_seeds"]
    if (not seeds or any(not isinstance(s, int) or isinstance(s, bool) or not 0 <= s < 2 ** 32 for s in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError("Freeze unique nonnegative 32-bit training seeds")
    if not math.isfinite(cfg["data"]["divisor"]) or cfg["data"]["divisor"] <= 0:
        raise ValueError("Data divisor must be finite and positive")
    cfg["_directory"] = str(path.parent)
    manifest = resolve(cfg, cfg["data"]["manifest"])
    actual = sha256(manifest)
    if actual != cfg["data"]["manifest_sha256"]:
        raise ValueError("Manifest SHA256 mismatch; do not silently change sample lists")
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    return cfg, fingerprint


def source_hashes():
    root = Path(__file__).parent
    names = ("train.py", "fit_train32.py", "evaluate.py", "check_data.py", "run_experiments.py", "summarize_runs.py", "pipeline.py", "data.py", "record_io.py", "model.py",
             "forest_unet.py", "forty_adapter.py", "metrics_reference.py")
    return {name: sha256(root / name) for name in names}


def make_data(cfg, split, allow_test=False):
    data = cfg["data"]
    if split == "test":
        path = resolve(cfg, data["test_manifest"])
        if sha256(path) != data["test_manifest_sha256"]:
            raise ValueError("Test manifest SHA256 mismatch")
    else:
        path = resolve(cfg, data["manifest"])
    rows = read_manifest(path, allow_test=allow_test)
    if split == "test":
        if any(row["split"] != "test" for row in rows):
            raise ValueError("Final test manifest must contain only test rows")
        reference = read_manifest(resolve(cfg, data["manifest"]))
        if {row["sample_id"] for row in rows} & {row["sample_id"] for row in reference}:
            raise ValueError("Test IDs overlap training/validation IDs")
    expected = data["expected_counts"]
    for name in ({"test"} if split == "test" else {"train", "validation"}):
        count = sum(row["split"] == name for row in rows)
        if count != expected[name]:
            raise ValueError(f"Expected {expected[name]} {name} samples, found {count}")
    return ForTyDataset(rows, resolve(cfg, data["root"]), split, storage=data["format"],
                        divisor=data["divisor"], limit=data.get("limits", {}).get(split),
                        cache=data["cache_samples"], index_sha256=data.get("index_sha256"))


def loader(dataset, cfg, training=False, seed=0):
    settings = cfg["training"]
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=settings["batch_size"] if training else settings["eval_batch_size"],
                      shuffle=training, generator=generator, num_workers=settings["workers"],
                      drop_last=False, pin_memory=False)


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False


class Clouds:
    """Provider receives no labels. All scientific cloud settings belong to Ruan's protocol."""
    def __init__(self, cfg, required):
        self.cfg, self.provider = cfg, None
        spec = cfg["clouds"]
        if required:
            if not spec.get("provider") or not spec.get("sha256") or spec.get("settings") is None:
                raise ValueError("Cloud protocol/provider is not frozen; augmented groups cannot run")
            module_name, function_name = spec["provider"].split(":")
            function = getattr(importlib.import_module(module_name), function_name)
            source = inspect.getsourcefile(function)
            if source is None or sha256(source) != spec["sha256"]:
                raise ValueError("Cloud provider source SHA256 mismatch")
            self.provider = function

    def apply(self, image, ids, kind, scope, seed, epoch):
        if kind == "clean":
            return image, None, {"kind": kind, "scope": scope}
        if self.provider is None:
            raise ValueError("Non-clean scene requires a frozen cloud provider")
        affected, mask, details = self.provider(image=image.clone(), sample_ids=ids.tolist(), kind=kind,
            scope=scope, seed=seed, epoch=epoch, settings=self.cfg["clouds"]["settings"])
        expected_mask = image.shape[:-1]
        if (not isinstance(affected, torch.Tensor) or affected.shape != image.shape
                or affected.dtype != image.dtype or affected.device != image.device
                or not bool(torch.isfinite(affected).all())):
            raise ValueError("Provider must return a finite image with unchanged shape/device/dtype")
        if not isinstance(mask, torch.Tensor) or mask.shape != expected_mask or mask.dtype != torch.bool:
            raise ValueError("Provider must return a boolean cloud mask [B,4,128,128]")
        mask = mask.to(image.device)
        seasons = details["seasons"]
        required = 1 if kind == "single" else 2
        if len(seasons) != required or len(set(seasons)) != required or any(s not in range(4) for s in seasons):
            raise ValueError("Provider must identify one/two distinct affected seasons")
        if any(bool(mask[:, s].any()) for s in range(4) if s not in seasons):
            raise ValueError("Cloud mask affects undeclared seasons")
        coverage = mask.float().mean(dim=(-1, -2)).cpu().tolist()
        iou = None
        if required == 2:
            left, right = mask[:, seasons[0]], mask[:, seasons[1]]
            union = (left | right).sum(dim=(-1, -2))
            iou = ((left & right).sum(dim=(-1, -2)) / union.clamp_min(1)).cpu().tolist()
        metadata = {"kind": kind, "scope": scope, "seed": seed, "epoch": epoch,
                    "ids": ids.tolist(), "coverage_per_sample_season": coverage,
                    "mask_iou_per_sample": iou, "provider": details}
        json.dumps(metadata, allow_nan=False)
        return affected, mask, metadata


def boundary_mask(target, width):
    """Adjacent valid labels define a boundary; dilate width pixels, exclude nodata."""
    valid = target > 0
    boundary = torch.zeros_like(valid)
    vertical = valid[:, 1:] & valid[:, :-1] & (target[:, 1:] != target[:, :-1])
    horizontal = valid[:, :, 1:] & valid[:, :, :-1] & (target[:, :, 1:] != target[:, :, :-1])
    boundary[:, 1:] |= vertical
    boundary[:, :-1] |= vertical
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    if width:
        boundary = F.max_pool2d(boundary[:, None].float(), 2 * width + 1, stride=1, padding=width)[:, 0].bool()
    return boundary & valid


def metric_values(matrix):
    return {name: value.tolist() if value.ndim else value.item() for name, value in summarize(matrix).items()}


def evaluate_loader(net, batches, cfg, clouds, scene, seed, device, export_dir=None, audit=None):
    if scene not in SCENES:
        raise ValueError("Unknown evaluation scene")
    net.eval()
    total, band = torch.zeros(9, 9, dtype=torch.int64), torch.zeros(9, 9, dtype=torch.int64)
    width = cfg["evaluation"]["boundary_width"]
    with torch.no_grad():
        for batch in batches:
            image, target = batch["s2"].to(device), batch["label"].to(device)
            affected, mask, details = clouds.apply(image, batch["id"], scene, "evaluation", seed, 0)
            logits = net(flatten_s2(affected))
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Non-finite model predictions")
            pred = decode_predictions(logits.argmax(1))
            total += confusion_matrix(pred, target).cpu()
            if width:
                selected = boundary_mask(target, width)
                band += confusion_matrix(pred[selected], target[selected]).cpu()
            if audit:
                audit.write(json.dumps(details, allow_nan=False) + "\n")
            if export_dir is not None:
                export_dir.mkdir(parents=True, exist_ok=True)
                for i, sid in enumerate(batch["id"].tolist()):
                    arrays = {"prediction": pred[i].cpu().numpy().astype(np.uint8),
                              "label": target[i].cpu().numpy().astype(np.uint8),
                              "s2_clean": image[i].cpu().numpy(), "s2_evaluated": affected[i].cpu().numpy()}
                    if mask is not None:
                        arrays["cloud_mask"] = mask[i].cpu().numpy()
                    np.savez_compressed(export_dir / f"{sid}.npz", **arrays)
    if not int(total.sum()):
        raise ValueError("No valid pixels in evaluation dataset")
    return {"whole": metric_values(total), "boundary": metric_values(band) if width and int(band.sum()) else None,
            "valid_pixels": int(total.sum()), "boundary_pixels": int(band.sum()), "confusion": total.tolist()}


def write_score_rows(path, records):
    fields = ("run_id", "group", "training_cloud", "training_seed", "split", "scene", "scene_seed",
              "protocol_version", "protocol_sha256", "checkpoint_sha256", "purpose",
              "forest_macro_f1", "planted_f1", "miou", "macro_f1", "boundary_miou")
    with open(path, "x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for metadata, scores in records:
            whole, boundary = scores["whole"], scores["boundary"]
            writer.writerow({**metadata, "forest_macro_f1": whole["forest_macro_f1"],
                "planted_f1": whole["f1"][1], "miou": whole["miou"], "macro_f1": whole["macro_f1"],
                "boundary_miou": boundary["miou"] if boundary else ""})
