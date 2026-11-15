"""Independent manifest-indexed data; no glob-based data splits."""
import csv
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from record_io import read_example


def equal_class_batches(dominant_classes, batch_size, seed):
    """Without-replacement stratification for an equally balanced fit cohort.

    Uses training labels only. Refuses unequal pools instead of oversampling or
    silently discarding records; not a general full-dataset sampling policy.
    """
    if batch_size < 8 or batch_size % 8:
        raise ValueError("Equal-class batches require batch size divisible by 8")
    labels = torch.as_tensor(dominant_classes, dtype=torch.long)
    if labels.ndim != 1 or bool(((labels < 0) | (labels > 7)).any()):
        raise ValueError("Every fit sample must have a valid dominant class")
    pools = [torch.where(labels == value)[0] for value in range(8)]
    sizes = [len(pool) for pool in pools]
    per_class = batch_size // 8
    if not sizes[0] or len(set(sizes)) != 1 or sizes[0] % per_class:
        raise ValueError("Fit cohort must contain equal, divisible class pools")
    generator = torch.Generator().manual_seed(seed)
    pools = [pool[torch.randperm(len(pool), generator=generator)] for pool in pools]
    batches = []
    for start in range(0, sizes[0], per_class):
        batch = torch.cat([pool[start:start + per_class] for pool in pools])
        batches.append(batch[torch.randperm(batch_size, generator=generator)].tolist())
    return batches


def read_manifest(path, allow_test=False):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty manifest")
    seen = set()
    for row in rows:
        for field in ("sample_id", "split", "shard"):
            if not row.get(field):
                raise ValueError(f"Missing manifest field {field}")
        row["sample_id"] = int(row["sample_id"])
        if row["sample_id"] in seen:
            raise ValueError("Duplicate sample ID, including across splits")
        seen.add(row["sample_id"])
        if row["split"] not in ("train", "validation", "test"):
            raise ValueError("Use train/validation/test split names")
        if row["split"] == "test" and not allow_test:
            raise ValueError("Sealed test cannot appear in the train/validation manifest")
    return rows


class ForTyDataset(Dataset):
    def __init__(self, rows, root, split, storage="tfrecord", divisor=10000.0, limit=None, cache=False,
                 index_sha256=None):
        self.rows = [row for row in rows if row["split"] == split]
        if limit is not None:
            if not isinstance(limit, int) or limit < 1 or limit > len(self.rows):
                raise ValueError("Invalid explicit sample limit")
            self.rows = self.rows[:limit]
        if not self.rows or storage not in ("tfrecord", "npz", "prepared") or divisor <= 0:
            raise ValueError("Invalid dataset/split/storage/divisor")
        self.root = Path(root).resolve()
        self.storage, self.divisor, self.cache = storage, divisor, cache
        self.cached = {}
        self.index = {}
        if storage == "prepared":
            if divisor != 1.0:
                raise ValueError("Prepared reflectance must use divisor=1; do not scale twice")
            index_path = self.root / "index.csv"
            if not index_sha256 or hashlib.sha256(index_path.read_bytes()).hexdigest() != index_sha256:
                raise ValueError("Prepared index SHA256 mismatch")
            with index_path.open(encoding="utf-8-sig", newline="") as stream:
                for entry in csv.DictReader(stream):
                    sid = int(entry["sample_id"])
                    if sid in self.index:
                        raise ValueError("Duplicate ID in prepared index")
                    self.index[sid] = entry
        for row in self.rows:
            path = self._path(row).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f"Missing shard or path outside root: {row['shard']}")

    def _path(self, row):
        if self.storage == "prepared":
            return self.root / row["split"] / f"{row['sample_id']}.npz"
        return self.root / row["shard"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if index not in self.cached:
            row = self.rows[index]
            path = self._path(row)
            valid_mask = None
            if self.storage == "prepared":
                entry = self.index.get(row["sample_id"])
                if entry is None or entry["npz_path"] != path.relative_to(self.root).as_posix():
                    raise ValueError("Prepared index ID/path mismatch")
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError("Prepared sample SHA256 mismatch")
                with np.load(path, allow_pickle=False) as record:
                    if str(record["unit"]) != "reflectance" or int(record["scale_applied"]) != 10000:
                        raise ValueError("Prepared units/scaling metadata mismatch")
                    if str(record["split"]) != row["split"]:
                        raise ValueError("Prepared split mismatch")
                    example = {"s2": record["s2"].copy(), "segmentation_labels": record["labels"].copy(),
                               "id": record["sample_id"].copy()}
                    valid_mask = record["valid_mask"].copy()
                    if valid_mask.shape != (4, 128, 128, 10) or not np.isin(valid_mask, (0, 1)).all():
                        raise ValueError("Invalid optical validity mask")
            elif self.storage == "npz":
                with np.load(path, allow_pickle=False) as record:
                    example = {name: record[name].copy() for name in ("s2", "segmentation_labels", "id")}
            else:
                example = read_example(path, int(row["byte_offset"]))
            image = np.asarray(example["s2"], dtype=np.float32)
            label = np.asarray(example["segmentation_labels"])
            if int(np.asarray(example["id"]).item()) != row["sample_id"]:
                raise ValueError("Manifest ID and record ID disagree")
            if image.shape != (4, 128, 128, 10) or not np.isfinite(image).all():
                raise ValueError("Invalid S2 shape or non-finite values")
            if label.shape != (128, 128) or label.dtype.kind not in "iu" or (label < 0).any() or (label > 8).any():
                raise ValueError("Labels must be integers in 0..8 at 128x128")
            item = {"s2": torch.from_numpy(image.copy()) / self.divisor,
                    "label": torch.from_numpy(label.astype(np.int64)), "id": row["sample_id"]}
            if valid_mask is not None:
                item["valid_mask"] = torch.from_numpy(valid_mask)
            if self.cache:
                self.cached[index] = item
        else:
            item = self.cached[index]
        return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in item.items()}
