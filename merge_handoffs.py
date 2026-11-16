"""Merge core/B_mul handoffs without losing paths or trusting copied hashes."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_handoffs(paths, output):
    merged = {"scope": None, "protocol_sha256": None, "runtime_protocol_sha256": None,
              "sources": [], "checkpoints": []}
    seen = set()
    for path in paths:
        path = Path(path).resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        root = Path(value.get("root", path.parent)).resolve()
        identity = (value.get("scope"), value.get("protocol_sha256"), value.get("runtime_protocol_sha256"))
        if merged["scope"] is None:
            merged["scope"], merged["protocol_sha256"], merged["runtime_protocol_sha256"] = identity
        elif identity != (merged["scope"], merged["protocol_sha256"], merged["runtime_protocol_sha256"]):
            raise ValueError("Do not merge handoffs from different protocols/scopes")
        source = {"manifest": str(path), "root": str(root), "sha256": sha256(path)}
        merged["sources"].append(source)
        for item in value.get("checkpoints", []):
            checkpoint = Path(item["path"])
            if not checkpoint.is_absolute():
                checkpoint = (root / checkpoint).resolve()
            if not checkpoint.is_file():
                raise ValueError(f"Checkpoint not found: {checkpoint}")
            key = (item["delivery_group"], int(item["seed"]))
            if key in seen:
                raise ValueError(f"Duplicate checkpoint group/seed: {key}")
            seen.add(key)
            actual = sha256(checkpoint)
            if actual != item["sha256"]:
                raise ValueError(f"Checkpoint SHA256 mismatch: {checkpoint}")
            merged["checkpoints"].append({
                "group": item["group"],
                "delivery_group": item["delivery_group"],
                "seed": int(item["seed"]),
                "path": str(checkpoint),
                "sha256": actual,
                "source_manifest": str(path),
                "source_root": str(root),
            })
    out = Path(output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merge_handoffs(args.handoff, args.output)


if __name__ == "__main__":
    main()
