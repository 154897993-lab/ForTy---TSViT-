"""Train a selected six-group/seed matrix; hand checkpoints to Ruan for evaluation.

This runner deliberately does not produce four-scene metrics. Ruan's fixed
evaluation entry with the frozen eval manifest is the authoritative evaluator.
"""
import argparse
from pathlib import Path
from model import GROUPS, delivery_group
from pipeline import Clouds, delivery_metadata, load_protocol, resolve, sha256, write_json
from train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", nargs="+", choices=tuple(GROUPS), default=list(GROUPS))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg, fingerprint = load_protocol(args.config)
    if len(set(args.groups)) != len(args.groups) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate group/seed")
    if any(seed not in cfg["training_seeds"] for seed in args.seeds):
        raise ValueError("All matrix seeds must be declared before starting any training")
    delivery = delivery_metadata(cfg)
    if delivery is None:
        raise ValueError("Six-group handoff requires a delivery block for Ruan's evaluator")
    # Only groups with added cloud need a provider; clean groups do not.
    Clouds(cfg, required=any(GROUPS[group][0] != "clean" for group in args.groups))
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in args.seeds:
        for group in args.groups:
            checkpoint = train(args.config, group, seed, root / f"{group}_seed{seed}", args.device)
            records.append({"group": group, "delivery_group": delivery_group(group), "seed": seed,
                            "path": Path(checkpoint).relative_to(root).as_posix(),
                            "sha256": sha256(checkpoint)})
    write_json(root / "handoff_checkpoints.json", {
        "scope": delivery["scope"],
        "evaluation_owner": "ruan",
        "note": "Unified four-scene evaluation is performed by Ruan's fixed entry with the frozen eval manifest; this runner does not produce official metrics.",
        "protocol_sha256": delivery["protocol_sha256"],
        "runtime_protocol_sha256": fingerprint,
        "train_manifest_sha256": sha256(resolve(cfg, cfg["data"]["manifest"])),
        "eval_manifest_sha256": delivery["eval_manifest_sha256"],
        "checkpoints": records,
    })


if __name__ == "__main__":
    main()
