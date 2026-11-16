"""Train a declared core or B_mul matrix; hand checkpoints to Ruan.

Core groups (7) and B_mul are separate invocations with separate output roots.
B_mul is frozen to seed 20261113 and is never expanded to the three P2 seeds
by this runner.
"""
import argparse
import json
from pathlib import Path
from model import GROUPS, CORE_GROUPS, VARIANT_GROUPS, delivery_group, training_variant
from pipeline import Clouds, delivery_metadata, load_protocol, resolve, sha256, write_json
from train import train


def p2_matrix(cfg):
    """Full P2 plan: 7 core groups x 3 seeds + B_mul x seed 20261113 = 22."""
    plan = matrix_plan(cfg, CORE_GROUPS, cfg["training_seeds"]) + matrix_plan(cfg, list(VARIANT_GROUPS), [20261113])
    if len(plan) != 22 or len({(x["group"], x["seed"]) for x in plan}) != 22:
        raise ValueError("P2 matrix must contain 22 unique (group, seed) jobs")
    return plan


def matrix_plan(cfg, groups, seeds):
    """Return the explicit unique (group, seed) plan; reject silent expansion."""
    groups, seeds = list(groups), list(seeds)
    if len(set(groups)) != len(groups) or len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate group/seed")
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        raise ValueError(f"Unknown groups: {unknown}")
    if any(seed not in cfg["training_seeds"] for seed in seeds):
        raise ValueError("All matrix seeds must be declared before starting any training")
    if "B_mul" in groups:
        if set(groups) != {"B_mul"}:
            raise ValueError("B_mul must run in its own output root, not mixed with core groups")
        if seeds != [20261113]:
            raise ValueError("B_mul is frozen to seed 20261113 only")
    plan = []
    for seed in seeds:
        for group in groups:
            plan.append({"group": group, "delivery_group": delivery_group(group), "seed": seed,
                         "core": group in CORE_GROUPS, "variant": training_variant(group)})
    if len({(x["group"], x["seed"]) for x in plan}) != len(plan):
        raise ValueError("Duplicate (group, seed) in matrix plan")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", nargs="+", choices=tuple(GROUPS), default=list(CORE_GROUPS))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true", help="Print the explicit matrix and exit")
    args = parser.parse_args()
    cfg, fingerprint = load_protocol(args.config)
    plan = matrix_plan(cfg, args.groups, args.seeds)
    if args.dry_run:
        print(json.dumps({"count": len(plan), "matrix": plan}, ensure_ascii=False, indent=2))
        return
    delivery = delivery_metadata(cfg)
    if delivery is None:
        raise ValueError("Handoff requires a delivery block for Ruan's evaluator")
    # G is an erasing group and does not require the cloud provider.
    Clouds(cfg, required=any(GROUPS[g][0] in ("independent", "overlap") for g in args.groups))
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    records = []
    for item in plan:
        group, seed = item["group"], item["seed"]
        checkpoint = train(args.config, group, seed, root / f"{group}_seed{seed}", args.device)
        item.update({"path": Path(checkpoint).relative_to(root).as_posix(),
                     "sha256": sha256(checkpoint)})
        records.append(item)
    write_json(root / "handoff_checkpoints.json", {
        "scope": delivery["scope"],
        "evaluation_owner": "ruan",
        "root": str(root),
        "source_manifest": str(root / "handoff_checkpoints.json"),
        "note": "Unified four-scene evaluation is performed by Ruan's fixed entry with the frozen eval manifest; this runner does not produce official metrics.",
        "protocol_sha256": delivery["protocol_sha256"],
        "runtime_protocol_sha256": fingerprint,
        "train_manifest_sha256": sha256(resolve(cfg, cfg["data"]["manifest"])),
        "eval_manifest_sha256": delivery["eval_manifest_sha256"],
        "matrix": plan,
        "checkpoints": records,
    })


if __name__ == "__main__":
    main()
