"""Aggregate unified evaluations and paired contrasts without mixing protocols."""
import argparse
import json
from pathlib import Path
import statistics
from pipeline import write_json

GROUP_ALIASES = {"B_ind_D": "B_ind+D", "B_ovl_D": "B_ovl+D"}


def canonical_group(group):
    return GROUP_ALIASES.get(group, group)


CONTRASTS = {
    "independent_vs_clean": {"B_ind": 1, "A": -1},
    "overlap_vs_clean": {"B_ovl": 1, "A": -1},
    "overlap_vs_independent": {"B_ovl": 1, "B_ind": -1},
    "overlap_vs_independent_with_edge": {"B_ovl+D": 1, "B_ind+D": -1},
    "edge_clean": {"D": 1, "A": -1},
    "edge_independent": {"B_ind+D": 1, "B_ind": -1},
    "edge_overlap": {"B_ovl+D": 1, "B_ovl": -1},
    "interaction_independent": {"B_ind+D": 1, "B_ind": -1, "D": -1, "A": 1},
    "interaction_overlap": {"B_ovl+D": 1, "B_ovl": -1, "D": -1, "A": 1},
}


def distribution(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "values": values}


def summarize_files(paths, output):
    records = []
    for path in paths:
        records.extend(json.loads(Path(path).read_text(encoding="utf-8")))
    if not records or len({(r["protocol_sha256"], r["split"], r["purpose"]) for r in records}) != 1:
        raise ValueError("Do not mix protocols, splits, or research/synthetic results")
    lookup = {}
    for row in records:
        row["group"] = canonical_group(row["group"])
        key = row["group"], row["training_seed"], row["scene"]
        if key in lookup:
            raise ValueError("Duplicate group/seed/scene")
        lookup[key] = row
    aggregate, contrasts, missing = [], [], []
    for group, scene in sorted({(r["group"], r["scene"]) for r in records}):
        selected = [r for r in records if (r["group"], r["scene"]) == (group, scene)]
        aggregate.append({"group": group, "scene": scene,
            **{name: distribution([r["scores"]["whole"][name] for r in selected])
               for name in ("forest_macro_f1", "miou", "macro_f1")}})
    for scene in sorted({r["scene"] for r in records}):
        for name, terms in CONTRASTS.items():
            seed_sets = [{r["training_seed"] for r in records if r["group"] == g and r["scene"] == scene} for g in terms]
            if not seed_sets[0] or any(s != seed_sets[0] for s in seed_sets):
                missing.append({"contrast": name, "scene": scene, "reason": "Missing or mismatched paired seeds"})
                continue
            seeds = sorted(seed_sets[0])
            values = [sum(weight * lookup[group, seed, scene]["scores"]["whole"]["forest_macro_f1"]
                          for group, weight in terms.items()) for seed in seeds]
            contrasts.append({"contrast": name, "scene": scene, "seeds": seeds,
                              "forest_f1_difference": distribution(values)})
    result = {"protocol_sha256": records[0]["protocol_sha256"], "purpose": records[0]["purpose"],
              "split": records[0]["split"], "units": "ratios; multiply by 100 for percentage points",
              "single_seed_is_directional_only": any(x["forest_macro_f1"]["n"] < 3 for x in aggregate),
              "aggregate": aggregate, "paired_contrasts": contrasts, "missing_contrasts": missing}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "summary.json", result)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summarize_files(args.metrics, args.output)
