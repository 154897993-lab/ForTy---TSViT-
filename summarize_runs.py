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
    valid = [value for value in values if value is not None]
    return {"n": len(values), "n_valid": len(valid),
            "mean": statistics.mean(valid) if valid else None,
            "sample_std": statistics.stdev(valid) if len(valid) > 1 else None,
            "na_count": len(values) - len(valid),
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
            raw = [[lookup[group, seed, scene]["scores"]["whole"]["forest_macro_f1"]
                    for group in terms] for seed in seeds]
            if any(value is None for row in raw for value in row):
                missing.append({"contrast": name, "scene": scene, "reason": "NA forest macro F1"})
                continue
            values = [sum(weight * value for weight, value in zip(terms.values(), row))
                      for row in raw]
            contrasts.append({"contrast": name, "scene": scene, "seeds": seeds,
                              "forest_f1_difference": distribution(values)})
    result = {"protocol_sha256": records[0]["protocol_sha256"], "purpose": records[0]["purpose"],
              "split": records[0]["split"], "units": "ratios; multiply by 100 for percentage points",
              "single_seed_is_directional_only": any(x["forest_macro_f1"]["n_valid"] < 3 for x in aggregate),
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
