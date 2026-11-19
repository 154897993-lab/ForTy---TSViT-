"""Run the D2 deep-validation stage in three lanes.

The scientific training function is imported unchanged from the frozen
d2_develop.py (run_trial): same loss, optimizer, seeds, 100 epochs and
selection.  This file only changes execution structure:

  prepare  -> read promotion_decision.json and split the deep tasks into 3 lanes
  lane     -> run one lane sequentially; each task writes only its own run dir
  finalize -> rebuild deep_100ep.json and completion.json in the parent schema

The original single-process runner must be stopped before the lanes start;
d2_pause_at_promotion.py does that at the promotion boundary.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "development_protocol.json"
TASKS = ROOT / "deep_tasks.json"
PROMOTION = ROOT / "promotion_decision.json"
SCREEN = ROOT / "screen_30ep.json"
DEEP = ROOT / "deep_100ep.json"
COMPLETION = ROOT / "completion.json"
STATUS = ROOT / "status.json"


def load_cfg():
    return json.loads(CONFIG.read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def prepare():
    cfg = load_cfg()
    if not PROMOTION.exists():
        raise SystemExit("promotion_decision.json not ready")
    if TASKS.exists():
        raise SystemExit("deep_tasks.json already exists")
    if not SCREEN.exists():
        raise SystemExit("screen_30ep.json missing")
    promo = json.loads(PROMOTION.read_text(encoding="utf-8"))
    deepids = set(promo["deep_case_ids"])
    order = []
    for seed in cfg["seeds"]:
        for case in cfg["cases"]:
            if case["id"] in deepids:
                weight = 70 if seed == cfg["screen_seed"] else 100
                order.append({"case_id": case["id"], "seed": seed, "target": 100, "weight": weight})
    lanes = {key: [] for key in ("A", "B", "C")}
    loads = {key: 0 for key in lanes}
    for task in sorted(order, key=lambda item: (-item["weight"], item["seed"], item["case_id"])):
        lane = min(lanes, key=lambda key: (loads[key], key))
        lanes[lane].append(task)
        loads[lane] += task["weight"]
    payload = {
        "order": order,
        "lanes": lanes,
        "load": loads,
        "deep_case_ids": sorted(deepids),
        "promoted_case_ids": [item["case"]["id"] for item in promo["promoted"]],
    }
    write_json(TASKS, payload)
    print(json.dumps({
        "deep_case_ids": payload["deep_case_ids"],
        "promoted_case_ids": payload["promoted_case_ids"],
        "lanes": {key: [f"{t['case_id']}:{t['seed']}" for t in value] for key, value in lanes.items()},
        "load": loads,
    }, ensure_ascii=False, indent=2))


def lane(name):
    if not TASKS.exists():
        raise SystemExit("deep_tasks.json missing")
    payload = json.loads(TASKS.read_text(encoding="utf-8"))
    tasks = payload["lanes"][name]
    import d2_develop as d2

    p = d2.verify()
    train_data, val_data = d2.make_data(p, "train"), d2.make_data(p, "validation")
    if len(train_data) != 3000 or len(val_data) != 500:
        raise SystemExit("unexpected data counts")
    original_write = d2.write_json

    def scoped_write(path, obj):
        if Path(path).name == "status.json":
            path = ROOT / f"status_lane_{name}.json"
        return original_write(path, obj)

    d2.write_json = scoped_write
    results = []
    try:
        for task in tasks:
            case = next(c for c in d2.cfg["cases"] if c["id"] == task["case_id"])
            print(json.dumps({"lane": name, "task": task}, ensure_ascii=False), flush=True)
            summary = d2.run_trial(p, train_data, val_data, case, task["seed"], task["target"])
            results.append(summary)
            write_json(ROOT / f"lane_{name}.json", results)
        print(f"LANE_DONE {name} tasks={len(results)}", flush=True)
    except BaseException:
        write_json(ROOT / f"failure_lane_{name}.json", {
            "status": "STOPPED_TECHNICAL_FAILURE",
            "lane": name,
            "tasks": tasks,
            "completed_tasks": len(results),
            "traceback": traceback.format_exc(),
            "time": time.time(),
        })
        raise


def finalize():
    cfg = load_cfg()
    if COMPLETION.exists():
        raise SystemExit("completion.json already exists")
    payload = json.loads(TASKS.read_text(encoding="utf-8"))
    promo = json.loads(PROMOTION.read_text(encoding="utf-8"))
    results = {}
    for name in ("A", "B", "C"):
        path = ROOT / f"lane_{name}.json"
        if not path.exists():
            raise SystemExit(f"lane_{name}.json missing")
        for summary in json.loads(path.read_text(encoding="utf-8")):
            key = (summary["case"]["id"], summary["seed"])
            if key in results:
                raise SystemExit(f"duplicate summary {key}")
            results[key] = summary
    deep = []
    deepids = set(promo["deep_case_ids"])
    for seed in cfg["seeds"]:
        for case in cfg["cases"]:
            if case["id"] in deepids:
                key = (case["id"], seed)
                if key not in results:
                    raise SystemExit(f"missing summary {key}")
                deep.append(results[key])
    write_json(DEEP, deep)

    comparisons = []
    for selected in promo["promoted"]:
        case = selected["case"]
        deltas = []
        for seed in cfg["seeds"]:
            x = next(v for v in deep if v["case"]["id"] == case["id"] and v["seed"] == seed)
            a = next(v for v in deep if v["case"]["id"] == "A_w0" and v["seed"] == seed)
            lc = next(v for v in deep if v["case"]["id"] == f"A_w{case['boundary_weight']}" and v["seed"] == seed)
            deltas.append({
                "seed": seed,
                "boundary_miou_pp": 100 * (x["boundary_miou"] - a["boundary_miou"]),
                "whole_miou_pp": 100 * (x["miou"] - a["miou"]),
                "forest_f1_pp": 100 * (x["forest_macro_f1"] - a["forest_macro_f1"]),
                "architecture_boundary_gain_over_matched_loss_pp": 100 * (x["boundary_miou"] - lc["boundary_miou"]),
            })
        means = {key: sum(v[key] for v in deltas) / 3 for key in deltas[0] if key != "seed"}
        passed = (
            means["boundary_miou_pp"] >= 1
            and all(v["boundary_miou_pp"] > 0 for v in deltas)
            and means["whole_miou_pp"] >= 0
            and means["forest_f1_pp"] >= -1
        )
        comparisons.append({
            "case": case,
            "paired_deltas": deltas,
            "means": means,
            "development_S0_numerical_gate": passed,
            "three_seed_winner_search_not_confirmatory": True,
        })
    write_json(COMPLETION, {
        "status": "COMPLETED_DEVELOPMENT_ONLY",
        "screen_trials": len(json.loads(SCREEN.read_text(encoding="utf-8"))),
        "deep_stage_entries": len(deep),
        "comparisons": comparisons,
        "cloud_combination_training_started": False,
        "full_allowed": False,
        "official_test_allowed": False,
    })
    write_json(STATUS, {"status": "COMPLETED_DEVELOPMENT_ONLY", "updated_unix": time.time()})
    print(json.dumps({"deep_entries": len(deep), "comparisons": len(comparisons)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "lane", "finalize"])
    parser.add_argument("--lane", choices=["A", "B", "C"])
    args = parser.parse_args()
    if args.action == "lane":
        if not args.lane:
            parser.error("--lane is required for lane")
        lane(args.lane)
    elif args.action == "prepare":
        prepare()
    else:
        finalize()
