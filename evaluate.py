"""Independent multi-checkpoint unified evaluation and raw prediction export."""
import argparse
import json
from pathlib import Path
import torch
from forest_unet import ForestUNet
from model import GROUPS
from pipeline import (SCENES, Clouds, evaluate_loader, load_protocol, loader, make_data,
                      seed_everything, sha256, source_hashes, write_json, write_score_rows)


def evaluate(config, checkpoints, output, scenes=SCENES, split="validation", device="cpu",
             export_predictions=False, confirm_test=False, test_approval=None):
    cfg, fingerprint = load_protocol(config)
    if split not in ("validation", "test") or not scenes or any(scene not in SCENES for scene in scenes):
        raise ValueError("Invalid split/scenes")
    if len(set(scenes)) != len(scenes):
        raise ValueError("Duplicate scenes")
    approval = None
    if split == "test":
        if not confirm_test or test_approval is None:
            raise ValueError("Official test is sealed; explicit approval file and confirmation are required")
        approval_path = Path(test_approval).resolve()
        approval = json.loads(approval_path.read_text(encoding="utf-8-sig"))
        if (approval.get("approved") is not True or not approval.get("decision_id")
                or not approval.get("approved_by") or approval.get("protocol_sha256") != fingerprint):
            raise ValueError("Invalid final-test approval")
        # Approval is separate, so unlocking test does not mutate the training protocol hash.
        cfg["data"]["test_manifest"] = str((approval_path.parent / approval["test_manifest"]).resolve())
        cfg["data"]["test_manifest_sha256"] = approval["test_manifest_sha256"]
        cfg["data"]["expected_counts"]["test"] = approval["expected_test_count"]
    if not checkpoints or len({str(Path(p).resolve()) for p in checkpoints}) != len(checkpoints):
        raise ValueError("Provide unique checkpoints")
    if approval is not None and sorted(approval["checkpoint_sha256"]) != sorted(sha256(p) for p in checkpoints):
        raise ValueError("Test approval must cover exactly this checkpoint matrix")
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    seed_everything(cfg["evaluation"]["seed"], cfg["training"]["deterministic"])
    clouds = Clouds(cfg, any(scene != "clean" for scene in scenes))
    loaded, keys = [], set()
    current_sources = source_hashes()
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        provenance = checkpoint.get("provenance", {})
        checkpoint_protocol = provenance.get(
            "runtime_protocol_sha256", checkpoint.get("runtime_protocol_sha256", checkpoint.get("protocol_sha256")))
        checkpoint_sources = provenance.get("source_sha256", checkpoint.get("source_sha256", {}))
        if checkpoint_protocol != fingerprint or checkpoint_sources != current_sources:
            raise ValueError("Checkpoint protocol/source differs from evaluation protocol")
        key = provenance.get("group", checkpoint.get("group")), provenance.get(
            "train_seed", checkpoint.get("seed"))
        if key in keys:
            raise ValueError("Duplicate group/seed checkpoints")
        keys.add(key)
        loaded.append((Path(path), checkpoint))
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", cfg)
    if split == "test":
        # A protocol-specific receipt covers the entire checkpoint matrix, not one model at a time.
        receipt = Path(cfg["_directory"]) / ("official_test_" + fingerprint + ".started")
        with receipt.open("x", encoding="utf-8") as stream:
            stream.write(str(out))
        write_json(out / "test_approval.json", approval)
    dataset = make_data(cfg, split, allow_test=split == "test")
    results, csv_rows = [], []
    with (out / "scene_audit.jsonl").open("x", encoding="utf-8") as audit:
        for path, checkpoint in loaded:
            provenance = checkpoint.get("provenance", {})
            net = ForestUNet(**checkpoint["model_config"]).to(device)
            net.load_state_dict(checkpoint.get("state_dict", checkpoint.get("model")), strict=True)
            group = provenance.get("group", checkpoint.get("group"))
            seed = provenance.get("train_seed", checkpoint.get("seed"))
            run_id = f"{group}_seed{seed}"
            for scene in scenes:
                prediction_dir = out / "predictions" / run_id / scene if export_predictions else None
                scores = evaluate_loader(net, loader(dataset, cfg), cfg, clouds, scene,
                    cfg["evaluation"]["seed"], device, prediction_dir, audit)
                metadata = {"run_id": run_id, "group": group, "training_cloud": GROUPS[group][0],
                    "training_seed": seed, "split": split, "scene": scene,
                    "scene_seed": cfg["evaluation"]["seed"], "protocol_version": cfg["version"],
                    "protocol_sha256": fingerprint, "checkpoint_sha256": sha256(path), "purpose": cfg["purpose"]}
                results.append({**metadata, "scores": scores})
                csv_rows.append((metadata, scores))
    write_json(out / "metrics.json", results)
    write_score_rows(out / "eval_manifest.csv", csv_rows)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument("--confirm-test", action="store_true")
    parser.add_argument("--test-approval", help="Separate approval JSON tied to protocol and checkpoint hashes")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.output, args.scene, args.split, args.device,
             args.export_predictions, args.confirm_test, args.test_approval)


if __name__ == "__main__":
    main()
