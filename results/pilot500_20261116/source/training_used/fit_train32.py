"""Independent clean train-only 32-sample fit; no validation/test/model selection."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from data import ForTyDataset, read_manifest, equal_class_batches
from forest_unet import MODEL_VERSION
from forty_adapter import IGNORE_INDEX, flatten_s2, encode_labels
from model import build
from pipeline import (Clouds, evaluate_loader, loader, seed_everything, sha256,
                      source_hashes, write_json)
from train import save_checkpoint


FORMAL_FROZEN = {"group": "D", "epochs": 50, "batch": 32,
                 "batch_order": "class_balanced", "lr": 0.001,
                 "seed": 20261113, "lr_schedule": "constant"}


def fit(package, output, device="cpu", epochs=50, batch=8, lr=0.001, seed=20261113,
        threads=4, resume=False, group="D", lr_schedule="constant", diagnostic=False,
        batch_order="random", frozen_acceptance=False):
    root, out = Path(package).resolve(), Path(output).resolve()
    if min(epochs, batch, threads) < 1 or lr <= 0:
        raise ValueError("Invalid epochs/batch/threads/lr")
    if group not in ("A", "D") or lr_schedule not in ("constant", "cosine"):
        raise ValueError("Clean fit supports A/D and constant/cosine only")
    if batch_order not in ("random", "class_balanced"):
        raise ValueError("Unknown fit batch order")
    if frozen_acceptance:
        actual = {"group": group, "epochs": epochs, "batch": batch,
                  "batch_order": batch_order, "lr": lr, "seed": seed,
                  "lr_schedule": lr_schedule}
        if actual != FORMAL_FROZEN:
            raise ValueError(f"Frozen acceptance config mismatch: {actual}")
        diagnostic = False
    else:
        diagnostic = bool(diagnostic or group != "D" or lr_schedule != "constant" or batch_order != "random")
    checks = json.loads((root / "CHECKSUMS.json").read_text(encoding="utf-8-sig"))
    for name, expected in checks.items():
        if sha256(root / name) != expected:
            raise ValueError(f"Input package checksum mismatch: {name}")
    selected = read_manifest(root / "manifest_train32.csv")
    frozen = read_manifest(root / "frozen_split_manifest_3500.csv")
    known = {(r["sample_id"], r["shard"], r["byte_offset"]) for r in frozen if r["split"] == "train"}
    if len(selected) != 32 or any(r["split"] != "train" or
            (r["sample_id"], r["shard"], r["byte_offset"]) not in known for r in selected):
        raise ValueError("Fit requires the exact 32 selected frozen training records")
    source_checks_path = root / "provenance" / "ORIGINAL_CHECKSUMS.json"
    source_checks = json.loads(source_checks_path.read_text(encoding="utf-8")) if source_checks_path.exists() else checks
    if source_checks["forest_unet.py"] != sha256(Path(__file__).parent / "forest_unet.py"):
        raise ValueError("Model source differs from the delivered model")
    dataset = ForTyDataset(selected, root / "prepared", "train", storage="prepared", divisor=1.0,
                          cache=True, index_sha256=checks["prepared/index.csv"])
    class_counts = torch.zeros(9, dtype=torch.long)
    dominant_classes = []
    for index in range(len(dataset)):
        counts = torch.bincount(dataset[index]["label"].flatten(), minlength=9)
        class_counts += counts
        dominant_classes.append(int(counts[1:].argmax()) if int(counts[1:].sum()) else -1)
    if not bool((class_counts[1:] > 0).all()):
        raise ValueError("Every real class must have training support")
    torch.set_num_threads(threads)
    seed_everything(seed, deterministic=True)
    net = build(group).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.9999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01) if lr_schedule == "cosine" else None
    # Only these train rows are ever materialized; no val/test loader exists here.
    cfg = {"training": {"batch_size": batch, "eval_batch_size": batch, "workers": 0},
           "evaluation": {"boundary_width": 0}, "clouds": {"provider": None}}
    clouds = Clouds(cfg, required=False)
    sources = source_hashes()
    reference = json.loads((root / "source_hashes.json").read_text(encoding="utf-8"))
    provenance = {"group": group, "train_seed": seed,
        "scope": "DIAGNOSTIC_TRAIN32_NOT_FORMAL_ACCEPTANCE" if diagnostic else "ACCEPTANCE50_FROZEN_TRAIN",
        "epochs": epochs, "batch": batch, "lr": lr, "optimizer": "Adam", "betas": [0.9, 0.9999],
        "lr_schedule": lr_schedule, "lr_min": lr * 0.01 if scheduler else lr,
        "diagnostic": diagnostic,
        "frozen_acceptance": frozen_acceptance,
        "frozen_config": FORMAL_FROZEN if frozen_acceptance else None,
        "batch_order": batch_order,
        "eps": 1e-8, "gradient_clip_norm": 10.0, "precision": "float32", "loss": "CE_ignore_-1",
        "cloud_mode": "none", "cloud_probability": 0.0, "cloud_engine_used": False,
        "checkpoint_selection": "last epoch; no validation-based selection",
        "train_manifest_sha256": sha256(root / "manifest_train32.csv"),
        "parent_manifest_sha256": sha256(root / "frozen_split_manifest_3500.csv"),
        "prepared_index_sha256": checks["prepared/index.csv"],
        "model_source_sha256": sha256(Path(__file__).parent / "forest_unet.py"),
        "protocol_sha256": sha256(root / "protocol_v1.1_reference.json"),
        "engine_source_sha256": reference["engine_source_sha256"],
        "training_cloud_source_sha256": reference["training_cloud_source_sha256"],
        "source_sha256": sources, "device": device, "threads": threads, "torch": str(torch.__version__),
        "sample_ids": [r["sample_id"] for r in selected], "data_order": f"{batch_order}, seed + epoch",
        "joint_six_group_protocol_frozen": False}
    start, history = 0, []
    if resume:
        saved = torch.load(out / "last.pt", map_location=device, weights_only=True)
        if saved["provenance"] != provenance:
            raise ValueError("Fit resume metadata/source mismatch")
        net.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(saved["scheduler_state_dict"])
        start, history = saved["epoch"], saved["history"]
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "run_config.json", provenance)
        if frozen_acceptance:
            write_json(out / "frozen_acceptance_config.json",
                       {"config": FORMAL_FROZEN, "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        write_json(out / "input_checks.json", {"files_verified": len(checks), "samples": 32,
            "class_pixels_raw0to8": class_counts.tolist(), "validation_samples_used": 0,
            "test_samples_used": 0, "divisor_applied_to_prepared": 1.0})
    if start >= epochs:
        raise ValueError("No remaining epochs")
    begin = time.perf_counter()
    for epoch in range(start + 1, epochs + 1):
        seed_everything(seed + epoch, deterministic=True)
        net.train()
        epoch_lr = optimizer.param_groups[0]["lr"]
        loss_sum, pixels = 0.0, 0
        batches = (DataLoader(dataset, batch_sampler=equal_class_batches(dominant_classes, batch, seed + epoch),
                              num_workers=0, generator=torch.Generator().manual_seed(seed + epoch))
                   if batch_order == "class_balanced" else loader(dataset, cfg, training=True, seed=seed + epoch))
        for item in batches:
            target = encode_labels(item["label"].to(device))
            valid = target != IGNORE_INDEX
            count = int(valid.sum())
            if not count:
                continue
            logits = net(flatten_s2(item["s2"].to(device)))
            scores = logits.movedim(1, -1)[valid]
            labels = target[valid]
            loss = (torch.logsumexp(scores, 1) - scores.gather(1, labels[:, None])[:, 0]).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite fit loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item() * count
            pixels += count
        if not pixels:
            raise ValueError("No valid training pixels")
        if scheduler is not None:
            scheduler.step()
        score = evaluate_loader(net, loader(dataset, cfg), cfg, clouds, "clean", seed, device)
        row = {"epoch": epoch, "lr": epoch_lr, "train_ce": loss_sum / pixels, "train_miou": score["whole"]["miou"],
               "class_iou": score["whole"]["iou"], "elapsed_seconds": time.perf_counter() - begin}
        history.append(row)
        saved = {"model_version": MODEL_VERSION, "model_config": net.config(),
                 "state_dict": net.state_dict(), "epoch": epoch, "provenance": provenance,
                 "train_metrics": score, "optimizer_state_dict": optimizer.state_dict(), "history": history}
        if scheduler is not None:
            saved["scheduler_state_dict"] = scheduler.state_dict()
        save_checkpoint(out / "last.pt", saved)
        write_json(out / "history.json", history)
        with (out / "train_log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)
    # Acceptance always refers to the last epoch, never the best intermediate score.
    save_checkpoint(out / "fit50.pt", {k: v for k, v in saved.items() if k != "optimizer_state_dict"})
    restored = build(group).to(device)
    restored.load_state_dict(torch.load(out / "fit50.pt", map_location=device, weights_only=True)["state_dict"], strict=True)
    verified = evaluate_loader(restored, loader(dataset, cfg), cfg, clouds, "clean", seed, device)
    if verified["confusion"] != score["confusion"]:
        raise ValueError("Reloaded checkpoint metrics disagree")
    write_json(out / "fit_acceptance.json", {"scope": provenance["scope"],
        "checkpoint_epoch": epochs, "samples": 32, "use_edge": group == "D", "no_added_cloud": True,
        "train_miou": verified["whole"]["miou"], "target_miou": 0.95,
        "threshold_met": epochs >= 50 and verified["whole"]["miou"] >= 0.95,
        "diagnostic": diagnostic,
        "frozen_acceptance": frozen_acceptance,
        "frozen_config": FORMAL_FROZEN if frozen_acceptance else None,
        "accepted": not diagnostic and group == "D" and epochs >= 50 and verified["whole"]["miou"] >= 0.95,
        "per_class_iou": verified["whole"]["iou"], "support": verified["whole"]["support"],
        "checkpoint_sha256": sha256(out / "fit50.pt"), "strict_reload_confusion_equal": True,
        "validation_or_test_used": False, "total_seconds_this_process": time.perf_counter() - begin})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--group", choices=("A", "D"), default="D")
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--diagnostic", action="store_true", help="Never promote diagnostic scores to formal acceptance")
    parser.add_argument("--batch-order", choices=("random", "class_balanced"), default="random")
    parser.add_argument("--frozen-acceptance", action="store_true",
                        help="Formal acceptance only for the frozen batch32/class-balanced 50-epoch config")
    args = parser.parse_args()
    fit(args.package, args.output, args.device, args.epochs, args.batch, args.lr, args.seed,
        args.threads, args.resume, args.group, args.lr_schedule, args.diagnostic,
        args.batch_order, args.frozen_acceptance)
