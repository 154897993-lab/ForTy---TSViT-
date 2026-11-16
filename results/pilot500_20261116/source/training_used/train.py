"""Independent training entry; no historical trainer is used."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from forty_adapter import IGNORE_INDEX, flatten_s2, encode_labels
from forest_unet import MODEL_VERSION
from model import GROUPS, build, delivery_group, trunk_hash
from pipeline import (Clouds, delivery_metadata, evaluate_loader, load_protocol, loader,
                      make_data, resolve, seed_everything, sha256, source_hashes, write_json)


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def build_provenance(cfg, group, seed, fingerprint, hashes, device):
    """Flat provenance compatible with the shared evaluator contract.

    protocol_sha256 is the delivery protocol hash when a delivery block is
    present; runtime_protocol_sha256 pins this project's runtime config. The
    two are recorded separately so neither side silently changes the other.
    """
    delivery = delivery_metadata(cfg)
    settings = cfg["training"]
    provenance = {
        "group": delivery_group(group),
        "train_seed": seed,
        "scope": delivery["scope"] if delivery else cfg["purpose"],
        "protocol_sha256": delivery["protocol_sha256"] if delivery else fingerprint,
        "runtime_protocol_sha256": fingerprint,
        "train_manifest_sha256": sha256(resolve(cfg, cfg["data"]["manifest"])),
        "model_source_sha256": hashes["forest_unet.py"],
        "source_sha256": hashes,
        "device": device,
        "epochs": settings["epochs"],
        "batch": settings["batch_size"],
        "optimizer": "Adam",
        "betas": list(settings.get("betas", [0.9, 0.9999])),
        "eps": settings.get("eps", 1e-8),
        "lr": settings["lr"],
        "weight_decay": settings["weight_decay"],
        "precision": settings["precision"],
        "grad_clip_norm": settings.get("grad_clip_norm", 10.0),
        "selection_metric": settings["select_metric"],
        "cloud_mode": GROUPS[group][0],
        "cloud_probability": (cfg.get("clouds", {}).get("settings") or {}).get("probability"),
    }
    if delivery:
        provenance.update({
            "expected_eval_manifest_sha256": delivery["eval_manifest_sha256"],
            "engine_source_sha256": delivery["engine_source_sha256"],
            "training_cloud_source_sha256": delivery["training_cloud_source_sha256"],
        })
    else:
        provenance.update({
            "expected_eval_manifest_sha256": cfg.get("evaluation", {}).get("manifest_sha256"),
            "engine_source_sha256": cfg.get("clouds", {}).get("engine_source_sha256"),
            "training_cloud_source_sha256": cfg.get("clouds", {}).get("training_cloud_source_sha256"),
        })
    return provenance


def train(config, group, seed, output, device="cpu", resume=False, stop_after=None):
    cfg, fingerprint = load_protocol(config)
    if group not in GROUPS or seed not in cfg["training_seeds"]:
        raise ValueError("Group/seed must belong to the frozen experiment matrix")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    if device.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    settings = cfg["training"]
    torch.set_num_threads(settings["cpu_threads"])
    seed_everything(seed, settings["deterministic"])
    clouds = Clouds(cfg, GROUPS[group][0] != "clean")
    train_data, val_data = make_data(cfg, "train"), make_data(cfg, "validation")
    run = Path(output).resolve()
    hashes = source_hashes()
    provenance = build_provenance(cfg, group, seed, fingerprint, hashes, device)
    net = build(group, settings["base_channels"]).to(device)
    initial_trunk = trunk_hash(net)
    provenance["initial_trunk_sha256"] = initial_trunk
    optimizer = torch.optim.Adam(net.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"],
                                 betas=tuple(settings.get("betas", [0.9, 0.9999])),
                                 eps=settings.get("eps", 1e-8))
    start, best, history = 0, -float("inf"), []
    if resume:
        checkpoint = torch.load(run / "last.pt", map_location=device, weights_only=True)
        saved_prov = checkpoint.get("provenance", {})
        for key, expected in (("runtime_protocol_sha256", fingerprint), ("source_sha256", hashes),
                              ("group", delivery_group(group)), ("train_seed", seed)):
            actual = saved_prov.get(key, checkpoint.get(key))
            if actual != expected:
                raise ValueError(f"Resume mismatch: {key}; use a new run instead")
        net.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        provenance["initial_trunk_sha256"] = saved_prov.get("initial_trunk_sha256", initial_trunk)
        start, best, history = checkpoint["epoch"], checkpoint["best_score"], checkpoint["history"]
    else:
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "protocol.json", cfg)
        write_json(run / "run.json", {"group": group, "delivery_group": delivery_group(group),
            "seed": seed, "training_cloud": GROUPS[group][0],
            "runtime_protocol_sha256": fingerprint, "provenance": provenance, "purpose": cfg["purpose"],
            "scientific_evidence": cfg["purpose"] == "research", "device": device,
            "torch": str(torch.__version__), "sample_counts": {"train": len(train_data), "validation": len(val_data)},
            "parameters": net.parameter_counts()})
    final_epoch = min(settings["epochs"], stop_after) if stop_after is not None else settings["epochs"]
    if final_epoch <= start:
        raise ValueError("No remaining epochs; resume will not silently rerun a completed job")
    with (run / "cloud_audit.jsonl").open("a", encoding="utf-8") as audit:
        for epoch in range(start + 1, final_epoch + 1):
            # Explicit epoch seeds make an interrupted epoch-boundary resume reproducible.
            seed_everything(seed + epoch, settings["deterministic"])
            net.train()
            begin = time.perf_counter()
            loss_sum, valid_total, updates, skipped = 0.0, 0, 0, 0
            for batch in loader(train_data, cfg, training=True, seed=seed + epoch):
                image, raw = batch["s2"].to(device), batch["label"].to(device)
                target = encode_labels(raw)
                valid = target != IGNORE_INDEX
                count = int(valid.sum())
                if count == 0:
                    skipped += 1
                    continue  # no Adam momentum/weight-decay update on all-ignore batches
                affected, _, details = clouds.apply(image, batch["id"], GROUPS[group][0], "training", seed, epoch)
                audit.write(json.dumps(details, allow_nan=False) + "\n")
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                                    enabled=settings["precision"] == "bfloat16"):
                    logits = net(flatten_s2(affected)).float()
                    # Explicit masked logsumexp CE, one ignore convention throughout.
                    scores = logits.movedim(1, -1)[valid]
                    labels = target[valid]
                    loss = (torch.logsumexp(scores, dim=1) - scores.gather(1, labels[:, None])[:, 0]).mean()
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in net.parameters()):
                    raise ValueError("Non-finite gradient")
                torch.nn.utils.clip_grad_norm_(net.parameters(), settings.get("grad_clip_norm", 10.0),
                                               error_if_nonfinite=True)
                optimizer.step()
                loss_sum += loss.item() * count
                valid_total += count
                updates += 1
            if not updates:
                raise ValueError("Epoch had no valid optimizer updates")
            # True end-of-epoch clean train-set evaluation, not online training accuracy.
            train_score = evaluate_loader(net, loader(train_data, cfg), cfg, clouds, "clean",
                                          cfg["evaluation"]["seed"], device) if settings["measure_train"] else None
            validation = evaluate_loader(net, loader(val_data, cfg), cfg, clouds, "clean",
                                         cfg["evaluation"]["seed"], device)
            score = validation["whole"][settings["select_metric"]]
            improved = score > best
            best = max(best, score)
            record = {"epoch": epoch, "loss": loss_sum / valid_total, "updates": updates,
                      "skipped_all_ignore_batches": skipped, "train_clean": train_score,
                      "validation_clean": validation, "seconds": time.perf_counter() - begin,
                      "selected_metric": settings["select_metric"], "score": score, "best_score": best}
            history.append(record)
            payload = {"format_version": 2, "model_version": MODEL_VERSION,
                       "model_config": net.config(), "state_dict": net.state_dict(),
                       "optimizer": optimizer.state_dict(), "epoch": epoch,
                       "group": delivery_group(group), "seed": seed, "best_score": best, "history": history,
                       "provenance": provenance, "purpose": cfg["purpose"]}
            if improved:
                save_checkpoint(run / "best.pt", payload)
            save_checkpoint(run / "last.pt", payload)
            write_json(run / "history.json", history)
            print(json.dumps({"group": group, "seed": seed, "epoch": epoch,
                              "loss": record["loss"], "validation": validation["whole"],
                              "train_miou": train_score["whole"]["miou"] if train_score else None}), flush=True)
    return run / "best.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--group", required=True, choices=tuple(GROUPS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, help="Explicit boundary stop; protocol epochs remain unchanged")
    args = parser.parse_args()
    train(args.config, args.group, args.seed, args.output, args.device, args.resume, args.stop_after_epoch)


if __name__ == "__main__":
    main()
