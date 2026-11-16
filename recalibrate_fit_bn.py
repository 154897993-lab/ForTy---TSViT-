"""Train-input-only BN diagnostic; never overwrites a training checkpoint.

Changes running statistics, not learned parameters. Export is explicitly a
diagnostic postprocessed candidate, not a replacement acceptance result.
"""
import argparse
import json
from pathlib import Path
import torch
from data import ForTyDataset, read_manifest
from forest_unet import ForestUNet
from forty_adapter import flatten_s2
from pipeline import Clouds, evaluate_loader, loader, seed_everything, sha256, write_json
from train import save_checkpoint


def recalibrate(package, checkpoint, output, device="cpu", batch=8):
    root, source, out = Path(package), Path(checkpoint), Path(output)
    checks = json.loads((root / "CHECKSUMS.json").read_text(encoding="utf-8-sig"))
    if any(sha256(root / name) != value for name, value in checks.items()):
        raise ValueError("Input dataset checksum mismatch")
    saved = torch.load(source, map_location="cpu", weights_only=True)
    if saved["provenance"]["train_manifest_sha256"] != sha256(root / "manifest_train32.csv"):
        raise ValueError("Checkpoint and calibration training IDs disagree")
    if saved["provenance"]["cloud_mode"] != "none":
        raise ValueError("This diagnostic is defined for clean fit checkpoints")
    if saved["provenance"]["model_source_sha256"] != sha256(Path(__file__).parent / "forest_unet.py"):
        raise ValueError("Model source mismatch")
    rows = read_manifest(root / "manifest_train32.csv")
    if len(rows) != 32 or any(row["split"] != "train" for row in rows):
        raise ValueError("Only the selected 32 training records may calibrate BN")
    dataset = ForTyDataset(rows, root / "prepared", "train", storage="prepared", divisor=1,
                          cache=True, index_sha256=checks["prepared/index.csv"])
    seed_everything(saved["provenance"]["train_seed"], True)
    torch.set_num_threads(4)
    model = ForestUNet(**saved["model_config"]).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    cfg = {"training": {"eval_batch_size": batch, "workers": 0},
           "evaluation": {"boundary_width": 0}, "clouds": {"provider": None}}
    clouds = Clouds(cfg, required=False)
    before = evaluate_loader(model, loader(dataset, cfg), cfg, clouds, "clean", 0, device)
    original = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    parameters = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    source_hash = sha256(source)
    model.train()
    normalization = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    if not normalization:
        raise ValueError("No BatchNorm layers")
    momenta = [module.momentum for module in normalization]
    for module in normalization:
        module.reset_running_stats()
        module.momentum = None
    with torch.no_grad():
        for item in loader(dataset, cfg):
            model(flatten_s2(item["s2"].to(device)))  # labels are not used to update statistics
    for module, momentum in zip(normalization, momenta):
        module.momentum = momentum
    if any(not torch.equal(value.detach().cpu(), parameters[name]) for name, value in model.named_parameters()):
        raise ValueError("Calibration unexpectedly changed learned parameters")
    changed = [name for name, value in model.state_dict().items() if not torch.equal(value.cpu(), original[name])]
    if any(not name.endswith(("running_mean", "running_var", "num_batches_tracked")) for name in changed):
        raise ValueError("Calibration changed state outside BatchNorm buffers")
    after = evaluate_loader(model, loader(dataset, cfg), cfg, clouds, "clean", 0, device)
    out.mkdir(parents=True, exist_ok=False)
    exported = {"model_version": saved["model_version"], "model_config": saved["model_config"],
                "state_dict": model.state_dict(), "epoch": saved["epoch"],
                "provenance": {**saved["provenance"], "scope": "DIAGNOSTIC_BN_POSTPROCESS_NOT_FORMAL_ACCEPTANCE",
                    "postprocessing": {"kind": "BN cumulative-average recalibration", "batch": batch,
                        "samples": 32, "data_split": "train", "source_checkpoint_sha256": source_hash,
                        "source_sha256": sha256(__file__)}}, "train_metrics": after}
    save_checkpoint(out / "bn_calibrated.pt", exported)
    restored = ForestUNet(**saved["model_config"]).to(device)
    restored.load_state_dict(torch.load(out / "bn_calibrated.pt", map_location=device, weights_only=True)["state_dict"], strict=True)
    verified = evaluate_loader(restored, loader(dataset, cfg), cfg, clouds, "clean", 0, device)
    if verified["confusion"] != after["confusion"] or sha256(source) != source_hash:
        raise ValueError("Reload/source preservation check failed")
    result = {"scope": "DIAGNOSTIC_POSTPROCESS_ONLY", "before_miou": before["whole"]["miou"],
              "after_miou": after["whole"]["miou"], "learned_parameters_unchanged": True,
              "changed_BN_buffer_keys": changed, "original_checkpoint_unchanged": True,
              "strict_reload_confusion_equal": True, "optimizer_updates": 0,
              "accepted": False, "formal_protocol_changed": False, "validation_or_test_used": False}
    write_json(out / "bn_diagnostic.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(recalibrate(args.package, args.checkpoint, args.output, args.device, args.batch)), flush=True)
