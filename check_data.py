"""Preflight selected train/validation samples without a GPU or test access."""
import argparse
from pathlib import Path
import torch
from pipeline import load_protocol, make_data, write_json


def check_data(config, output):
    cfg, fingerprint = load_protocol(config)
    result = {"protocol_version": cfg["version"], "protocol_sha256": fingerprint,
              "purpose": cfg["purpose"], "splits": {}}
    for split in ("train", "validation"):
        dataset = make_data(cfg, split)
        pixels, all_ignore, low, high = torch.zeros(9, dtype=torch.int64), 0, float("inf"), -float("inf")
        for index in range(len(dataset)):
            item = dataset[index]  # validates ID, CRC, shape, label range and finite S2
            pixels += torch.bincount(item["label"].flatten(), minlength=9)
            all_ignore += int(not bool((item["label"] > 0).any()))
            low, high = min(low, item["s2"].min().item()), max(high, item["s2"].max().item())
        if not int(pixels[1:].sum()) or (split == "validation" and not bool((pixels[1:4] > 0).all())):
            raise ValueError("No valid pixels or a forest class has no validation support")
        result["splits"][split] = {"samples": len(dataset), "class_pixels_raw_0_to_8": pixels.tolist(),
            "all_ignore_samples": all_ignore, "normalized_s2_min": low, "normalized_s2_max": high}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "data_check.json", result)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    check_data(args.config, args.output)
