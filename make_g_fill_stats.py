"""Freeze the G per-band fill means from the current training manifest."""
import argparse
from pathlib import Path

from g_erase import compute_fill_stats, write_fill_stats
from pipeline import load_protocol, make_data, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg, _ = load_protocol(args.config)
    dataset = make_data(cfg, "train")
    stats = compute_fill_stats(dataset, manifest_sha256=sha256(
        (Path(cfg["_directory"]) / cfg["data"]["manifest"]).resolve()))
    write_fill_stats(args.output, stats)
    print({"output": str(Path(args.output).resolve()), "sample_count": stats["sample_count"],
           "pixel_count": stats["pixel_count"], "mean": stats["mean"]})


if __name__ == "__main__":
    main()
