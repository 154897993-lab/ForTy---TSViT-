"""Run a selected six-group/seed matrix sequentially, stopping on the first error."""
import argparse
from pathlib import Path
from evaluate import evaluate
from model import GROUPS
from pipeline import Clouds, load_protocol
from train import train
from summarize_runs import summarize_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", nargs="+", choices=tuple(GROUPS), default=list(GROUPS))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--export-predictions", action="store_true")
    args = parser.parse_args()
    cfg, _ = load_protocol(args.config)
    if len(set(args.groups)) != len(args.groups) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate group/seed")
    if any(seed not in cfg["training_seeds"] for seed in args.seeds):
        raise ValueError("All matrix seeds must be declared before starting any training")
    # All four evaluation scenes require the provider, even for clean-trained models.
    Clouds(cfg, required=True)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    checkpoints = []
    for seed in args.seeds:
        for group in args.groups:
            checkpoints.append(train(args.config, group, seed, root / f"{group}_seed{seed}", args.device))
    evaluate(args.config, checkpoints, root / "evaluation", device=args.device,
             export_predictions=args.export_predictions)
    summarize_files([root / "evaluation" / "metrics.json"], root / "summary")


if __name__ == "__main__":
    main()
