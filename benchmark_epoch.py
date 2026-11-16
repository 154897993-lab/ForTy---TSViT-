"""Measure complete cached-data training epochs and verify cloud parallelism.

Uses train.py unchanged, including clean validation, checkpoints and audit I/O.
The four short runs are performance measurements, not paper experiment arms.
"""
import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import sys
import time
import torch
from pipeline import sha256, source_hashes, write_json
from train import train


class EpochLog:
    def __init__(self, stream, label):
        self.stream, self.label, self.pending, self.events = stream, label, "", []

    def write(self, value):
        self.stream.write(value)
        self.pending += value
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if line.strip():
                record = json.loads(line)
                self.events.append({"epoch": record["epoch"], "finished": time.perf_counter()})
                print(f"{self.label}: epoch {record['epoch']} saved", file=sys.__stdout__, flush=True)
        return len(value)

    def flush(self):
        self.stream.flush()


def state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return a == b


def benchmark(config, output, workers=4, epochs=3):
    if epochs < 3 or workers < 1:
        raise ValueError("Need at least three epochs and one parallel worker")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    base = json.loads(Path(config).read_text(encoding="utf-8-sig"))
    if not base["training"]["deterministic"] or base["training"]["precision"] != "float32":
        raise ValueError("Require the frozen deterministic fp32 settings")
    seed = base["training_seeds"][0]
    records, comparisons = [], []
    for group in ("B_ind+D", "B_ovl+D"):
        reference = None
        for count in (0, workers):
            label = f"{group.replace('+', '_')}_workers{count}"
            cfg = copy.deepcopy(base)
            cfg["version"] = "performance-acceptance-v1.0-" + label
            cfg["training"].update(epochs=epochs, batch_size=32, eval_batch_size=32, measure_train=False)
            cfg["clouds"]["sha256"] = sha256(Path(__file__).parent / "ruan_cloud_provider.py")
            cfg["clouds"]["settings"]["cpu_workers"] = count
            cfg["delivery"]["scope"] = "PERFORMANCE_ACCEPTANCE_NOT_PAPER_RESULTS"
            config_path = root / f"{label}.json"
            write_json(config_path, cfg)
            run = root / label
            begin = time.perf_counter()
            with (root / f"{label}.log").open("w", encoding="utf-8") as stream:
                log = EpochLog(stream, label)
                with contextlib.redirect_stdout(log):
                    train(config_path, group, seed, run, device="cuda")
            elapsed = time.perf_counter() - begin
            intervals = [b["finished"] - a["finished"] for a, b in zip(log.events, log.events[1:])]
            if len(intervals) != epochs - 1:
                raise ValueError("Missing completed-epoch timestamps")
            checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
            history = checkpoint["history"]
            if any(row["updates"] != 16 for row in history):
                raise ValueError("Expected every 500-sample epoch to make 16 batch32 updates")
            row = {"group": group, "cpu_workers": count, "epochs": epochs,
                   "train_samples": 500, "validation_samples": 500,
                   "run_wall_seconds_including_setup": elapsed,
                   "first_completed_epoch_seconds_including_setup": log.events[0]["finished"] - begin,
                   "steady_complete_epoch_seconds": intervals,
                   "mean_steady_complete_epoch_seconds": sum(intervals)/len(intervals),
                   "mean_steady_history_seconds": sum(x["seconds"] for x in history[1:])/(epochs-1),
                   "last_checkpoint_sha256": sha256(run / "last.pt"),
                   "config_sha256": sha256(config_path), "source_sha256": checkpoint["provenance"]["source_sha256"]}
            records.append(row)
            if reference is None:
                reference = (checkpoint, run, row)
            else:
                original, serial_run, serial_row = reference
                assert state_equal(original["state_dict"], checkpoint["state_dict"])
                assert state_equal(original["optimizer"], checkpoint["optimizer"])
                for old, new in zip(original["history"], history):
                    assert state_equal({k:v for k,v in old.items() if k != "seconds"},
                                       {k:v for k,v in new.items() if k != "seconds"})
                assert (serial_run / "cloud_audit.jsonl").read_bytes() == (run / "cloud_audit.jsonl").read_bytes()
                comparisons.append({"group":group,"samples_per_mode":500*epochs,"updates_per_mode":16*epochs,
                    "model_bn_optimizer_history_audit_exact":True,
                    "complete_epoch_speedup":serial_row["mean_steady_complete_epoch_seconds"]/row["mean_steady_complete_epoch_seconds"]})
    result = {"scope":"PERFORMANCE_ACCEPTANCE_NOT_PAPER_RESULTS","gpu":torch.cuda.get_device_name(0),
              "torch":str(torch.__version__),"benchmark_source_sha256":sha256(Path(__file__)),
              "source_sha256":source_hashes(),"measurements":records,"equivalence":comparisons,
              "official_test_accessed":False,
              "timing_includes":["train data loading","cloud generation","forward/backward/Adam",
                  "cloud audit writes","500-sample clean validation","best/last checkpoint writes","history writes"],
              "timing_excludes":["first-epoch setup in steady timings","fixed four-scene post-training evaluation","idle time"],
              "qualification":"Two steady epochs per condition; a short performance acceptance, not a 100-epoch convergence result."}
    write_json(root / "epoch_benchmark.json", result)
    print(json.dumps({"result":str(root / 'epoch_benchmark.json'),"equivalence":comparisons}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--epochs",type=int,default=3)
    args=parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    benchmark(args.config,args.output,args.workers,args.epochs)
