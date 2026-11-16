"""Benchmark the actual deterministic training step without creating research runs.

Run from an isolated source snapshot, using an existing frozen runtime config.
No validation/test loader is constructed, and benchmark weights are discarded.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import time

import torch
from forty_adapter import IGNORE_INDEX, encode_labels, flatten_s2
from model import build
from pipeline import Clouds, load_protocol, loader, make_data, seed_everything, sha256, source_hashes, write_json
import ruan_cloud_provider as provider


def training_step(net, optimizer, batch, clouds, kind, seed, epoch, clip):
    image, target = batch["s2"].cuda(), encode_labels(batch["label"].cuda())
    valid = target != IGNORE_INDEX
    torch.cuda.synchronize()
    begin = time.perf_counter()
    affected, _, _ = clouds.apply(image, batch["id"], kind, "training", seed, epoch)
    torch.cuda.synchronize()
    augmentation_seconds = time.perf_counter() - begin
    optimizer.zero_grad(set_to_none=True)
    logits = net(flatten_s2(affected)).float()
    scores, labels = logits.movedim(1, -1)[valid], target[valid]
    loss = (torch.logsumexp(scores, 1) - scores.gather(1, labels[:, None])[:, 0]).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("Non-finite benchmark loss")
    loss.backward()
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in net.parameters()):
        raise ValueError("Non-finite benchmark gradient")
    torch.nn.utils.clip_grad_norm_(net.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    return augmentation_seconds, loss.item()


def benchmark(config, output, baseline_provider=None, steps=6, warmup=2, workers=4):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg, fingerprint = load_protocol(config)
    settings = cfg["training"]
    if settings["precision"] != "float32" or settings["deterministic"] is not True:
        raise ValueError("Benchmark requires the agreed float32 deterministic protocol")
    torch.set_num_threads(settings["cpu_threads"])
    seed = cfg["training_seeds"][0]
    seed_everything(seed, True)
    dataset = make_data(cfg, "train")
    for i in range(len(dataset)):
        dataset[i]  # checksum verification and cache warmup outside measured step
    updated = copy.deepcopy(cfg)
    updated["clouds"]["sha256"] = sha256(Path(provider.__file__))
    updated["clouds"]["settings"]["cpu_workers"] = workers
    parallel = Clouds(updated, True)
    serial_cfg = copy.deepcopy(updated)
    serial_cfg["clouds"]["settings"]["cpu_workers"] = 0
    serial = Clouds(serial_cfg, True)
    old = serial
    if baseline_provider:
        spec = importlib.util.spec_from_file_location("historical_cloud_adapter", baseline_provider)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        old_cfg = copy.deepcopy(cfg)
        old = Clouds(old_cfg, False)
        old.provider = module.augment
    result = {"scope": "PERFORMANCE_REVIEW_NOT_RESEARCH_TRAINING", "gpu": torch.cuda.get_device_name(0),
              "torch": str(torch.__version__), "runtime_protocol_sha256": fingerprint,
              "benchmark_source_sha256": sha256(Path(__file__)), "source_sha256": source_hashes(),
              "baseline_provider_sha256": sha256(baseline_provider) if baseline_provider else None,
              "workers": workers, "warmup": warmup, "measured_steps": steps,
              "deterministic": True, "precision": "float32", "equivalence": [], "measurements": [],
              "notes": ["Cached training batches; no validation, checkpoints, history I/O or cloud-audit file I/O in step timings.",
                        "Same masked logsumexp CE, Adam settings, finite checks and gradient clipping as train.py.",
                        "Each timed step uses a different training batch/epoch; all modes start with the same weights.",
                        "GPU memory is PyTorch peak allocated/reserved; CPU worker RSS is read separately."]}
    test_cfg = copy.deepcopy(cfg); test_cfg["training"]["batch_size"] = 32
    batch = next(iter(loader(dataset, test_cfg)))
    for kind in ("independent", "overlap"):
        for epoch in (1, 2):
            image = batch["s2"].cuda()
            args = dict(image=image, sample_ids=batch["id"].tolist(), kind=kind, scope="training",
                        seed=seed, epoch=epoch)
            a = old.provider(**args, settings=cfg["clouds"]["settings"])
            for label, settings_ in (("serial", serial_cfg["clouds"]["settings"]),
                                     ("parallel", updated["clouds"]["settings"])):
                begin = time.perf_counter()
                b = provider.augment(**args, settings=settings_)
                torch.cuda.synchronize()
                assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2]
                result["equivalence"].append({"kind":kind,"epoch":epoch,"mode":label,"samples":32,
                                               "image_mask_metadata_bitwise_equal":True,
                                               "seconds_including_first_pool_start":time.perf_counter()-begin})
    # Two true updates, including BN buffers and Adam state, must also agree.
    states, losses = [], []
    for clouds in (old, parallel):
        seed_everything(seed, True)
        net = build("B_ind+D").cuda().train()
        optimizer = torch.optim.Adam(net.parameters(), lr=settings["lr"],
            betas=tuple(settings["betas"]), eps=settings["eps"], weight_decay=settings["weight_decay"])
        values = [training_step(net,optimizer,batch,clouds,"independent",seed,e,settings["grad_clip_norm"])[1]
                  for e in (1,2)]
        states.append(({k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                       copy.deepcopy(optimizer.state_dict())))
        losses.append(values)
        del optimizer, net
    assert losses[0] == losses[1]
    assert all(torch.equal(states[0][0][k],states[1][0][k]) for k in states[0][0])
    for key, state in states[0][1]["state"].items():
        assert all(torch.equal(v,states[1][1]["state"][key][k]) if isinstance(v,torch.Tensor)
                   else v == states[1][1]["state"][key][k] for k,v in state.items())
    result["two_update_model_bn_optimizer_bitwise_equal"] = True
    del states, a, b, args, image, batch, values, losses, state
    torch.cuda.empty_cache()
    for bs in (8,16,32):
        test_cfg["training"]["batch_size"] = bs
        batches = list(loader(dataset,test_cfg,training=True,seed=seed+1))[:steps+warmup]
        if len(batches) < steps + warmup or any(len(b["id"]) != bs for b in batches):
            raise ValueError("Insufficient full training batches for requested timing steps")
        for kind, label, clouds in (("clean","clean",serial),("independent","serial",old),
                                   ("independent","parallel",parallel),("overlap","serial",old),
                                   ("overlap","parallel",parallel)):
            seed_everything(seed,True)
            net = build("B_ind+D").cuda().train()
            optimizer = torch.optim.Adam(net.parameters(),lr=settings["lr"],betas=tuple(settings["betas"]),
                                         eps=settings["eps"],weight_decay=settings["weight_decay"])
            for i in range(warmup):
                training_step(net,optimizer,batches[i],clouds,kind,seed,i+1,settings["grad_clip_norm"])
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter(); aug = 0.0
            for i in range(steps):
                value,_ = training_step(net,optimizer,batches[i+warmup],clouds,kind,seed,i+warmup+1,
                                        settings["grad_clip_norm"])
                aug += value
            torch.cuda.synchronize(); elapsed = time.perf_counter()-begin
            row = {"batch":bs,"kind":kind,"implementation":label,"samples":bs*steps,
                   "step_ms":elapsed*1000/steps,"augment_ms":aug*1000/steps,
                   "samples_per_second":bs*steps/elapsed,
                   "peak_allocated_mb":torch.cuda.max_memory_allocated()/1024**2,
                   "peak_reserved_mb":torch.cuda.max_memory_reserved()/1024**2}
            result["measurements"].append(row); print(json.dumps(row),flush=True)
            del optimizer, net
    children = []
    for pool in provider._POOLS.values():
        for process in pool._processes.values():
            path = Path(f"/proc/{process.pid}/status")
            if path.exists():
                children.append({"pid":process.pid,"VmRSS_kB":int(next(
                    line.split()[1] for line in path.read_text().splitlines() if line.startswith("VmRSS:")))})
    result["cpu_worker_memory"] = children
    write_json(output,result)
    print(json.dumps({"result":str(output),"equivalence":True}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--baseline-provider")
    parser.add_argument("--steps",type=int,default=6)
    parser.add_argument("--warmup",type=int,default=2)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1:
        parser.error("steps and warmup must be positive")
    benchmark(args.config,args.output,args.baseline_provider,args.steps,args.warmup,args.workers)
