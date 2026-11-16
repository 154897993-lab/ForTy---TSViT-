"""v1.2: explicit evaluation/prepared bindings and finite-value checks."""
import csv,hashlib,json
from pathlib import Path
import torch
VERSION="first-batch-safety-v1.2"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def finite_state(state):
    for name,value in state.items():
        if not isinstance(value,torch.Tensor):raise ValueError("Non-tensor state entry: "+name)
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise FloatingPointError("Non-finite checkpoint tensor: "+name)
def checked_logits(logits):
    if not isinstance(logits,torch.Tensor) or logits.ndim!=4 or logits.shape[1:]!=(8,128,128):
        raise ValueError("Expected logits [B,8,128,128]")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite or invalid inference logits; no metrics written")
    return logits
def contract_for(prov,data_manifest,eval_manifest,require_checkpoint=True):
    path=Path(__file__).resolve().parent/"evaluation_contract_v1.2.json"
    contract=json.loads(path.read_text(encoding="utf-8"))
    scope=prov["scope"]
    if scope not in contract["scopes"]:raise ValueError("Scope not eligible for four-scene validation")
    expected=contract["scopes"][scope]
    if sha(data_manifest)!=expected["data_manifest_sha256"]:
        raise ValueError("Data manifest differs from pinned evaluation contract")
    if sha(eval_manifest)!=expected["eval_manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from pinned evaluation contract; bump version to change")
    if prov["train_manifest_sha256"]!=expected["data_manifest_sha256"]:
        raise ValueError("Checkpoint/data manifest binding mismatch")
    if require_checkpoint and prov.get("expected_eval_manifest_sha256")!=expected["eval_manifest_sha256"]:
        raise ValueError("Checkpoint lacks/mismatches explicit expected evaluation hash; export contract first")
    return expected,sha(path)
def verify_prepared(rows,root,data_manifest):
    from src.data import manifest
    root=Path(root);index=root/"index.csv"
    sidecar=root/"PREPARED_BINDING_v1.2.json";complete=root/"COMPLETE.json"
    if sidecar.exists():
        binding=json.loads(sidecar.read_text(encoding="utf-8"));kind="sidecar"
    elif complete.exists():
        binding=json.loads(complete.read_text(encoding="utf-8"));kind="COMPLETE"
    else:raise ValueError("Prepared data lacks COMPLETE or explicit v1.2 binding")
    if binding.get("data_manifest_sha256")!=sha(data_manifest) or binding.get("index_sha256")!=sha(index):
        raise ValueError("Prepared COMPLETE/index/data-manifest binding mismatch")
    with index.open(encoding="utf-8-sig",newline="") as f:entries=list(csv.DictReader(f))
    lookup={(r["split"],r["sample_id"]):r for r in entries}
    full=manifest(data_manifest)
    if len(lookup)!=len(entries) or set(lookup)!={(r["split"],r["sample_id"]) for r in full}:
        raise ValueError("Prepared index must match the exact full supplied manifest")
    file_hashes=binding.get("file_sha256",{})
    for row in rows:
        entry=lookup[row["split"],row["sample_id"]];rel=entry["npz_path"]
        expected_rel=row["split"]+"/"+row["sample_id"]+".npz"
        if rel!=expected_rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError("Unexpected prepared path")
        expected=file_hashes.get(rel) if kind=="sidecar" else entry.get("sha256")
        if not expected or sha(root/rel)!=expected:raise ValueError("Prepared NPZ hash mismatch: "+rel)
    return sha(sidecar if kind=="sidecar" else complete)
