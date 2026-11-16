"""Independently verify public pilot evidence and recompute CM-based metrics."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def metrics(matrix):
    cm=np.asarray(matrix,dtype=np.float64)
    assert cm.shape==(9,9) and np.isfinite(cm).all() and (cm>=0).all()
    cm=cm.copy();cm[0]=0
    tp=cm.diagonal()[1:];support=cm[1:].sum(1);predicted=cm[:,1:].sum(0)
    iou=tp/np.maximum(support+predicted-tp,1)
    f1=2*tp/np.maximum(support+predicted,1)
    return dict(miou=iou.mean(),macro_f1=f1.mean(),forest_f1=f1[:3].mean(),
                planted_f1=f1[1],iou=iou,f1=f1,support=support)


def main():
    root=Path(__file__).resolve().parent
    checks=json.loads((root/"SHA256.json").read_text(encoding="utf-8"))
    for name,expected in checks["files"].items():
        path=(root/name).resolve()
        assert path.is_relative_to(root) and path.is_file(),name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    comparison={(r["run"],r["selection"]):r for r in rows(root/"evaluation/comparison_best_last.csv")}
    handoff={r['run']:r for r in json.loads((root/'handoff_checkpoints.json').read_text(encoding='utf-8'))['checkpoints']}
    manifest=rows(root/'protocols/manifest_train500_validation500_v0.2.csv')
    train_ids={r['sample_id'] for r in manifest if r['split']=='train'}
    val_ids={r['sample_id'] for r in manifest if r['split']=='validation'}
    assert len(train_ids)==len(val_ids)==500 and not train_ids & val_ids and len(manifest)==1000
    scenarios=rows(root/'protocols/eval_manifest_v1.0.csv')
    assert len(scenarios)==2000 and {(r['sample_id'],r['scene']) for r in scenarios}=={(sid,s) for sid in val_ids for s in ('S0','S1','S2','S3')}
    models=0
    for path in sorted((root/"evaluation/per_run").glob("*/*/confusion_matrices.json")):
        folder=path.parent;name=folder.parent.name;selection=folder.name
        provenance=json.loads((folder/'provenance.json').read_text(encoding='utf-8'))
        assert provenance['checkpoint_sha256']==handoff[name][selection+'_checkpoint_sha256']
        for filename,expected in provenance['source_sha256'].items():
            assert hashlib.sha256((root/'source/training_used'/filename).read_bytes()).hexdigest()==expected
        config=root/'protocols/runtime'/f'{name}.json'
        assert hashlib.sha256(config.read_bytes()).hexdigest()==handoff[name]['runtime_protocol_sha256']
        cm=json.loads(path.read_text(encoding="utf-8"))
        for row in rows(folder/"scene_metrics.csv"):
            for region in ("whole","boundary"):
                result=metrics(cm[row["scene"]][region]);prefix="" if region=="whole" else "boundary_"
                for key in ("miou","forest_f1","planted_f1"):
                    assert abs(result[key]-float(row[prefix+key]))<1e-12,(name,selection,region,key)
                if region=="whole":
                    for key in ("miou","macro_f1","forest_f1","planted_f1"):
                        assert abs(result[key]-float(comparison[name,selection][row["scene"]+"_"+key]))<1e-12
        for row in rows(folder/"per_class.csv"):
            result=metrics(cm[row["scene"]][row["region"]]);index=int(row["class_id"])-1
            for key in ("iou","f1","support"):
                assert abs(result[key][index]-float(row[key]))<1e-10
        models+=1
    assert models==26
    print(json.dumps(dict(files_verified=len(checks["files"]),checkpoints=26,
                          scene_class_boundary_metrics_exact=True,official_test_accessed=False)))


if __name__=="__main__":
    main()
