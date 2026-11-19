"""Replay factorial weights and measure available corrections without label-derived gates."""
from pathlib import Path
import collections
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
FIRST = OUT.parent / 'D2_direction_screen_20261118_v1'
SECOND = OUT.parent / 'D2_followup_20261119_v1'
EVAL = ROOT / 'contracts/P2_formal_eval_20261118_v2'
sys.path[:0] = [str(EVAL/'code'), str(FIRST), str(ROOT/'ruan_delivery_20261117/training_494ebd3')]
from common import read_csv, sha
from contracts_v2 import read_contract, load_prepared
from d2_models import build_dev
from evaluate_v2 import configure_runtime
from metrics_v2 import boundary_band, confusion, summarize
from pipeline import write_json
from scenes_v2 import load_scene_rows, load_masks, apply_scene

SEEDS = (20261113, 20261114, 20261115)
SCENES = ('S0','S1','S2','S3')
CAP = 0.75
RADIUS = 2

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def predicted_edge_gate(scores, radius=RADIUS):
    classes = scores.argmax(1)
    edge = torch.zeros_like(classes, dtype=torch.bool)
    horizontal = classes[:, :, 1:] != classes[:, :, :-1]
    vertical = classes[:, 1:, :] != classes[:, :-1, :]
    edge[:, :, 1:] |= horizontal
    edge[:, :, :-1] |= horizontal
    edge[:, 1:, :] |= vertical
    edge[:, :-1, :] |= vertical
    return F.max_pool2d(edge[:, None].float(), 2*radius+1, stride=1, padding=radius).bool()

def audited_models():
    protocol = read(FIRST/'development_protocol.json')
    assert all(sha(FIRST/k)==v for k,v in protocol['development_sources'].items())
    assert all(sha(Path(protocol['parent_source_root'])/k)==v for k,v in protocol['parent_source_hashes'].items())
    jobs = read(SECOND/'factorial_completion.json')['jobs']
    expected = {(case,seed) for case in ('depthwise_zero_w0','depthwise_zero_w1') for seed in SEEDS}
    assert len(jobs)==6 and {(j['case']['id'],j['seed']) for j in jobs}==expected
    first_jobs = {(j['case']['id'],j['seed']):j for j in read(FIRST/'deep_100ep.json')}
    models = {}
    for j in jobs+[first_jobs['A_w0',s] for s in SEEDS]:
        case,seed = j['case'],j['seed']
        root = Path(j['path'])
        assert root.parent in (FIRST/'runs', SECOND/'runs')
        history = read(root/'history.json')
        assert [e['epoch'] for e in history]==list(range(1,101))
        assert all(e['updates']+e['skipped']==94 and np.isfinite(e['loss']) for e in history)
        matched = first_jobs[f"A_w{case['boundary_weight']}",seed]
        control_history = read(Path(matched['path'])/'history.json')
        assert [e['train_order_sha256'] for e in history]==[e['train_order_sha256'] for e in control_history]
        assert j['initial_trunk_sha256']==matched['initial_trunk_sha256']
        net = build_dev(case['architecture'])
        for name in ('last','best'):
            path=root/f'{name}.pt'
            assert sha(path)==j[f'{name}_sha256']
            saved=torch.load(path,map_location='cpu',weights_only=True)
            epoch = 100 if name=='last' else max(history,key=lambda e:e['score'])['epoch']
            assert saved['epoch']==epoch and saved['history']==history[:epoch]
            assert saved['initial_trunk_sha256']==j['initial_trunk_sha256']
            assert saved['seed']==seed and saved['case']==case and saved['protocol_sha256']==sha(FIRST/'development_protocol.json')
            assert all(torch.isfinite(v).all() for v in saved['state_dict'].values())
            net.load_state_dict(saved['state_dict'],strict=True)
        models[case['id'],seed]=net.cuda().eval()
    return models

def main():
    start=time.perf_counter()
    configure_runtime('cuda',2)
    models=audited_models()
    contract,digest,manifest=read_contract(EVAL/'P2_eval_contract_FROZEN_20261118_v2.json')
    index={(int(row['sample_id']),row['split']):row for row in read_csv(contract['data']['index'])}
    val=[row for row in manifest if row['split']=='validation']
    samples={int(row['sample_id']):load_prepared(row,contract['data']['prepared_root'],index)[:2] for row in val}
    assert len(samples)==500
    bands={sid:boundary_band(y,2) for sid,(x,y) in samples.items()}
    rows=load_scene_rows(contract['scenes']['manifest'],set(samples),contract['stage'],SCENES)
    total=collections.defaultdict(lambda:np.zeros((9,9),np.int64))
    reach=collections.defaultdict(collections.Counter)
    with torch.inference_mode():
        for i,row in enumerate(rows,1):
            sid,scene=int(row['sample_id']),row['scene']
            x,y=samples[sid]
            boundary=bands[sid]
            binary,soft=load_masks(row,contract['scenes']['manifest'])
            x=apply_scene(x,row,soft)
            tensor=torch.from_numpy(np.ascontiguousarray(x.transpose(0,3,1,2).reshape(1,40,128,128))).cuda()
            for (case,seed),net in models.items():
                logits=net(tensor)
                assert torch.isfinite(logits).all()
                pred=(logits.argmax(1)[0].cpu().numpy()+1).astype(np.uint8)
                for region,mask in [('whole',None),('boundary',boundary),('interior',~boundary)]:
                    total[case,seed,scene,region]+=confusion(pred,y,mask)
                if case=='A_w0':
                    gate=predicted_edge_gate(logits)[0,0].cpu().numpy()
                    scores=logits[0].cpu().numpy()
                    oracle_margin=scores.max(0)-np.take_along_axis(scores,np.maximum(y.astype(np.int64)-1,0)[None],axis=0)[0]
                    valid=y!=0
                    wrong=valid & (pred!=y)
                    stat=reach[seed,scene]
                    stat['valid']+=int(valid.sum())
                    stat['gate_valid']+=int((gate & valid).sum())
                    stat['boundary']+=int(boundary.sum())
                    stat['gate_boundary']+=int((gate & boundary).sum())
                    stat['boundary_errors']+=int((wrong & boundary).sum())
                    stat['gated_boundary_errors']+=int((wrong & boundary & gate).sum())
                    stat['reachable_boundary_errors']+=int((wrong & boundary & gate & (oracle_margin<=2*CAP)).sum())
                    stat['planted_boundary_errors']+=int((wrong & boundary & (y==2)).sum())
                    stat['reachable_planted_errors']+=int((wrong & boundary & (y==2) & gate & (oracle_margin<=2*CAP)).sum())
            if i%500==0:
                print(f'completed {i}/{len(rows)} scene rows',flush=True)
    payload={'development_only':True,'official_test_accessed':False,'contract_sha256':digest,
        'evaluated_checkpoints':len(models),'audited_new_checkpoint_files':12,
        'seconds':time.perf_counter()-start,
        'results':[{'case':c,'seed':s,'scene':q,'region':r,'metrics':summarize(cm),'confusion':cm.tolist()} for (c,s,q,r),cm in sorted(total.items())],
        'reach':[{'seed':s,'scene':q,**x} for (s,q),x in sorted(reach.items())],
        'gate_radius':RADIUS,'logit_cap':CAP}
    write_json(OUT/'factorial_scene_diagnostic.json',payload)

if __name__=='__main__':
    main()

