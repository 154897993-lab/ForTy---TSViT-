from pathlib import Path
import argparse, collections, hashlib, json, sys, time
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
PARENT = OUT.parent / 'D2_direction_screen_20261118_v1'
EVAL = ROOT / 'contracts/P2_formal_eval_20261118_v2'
sys.path[:0] = [str(EVAL / 'code'), str(PARENT), str(ROOT / 'ruan_delivery_20261117/training_494ebd3')]
from common import read_csv, sha
from contracts_v2 import read_contract, load_prepared
from evaluate_v2 import configure_runtime
from metrics_v2 import boundary_band, confusion, summarize
from scenes_v2 import load_scene_rows, load_masks, apply_scene
from d2_models import build_dev
from pipeline import write_json

def main():
    started = time.perf_counter()
    configure_runtime('cuda', 2)
    contract, contract_sha, manifest = read_contract(EVAL / 'P2_eval_contract_FROZEN_20261118_v2.json')
    protocol_sha = sha(PARENT / 'development_protocol.json')
    cfg = json.loads((PARENT / 'development_protocol.json').read_text(encoding='utf-8-sig'))
    for name, digest in cfg['development_sources'].items():
        assert sha(PARENT / name) == digest
    index = {(int(row['sample_id']), row['split']):row for row in read_csv(contract['data']['index'])}
    val = [row for row in manifest if row['split']=='validation']
    samples = {int(row['sample_id']):load_prepared(row, contract['data']['prepared_root'], index)[:2] for row in val}
    assert len(samples)==500
    scene_path = contract['scenes']['manifest']
    scenes = load_scene_rows(scene_path, set(samples), contract['stage'], ('S0','S1','S2','S3'))
    bands = {key:(boundary_band(y,2), boundary_band(y,0)) for key,(x,y) in samples.items()}
    summaries = json.loads((PARENT / 'deep_100ep.json').read_text())
    expected = {(case,seed) for case in ['A_w0','A_w1','A_w3','D_original_w0','depthwise_zero_w3','spatial_zero_w1'] for seed in cfg['seeds']}
    models = {}
    audited = []
    orders = collections.defaultdict(set)
    for row in summaries:
        run = Path(row['path'])
        assert run.parent.resolve() == (PARENT/'runs').resolve()
        k = (row['case']['id'],row['seed'])
        assert k in expected and k not in models
        for field,name in [('best_sha256','best.pt'),('last_sha256','last.pt')]:
            assert sha(run/name)==row[field], (k,name)
        hist = json.loads((run/'history.json').read_text())
        assert [epoch['epoch'] for epoch in hist] == list(range(1,101))
        assert all(epoch['updates']+epoch['skipped']==94 and np.isfinite(epoch['loss']) for epoch in hist)
        assert all(epoch['score']==epoch['validation_clean']['whole']['macro_f1'] for epoch in hist)
        orders[row['seed']].add(tuple(epoch['train_order_sha256'] for epoch in hist))
        winner = max(hist,key=lambda x:x['score'])
        best = torch.load(run/'best.pt',map_location='cpu',weights_only=True)
        last = torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        assert best['epoch']==winner['epoch']==row['best_epoch'] and last['epoch']==100
        for obj in [best,last]:
            assert obj['protocol_sha256']==protocol_sha and obj['case']==row['case'] and obj['seed']==row['seed']
            assert obj['history']==hist[:obj['epoch']]
            assert all(bool(torch.isfinite(t).all()) for t in obj['state_dict'].values())
        net = build_dev(row['case']['architecture'])
        net.load_state_dict(last['state_dict'],strict=True)
        net.load_state_dict(best['state_dict'],strict=True)
        models[k] = net.cuda().eval()
        snapshots={}
        for target in [30,50,75,100]:
            v=max(hist[:target],key=lambda x:x['score'])
            snapshots[str(target)]={'best_epoch':v['epoch'],'macro_f1':v['score'],'forest_macro_f1':v['validation_clean']['whole']['forest_macro_f1'],'miou':v['validation_clean']['whole']['miou'],'boundary_miou':v['validation_clean']['boundary']['miou']}
        audited.append({**row,'selection_by_budget':snapshots,'initial_trunk_sha256':best['initial_trunk_sha256']})
    assert models.keys()==expected
    assert all(len(x)==1 for x in orders.values())
    assert all(len({x['initial_trunk_sha256'] for x in audited if x['seed']==seed})==1 for seed in cfg['seeds'])
    write_json(OUT/'parent_audit.json',{'status':'PASS','runs':audited,'matching_sample_order':True,'contract_sha256':contract_sha,'protocol_sha256':protocol_sha})
    totals = collections.defaultdict(lambda:np.zeros((9,9),np.int64))
    with torch.inference_mode():
        for i,row in enumerate(scenes,1):
            sid,scene = int(row['sample_id']),row['scene']
            x,y=samples[sid]
            boundary,edge=bands[sid]
            binary,soft=load_masks(row,scene_path)
            scene_x=apply_scene(x,row,soft)
            t=torch.from_numpy(np.ascontiguousarray(scene_x.transpose(0,3,1,2).reshape(1,40,128,128))).cuda()
            cloud=binary.any(axis=0)
            regions={'whole':None,'boundary':boundary,'edge':edge,'interior':~boundary,'cloud':cloud,'noncloud':~cloud}
            for key,net in models.items():
                logits=net(t)
                assert bool(torch.isfinite(logits).all())
                pred=(logits.argmax(1)[0].cpu().numpy()+1).astype(np.uint8)
                for region,mask in regions.items():
                    totals[(*key,scene,region)] += confusion(pred,y,mask)
            if i%250==0:
                print(json.dumps({'scene_rows':i,'total':len(scenes),'seconds':round(time.perf_counter()-started,1)}),flush=True)
    result=[]
    for (case,seed,scene,region),cm in sorted(totals.items()):
        if region=='whole':
            assert np.array_equal(cm,totals[case,seed,scene,'cloud']+totals[case,seed,scene,'noncloud'])
            assert np.array_equal(cm,totals[case,seed,scene,'boundary']+totals[case,seed,scene,'interior'])
        result.append({'case':case,'seed':seed,'scene':scene,'region':region,'metrics':summarize(cm),'confusion':cm.tolist()})
    write_json(OUT/'frozen_s0_s3.json',{'status':'COMPLETED_DEVELOPMENT_DIAGNOSTIC','scope':'Same frozen S0-S3 evaluation assets applied to clean-trained D2 models; this is not B-trained increment evidence. Edge uses valid four-neighbor transitions before dilation. Interior is the valid complement of the official boundary band and can include pixels near unknown.','evaluation_batch':1,'validation_count':len(samples),'checkpoints':len(models),'contract_sha256':contract_sha,'scene_manifest_sha256':sha(scene_path),'official_test_accessed':False,'seconds':time.perf_counter()-started,'results':result})

if __name__=='__main__':
    main()

