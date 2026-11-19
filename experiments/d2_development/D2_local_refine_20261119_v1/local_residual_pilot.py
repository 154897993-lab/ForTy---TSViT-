"""Compare global and local residual heads on identical frozen validation-selected A trunks."""
import collections
import datetime
import fcntl
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from factorial_scene_diagnostic import (
    CAP, RADIUS, OUT, FIRST, EVAL, SCENES, SEEDS, read, predicted_edge_gate,
    configure_runtime, read_contract, read_csv, load_prepared, boundary_band,
    load_scene_rows, load_masks, apply_scene, confusion, summarize, sha, write_json,
)
from d2_models import build_dev, Refine
from forty_adapter import flatten_s2, encode_labels, IGNORE_INDEX
from pipeline import boundary_mask, load_protocol, make_data, loader, seed_everything

MODES = ('global','local')
EPOCHS = 15

def get_features(net, x):
    features = {}
    def hook(name):
        def capture(module, inputs, outputs):
            features[name]=outputs
        return capture
    handlers = [net.enc1.register_forward_hook(hook('shallow')),net.up1.register_forward_hook(hook('full'))]
    try:
        scores=net(x)
        return torch.cat((features['full'],features['shallow']),1),scores
    finally:
        for h in handlers:
            h.remove()

def apply_head(head, features, baseline, gate=None):
    adjust=CAP*torch.tanh(head.head(head.body(features)))
    if gate is not None:
        adjust=adjust*gate
    return baseline+adjust

def cache_train(net, data, cfg):
    image_count=len(data)
    x = torch.empty((image_count,32,128,128),device='cuda',dtype=torch.float32)
    z = torch.empty((image_count,8,128,128),device='cuda',dtype=torch.float32)
    target = torch.empty((image_count,128,128),device='cuda',dtype=torch.int64)
    weight = torch.empty((image_count,128,128),device='cuda',dtype=torch.float32)
    gate = torch.empty((image_count,1,128,128),device='cuda',dtype=torch.bool)
    pos=0
    with torch.no_grad():
        for b in loader(data,cfg,training=False):
            raw=b['label'].cuda()
            features,scores=get_features(net,flatten_s2(b['s2'].cuda()))
            ix=slice(pos,pos+len(raw))
            x[ix]=features
            z[ix]=scores
            target[ix]=encode_labels(raw)
            weight[ix]=(raw!=0).float()*(1+boundary_mask(raw,RADIUS).float())
            gate[ix]=predicted_edge_gate(scores)
            pos+=len(raw)
    assert pos==3000
    return x,z,target,weight,gate

def weighted_ce_terms(pred, target, weight):
    valid=target!=IGNORE_INDEX
    scores=pred.movedim(1,-1)[valid]
    labels=target[valid]
    valid_weight=weight[valid]
    ce=torch.logsumexp(scores,dim=1)-scores.gather(1,labels[:,None])[:,0]
    return (ce*valid_weight).sum(),valid_weight.sum()

def run_pair(seed,net,data,cfg,base_hash,protocol_hash):
    start=time.perf_counter()
    x,z,target,weight,gate=cache_train(net,data,cfg)
    print(json.dumps({'seed':seed,'cache_seconds':time.perf_counter()-start}),flush=True)
    heads={}
    for mode in MODES:
        write_json(OUT/'pilot_status.json',{'phase':'training','seed':seed,'mode':mode,'epoch':0,'target':EPOCHS})
        seed_everything(seed,True)
        head=Refine('depthwise_zero').cuda()
        frozen=z[:4]
        with torch.no_grad():
            assert torch.equal(frozen,apply_head(head.eval(),x[:4],frozen,gate[:4] if mode=='local' else None))
        opt=torch.optim.Adam(head.parameters(),lr=.001)
        head.train()
        hist=[]
        for ep in range(1,EPOCHS+1):
            t0=time.perf_counter()
            permutation=torch.randperm(len(x),generator=torch.Generator().manual_seed(seed+ep))
            loss_num=valid_weight=0.
            steps=0
            for ids in permutation.split(32):
                ids=ids.cuda()
                im,base,y,w = x[ids],z[ids],target[ids],weight[ids]
                enabled=gate[ids] if mode=='local' else None
                opt.zero_grad(set_to_none=True)
                pred=apply_head(head,im,base,enabled)
                numerator,den=weighted_ce_terms(pred,y,w)
                if not bool(den>0):
                    continue
                loss=numerator/den
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),10,error_if_nonfinite=True)
                opt.step()
                loss_num+=float(numerator.detach())
                valid_weight+=float(den)
                steps+=1
            assert steps==94
            hist.append({'epoch':ep,'weighted_ce':loss_num/valid_weight,'updates':steps,'seconds':time.perf_counter()-t0})
            write_json(OUT/'pilot_status.json',{'phase':'training','seed':seed,'mode':mode,'epoch':ep,'target':EPOCHS,'epoch_seconds':hist[-1]['seconds']})
        path=OUT/f'{mode}_seed{seed}.pt'
        tmp=path.with_suffix('.tmp')
        torch.save({'mode':mode,'seed':seed,'base_sha256':base_hash,'plan_sha256':protocol_hash,'epochs':EPOCHS,'state_dict':head.state_dict(),'history':hist},tmp)
        tmp.rename(path)
        heads[mode]=head.eval()
        print(json.dumps({'seed':seed,'mode':mode,'seconds':sum(e['seconds'] for e in hist),'head_sha256':sha(path)}),flush=True)
    return heads

def evaluate(models, contract, manifest):
    val=[row for row in manifest if row['split']=='validation']
    index={(int(row['sample_id']),row['split']):row for row in read_csv(contract['data']['index'])}
    samples={int(row['sample_id']):load_prepared(row,contract['data']['prepared_root'],index)[:2] for row in val}
    assert len(samples)==500
    rows=load_scene_rows(contract['scenes']['manifest'],set(samples),contract['stage'],SCENES)
    bands={sid:boundary_band(y,RADIUS) for sid,(x,y) in samples.items()}
    totals=collections.defaultdict(lambda:np.zeros((9,9),np.int64))
    transitions=collections.defaultdict(collections.Counter)
    with torch.inference_mode():
        for count,row in enumerate(rows,1):
            sid,scene=int(row['sample_id']),row['scene']
            x,y=samples[sid]
            band=bands[sid]
            valid=y!=0
            _,soft=load_masks(row,contract['scenes']['manifest'])
            x=apply_scene(x,row,soft)
            x=torch.from_numpy(np.ascontiguousarray(x.transpose(0,3,1,2).reshape(1,40,128,128))).cuda()
            for seed,(net,heads) in models.items():
                feat,baseline=get_features(net,x)
                gate=predicted_edge_gate(baseline)
                old=(baseline.argmax(1)[0].cpu().numpy()+1).astype(np.uint8)
                preds={'A_frozen':old}
                for mode,head in heads.items():
                    logits=apply_head(head,feat,baseline,gate if mode=='local' else None)
                    assert torch.isfinite(logits).all()
                    if mode=='local':
                        assert torch.equal(logits.masked_select(~gate),baseline.masked_select(~gate))
                    preds[mode]=(logits.argmax(1)[0].cpu().numpy()+1).astype(np.uint8)
                for mode,pred in preds.items():
                    for region,mask in [('whole',valid),('boundary',band),('interior',valid & ~band)]:
                        totals[mode,seed,scene,region]+=confusion(pred,y,mask)
                        if mode!='A_frozen':
                            c=transitions[mode,seed,scene,region]
                            c['fixed']+=int(((old!=y)&(pred==y)&mask).sum())
                            c['broken']+=int(((old==y)&(pred!=y)&mask).sum())
                            c['changed']+=int(((old!=pred)&mask).sum())
            if count%250==0:
                write_json(OUT/'pilot_status.json',{'phase':'validation','scene_rows':count,'target':len(rows)})
    return [{'mode':m,'seed':s,'scene':q,'region':r,'metrics':summarize(cm),'confusion':cm.tolist(),'transitions':dict(transitions[m,s,q,r])} for (m,s,q,r),cm in sorted(totals.items())]

def main():
    with (OUT/'pilot.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path=OUT/'pilot_plan.json'
        plan=read(plan_path)
        assert plan['runner_sha256']==sha(Path(__file__)) and plan['diagnostic_source_sha256']==sha(OUT/'factorial_scene_diagnostic.py')
        assert plan['epochs']==EPOCHS and plan['seeds']==list(SEEDS) and plan['modes']==list(MODES)
        assert plan['radius']==RADIUS and plan['residual_logit_cap']==CAP and plan['official_test_allowed'] is False
        assert sha(OUT/'factorial_scene_diagnostic.json')==plan['diagnostic_sha256']
        start=time.perf_counter()
        configure_runtime('cuda',4)
        contract,contract_hash,manifest=read_contract(EVAL/'P2_eval_contract_FROZEN_20261118_v2.json')
        assert contract_hash==plan['evaluation_contract_sha256']
        proto=read(FIRST/'development_protocol.json')
        assert sha(FIRST/'development_protocol.json')==plan['parent_development_protocol_sha256']
        assert sha(Path(proto['parent_protocol']))==proto['parent_protocol_sha256']
        assert all(sha(FIRST/k)==v for k,v in proto['development_sources'].items())
        assert all(sha(Path(proto['parent_source_root'])/k)==v for k,v in proto['parent_source_hashes'].items())
        cfg,_=load_protocol(proto['parent_protocol'])
        data=make_data(cfg,'train')
        summary={(j['case']['id'],j['seed']):j for j in read(FIRST/'deep_100ep.json')}
        models={}
        for seed in SEEDS:
            item=summary['A_w0',seed]
            checkpoint=Path(item['path'])/'best.pt'
            assert sha(checkpoint)==item['best_sha256']==plan['base_checkpoints'][str(seed)]
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert saved['seed']==seed and saved['case']['id']=='A_w0'
            assert saved['protocol_sha256']==plan['parent_development_protocol_sha256']
            net=build_dev('A').cuda().eval().requires_grad_(False)
            net.load_state_dict(saved['state_dict'],strict=True)
            write_json(OUT/'pilot_status.json',{'phase':'cache_train_features','seed':seed})
            heads=run_pair(seed,net,data,cfg,item['best_sha256'],sha(plan_path))
            models[seed]=net,heads
        rows=evaluate(models,contract,manifest)
        write_json(OUT/'pilot_completion.json',{'status':'COMPLETED_DEVELOPMENT_ONLY','official_test_accessed':False,'scope':'fixed final epoch using clean training; nonconfirmatory validation development','contract_sha256':contract_hash,'plan_sha256':sha(plan_path),'seconds':time.perf_counter()-start,'results':rows})
        write_json(OUT/'pilot_status.json',{'phase':'completed'})

if __name__=='__main__':
    try:
        main()
    except BaseException:
        write_json(OUT/'pilot_failure.json',{'traceback':traceback.format_exc()})
        raise

