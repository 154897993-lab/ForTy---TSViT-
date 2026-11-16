"""Independently recount all predictions and make best/last pilot tables."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time
import numpy as np

HERE=Path(__file__).parent
BASE=Path('/root/autodl-tmp/paper1_forty_unet')
PACKAGE=BASE/'data/forty_train500_val500_v1_0_20261115'
EVAL=HERE/'unified_evaluation'
INPUT=HERE/'delivery_inputs'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write_csv(p,rs):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def band(y):
    valid=y>0;e=np.zeros_like(valid)
    a=valid[1:]&valid[:-1]&(y[1:]!=y[:-1]);b=valid[:,1:]&valid[:,:-1]&(y[:,1:]!=y[:,:-1])
    e[1:]|=a;e[:-1]|=a;e[:,1:]|=b;e[:,:-1]|=b
    def dilate(x):
        q=np.pad(x,2)
        return np.logical_or.reduce([q[i:i+128,j:j+128] for i in range(5) for j in range(5)])
    return dilate(e)&~dilate(~valid)&valid
def summary(m):
    c=np.asarray(m,dtype=np.float64);t=c.diagonal()[1:];support=c[1:].sum(1);predicted=c[:,1:].sum(0)
    iou=t/np.maximum(support+predicted-t,1);f1=2*t/np.maximum(support+predicted,1)
    return {'miou':float(iou.mean()),'macro_f1':float(f1.mean()),'forest_f1':float(f1[:3].mean()),
            'planted_f1':float(f1[1]),'support':support.tolist(),'iou':iou.tolist(),'f1':f1.tolist()}

def main():
    entries=read(INPUT/'handoff_checkpoints.json')['checkpoints']
    manifest=rows(PACKAGE/'02_固定清单/manifest_train500_validation500_v0.2.csv')
    ids={int(r['sample_id']) for r in manifest if r['split']=='validation'}
    labels={};bands={}
    for sid in ids:
        with np.load(PACKAGE/'prepared/validation'/f'{sid}.npz',allow_pickle=False) as z:labels[sid]=z['labels']
        bands[sid]=band(labels[sid])
    ordered=sorted(ids)
    np.savez_compressed(EVAL/'validation_labels.npz',sample_ids=np.asarray(ordered,dtype=np.int64),
                        labels=np.stack([labels[sid] for sid in ordered]).astype(np.uint8))
    comparisons=[];class_rows=[];scene_rows=[];runtimes=[];verified=0
    for e in entries:
        for selection in ('best','last'):
            root=EVAL/e['run']/selection;runtime=read(root/'runtime.json');runtimes.append(runtime)
            assert sha(Path(e[selection+'_checkpoint_path']))==e[selection+'_checkpoint_sha256']
            provenance=read(root/'provenance.json')
            assert provenance['checkpoint_sha256']==e[selection+'_checkpoint_sha256']
            stored=read(root/'confusion_matrices.json')
            all_rows=rows(root/'prediction_manifest.csv')
            assert len(all_rows)==2000 and {(int(r['sample_id']),r['scene']) for r in all_rows}=={(sid,s) for sid in ids for s in ('S0','S1','S2','S3')}
            totals={s:{region:np.zeros((9,9),np.int64) for region in ('whole','boundary')} for s in ('S0','S1','S2','S3')}
            for r in all_rows:
                path=root/r['prediction'];assert sha(path)==r['sha256']
                sid=int(r['sample_id']);scene=r['scene'];y=labels[sid]
                with np.load(path,allow_pickle=False) as z:
                    p=z['pred_raw'];assert p.shape==(128,128) and p.dtype==np.uint8 and p.min()>=1 and p.max()<=8
                    assert int(z['sample_id'])==sid and str(z['scene'])==scene and str(z['group'])==e['group']
                for region,mask in (('whole',y>0),('boundary',bands[sid])):
                    totals[scene][region]+=np.bincount((y[mask].astype(np.int64)*9+p[mask]).ravel(),minlength=81).reshape(9,9)
                verified+=1
            comparison={'run':e['run'],'group':e['group'],'batch':e['batch'],'seed':e['seed'],
                        'selection':selection,'epoch':e['best_epoch'] if selection=='best' else 100}
            for r in rows(root/'scene_metrics.csv'):
                s=r['scene']
                for region in ('whole','boundary'):
                    assert np.array_equal(totals[s][region],stored[s][region]),(e['run'],selection,s,region)
                    m=summary(totals[s][region]);prefix='' if region=='whole' else 'boundary_'
                    for key in ('miou','forest_f1','planted_f1'):
                        assert abs(m[key]-float(r[prefix+key]))<1e-12
                whole=summary(totals[s]['whole'])
                for key in ('miou','macro_f1','forest_f1','planted_f1'):comparison[s+'_'+key]=whole[key]
                scene_rows.append(dict(run=e['run'],selection=selection,batch=e['batch'],**r))
            for r in rows(root/'per_class.csv'):
                m=summary(totals[r['scene']][r['region']]);c=int(r['class_id'])-1
                for key in ('support','iou','f1'):assert abs(float(r[key])-m[key][c])<1e-10
                class_rows.append(dict(run=e['run'],selection=selection,batch=e['batch'],**r))
            comparisons.append(comparison)
            print(e['run']+'/'+selection+' independently verified',flush=True)
    assert all(r==runtimes[0] for r in runtimes)
    # Re-run a fixed subset after the parallel evaluator processes have exited.
    # This binds archived predictions to the actual checkpoint in the pinned runtime.
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    evaluator=INPUT/'source/fixed_evaluator/01_代码'
    sys.path.insert(0,str(evaluator))
    import torch
    from evaluate import load_model
    from src.protocol import apply_eval
    from src.contracts import model_input
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    selected={ordered[0],ordered[len(ordered)//2]}
    scenes=[r for r in rows(PACKAGE/'02_固定清单/eval_manifest_v1.0.csv')
            if int(r['sample_id']) in selected and r['scene'] in ('S0','S3')]
    images=[]
    for r in scenes:
        sid=int(r['sample_id'])
        with np.load(PACKAGE/'prepared/validation'/f'{sid}.npz',allow_pickle=False) as z:
            x=apply_eval(z['s2'],r,verify=True)
        images.append((sid,r['scene'],torch.from_numpy(np.ascontiguousarray(x[None])).cuda()))
    replayed=0
    for e in entries:
        for selection in ('best','last'):
            net,_=load_model(e[selection+'_checkpoint_path'],'cuda')
            with torch.inference_mode():
                for sid,scene,x in images:
                    pred=net(model_input(x)).argmax(1)[0].cpu().numpy()+1
                    with np.load(EVAL/e['run']/selection/'predictions'/f'{sid}_{scene}.npz',allow_pickle=False) as z:
                        assert np.array_equal(pred,z['pred_raw']),(e['run'],selection,sid,scene)
                    replayed+=1
            del net
    write_csv(EVAL/'comparison_best_last.csv',comparisons)
    write_csv(EVAL/'scene_metrics_all.csv',scene_rows)
    write_csv(EVAL/'per_class_all.csv',class_rows)
    chosen=[r for r in comparisons if r['selection']=='best' and r['batch']==32]
    write_csv(EVAL/'comparison_batch32_best.csv',chosen)
    write(EVAL/'evaluation_runtime_lock.json',dict(frozen=True,**runtimes[0]))
    write(EVAL/'independent_verification.json',{'checkpoints':len(comparisons),'prediction_files_hashed_recounted':verified,
        'whole_boundary_confusions_exact':True,'scene_and_class_metrics_exact':True,'same_pinned_runtime':True,
        'checkpoint_prediction_subset_replay_exact':True,'subset_predictions_replayed':replayed,
        'official_test_accessed':False,'driver_sha256':sha(Path(__file__))})
    print(json.dumps({'verified_checkpoints':len(comparisons),'verified_predictions':verified}),flush=True)
    with tarfile.open(HERE/'evaluation_and_performance.tar.gz','w:gz',compresslevel=1) as archive:
        archive.add(EVAL,arcname='unified_evaluation')
        for name in ('epoch_performance','clean_performance'):
            source=HERE/name
            for p in source.rglob('*'):
                if p.is_file() and p.suffix in ('.json','.log','.jsonl'):
                    archive.add(p,arcname=str(Path('performance_records')/name/p.relative_to(source)))
    print('Archive ready: '+str(HERE/'evaluation_and_performance.tar.gz'),flush=True)

if __name__=='__main__':main()
