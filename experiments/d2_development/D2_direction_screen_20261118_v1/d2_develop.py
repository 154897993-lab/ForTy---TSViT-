"""Bounded D2 development screening. Validation exploration, never P2 confirmation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,time,hashlib,math,traceback,argparse,collections,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CONFIG=ROOT/'development_protocol.json'
cfg=json.loads(CONFIG.read_text(encoding='utf-8-sig'))
sys.path.insert(0,cfg['parent_source_root'])
import torch,numpy as np
from pipeline import load_protocol,make_data,loader,seed_everything,evaluate_loader,Clouds,boundary_mask,write_json,sha256
from forty_adapter import flatten_s2,encode_labels,IGNORE_INDEX
from model import trunk_hash
from d2_models import build_dev
from torch.nn import functional as F

def verify():
    for rel,want in cfg['parent_source_hashes'].items():
        if sha256(Path(cfg['parent_source_root'])/rel)!=want:raise ValueError('Parent source drift '+rel)
    for rel,want in cfg['development_sources'].items():
        if sha256(ROOT/rel)!=want:raise ValueError('Development source drift '+rel)
    if sha256(cfg['parent_protocol'])!=cfg['parent_protocol_sha256']:raise ValueError('Parent config drift')
    p,_=load_protocol(cfg['parent_protocol']);assert p['official_test_allowed'] is False
    torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    return p

def loss_fn(logits,raw,weight):
    target=encode_labels(raw);valid=target!=IGNORE_INDEX
    if not bool(valid.any()):return None
    scores=logits.movedim(1,-1)[valid];labels=target[valid]
    ce=torch.logsumexp(scores,dim=1)-scores.gather(1,labels[:,None])[:,0]
    if weight==0:return ce.mean()
    bw=boundary_mask(raw,2)[valid].to(ce.dtype);w=1+weight*bw
    return (ce*w).sum()/w.sum()

def save(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def preflight():
    verify();results=[];torch.manual_seed(20261113);x=torch.randn(2,40,128,128,device='cuda');raw=torch.randint(1,9,(2,128,128),device='cuda');raw[:,0:12,0:12]=0
    seed_everything(20261113,True);base=build_dev('A').cuda().eval();expected=base(x);initial=trunk_hash(base)
    seed_everything(20261113,True);d=build_dev('D_original').cuda().eval();assert trunk_hash(d)==initial
    for arch in cfg['candidate_architectures']:
        seed_everything(20261113,True);net=build_dev(arch).cuda().eval();assert trunk_hash(net)==initial
        assert torch.equal(net(x),expected),'Zero residual initial equivalence '+arch
        initial_ref={k:v.clone() for k,v in net.refinement.state_dict().items()};net.train();opt=torch.optim.Adam(net.parameters(),lr=.001,betas=(.9,.9999),eps=1e-8)
        for _ in range(2):
            opt.zero_grad(set_to_none=True);loss=loss_fn(net(x),raw,1);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10,error_if_nonfinite=True);opt.step()
        assert any(not torch.equal(v,initial_ref[k]) for k,v in net.refinement.state_dict().items()),'No branch update'
        count=sum(v.numel() for v in net.refinement.parameters());assert count<=cfg['refinement_parameter_limit']
        results.append({'arch':arch,'initial_forward_bitwise_equal_A':True,'initial_trunk_equal_A':True,'two_steps_finite_and_branch_updated':True,'refinement_parameters':count})
        del net,opt
    sys.path.insert(0,cfg['accepted_evaluator_root'])
    from metrics_v2 import boundary_band
    assert np.array_equal(boundary_mask(raw,2).cpu().numpy(),np.stack([boundary_band(v) for v in raw.cpu().numpy()]))
    scores=torch.randn(2,8,128,128,device='cuda');target=encode_labels(raw);v=target!=IGNORE_INDEX;s=scores.movedim(1,-1)[v];labels=target[v]
    manual=(torch.logsumexp(s,1)-s.gather(1,labels[:,None])[:,0]).mean();assert torch.equal(loss_fn(scores,raw,0),manual)
    assert loss_fn(scores,torch.zeros_like(raw),1) is None
    before=loss_fn(scores,raw,3);altered=scores.clone();altered.permute(0,2,3,1)[~v]=1234;assert torch.equal(loss_fn(altered,raw,3),before)
    write_json(ROOT/'engineering_preflight.json',{'status':'PASS','candidates':results,'boundary_mask_matches_accepted':True,'zero_weight_loss_exact_parent_formula':True,'all_unknown_skips':True,'unknown_logits_excluded':True})


def run_trial(p,train_data,val_data,case,seed,target):
    identity=case['id']+f'_seed{seed}';out=ROOT/'runs'/identity;out.mkdir(parents=True,exist_ok=True)
    fingerprint=sha256(CONFIG);seed_everything(seed,True);net=build_dev(case['architecture']).cuda();init=trunk_hash(net)
    opt=torch.optim.Adam(net.parameters(),lr=.001,betas=(.9,.9999),eps=1e-8,weight_decay=0)
    history=[];best=-math.inf;start=0
    if (out/'last.pt').exists():
        v=torch.load(out/'last.pt',map_location='cuda',weights_only=True);assert v['protocol_sha256']==fingerprint and v['case']==case and v['seed']==seed
        net.load_state_dict(v['state_dict'],strict=True);opt.load_state_dict(v['optimizer']);history=v['history'];start=v['epoch'];best=v['best_score'];assert v['initial_trunk_sha256']==init;del v
    else:
        write_json(out/'config.json',{'case':case,'seed':seed,'parent_protocol':cfg['parent_protocol'],'parent_protocol_sha256':cfg['parent_protocol_sha256'],'development_protocol_sha256':fingerprint,'training':p['training'],'data':p['data'],'development_only':True})
    clouds=Clouds(p,required=False)
    for epoch in range(start+1,target+1):
        seed_everything(seed+epoch,True);net.train();begin=time.perf_counter();loss_sum=valid_total=updates=skipped=0
        order=hashlib.sha256()
        for batch in loader(train_data,p,training=True,seed=seed+epoch):
            order.update(np.asarray(batch['id'],dtype='<i8').tobytes());image=flatten_s2(batch['s2'].cuda());raw=batch['label'].cuda();n=int((raw!=0).sum())
            opt.zero_grad(set_to_none=True);loss=loss_fn(net(image).float(),raw,case['boundary_weight'])
            if loss is None:skipped+=1;continue
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10,error_if_nonfinite=True);opt.step();updates+=1;loss_sum+=loss.item()*n;valid_total+=n
        if updates==0:raise ValueError('No updates')
        validation=evaluate_loader(net,loader(val_data,p),p,clouds,'clean',p['evaluation']['seed'],'cuda');score=validation['whole']['macro_f1']
        if score is None or not math.isfinite(score):raise ValueError('NA selection score')
        improved=score>best;best=max(best,score)
        record={'epoch':epoch,'loss':loss_sum/valid_total,'updates':updates,'skipped':skipped,'train_order_sha256':order.hexdigest(),'seconds':time.perf_counter()-begin,'score':score,'best_score':best,'validation_clean':validation}
        history.append(record);payload={'format':'D2-development-v1','development_only':True,'protocol_sha256':fingerprint,'case':case,'seed':seed,'epoch':epoch,'best_score':best,'initial_trunk_sha256':init,'state_dict':net.state_dict(),'optimizer':opt.state_dict(),'history':history}
        if improved:save(out/'best.pt',payload)
        save(out/'last.pt',payload);write_json(out/'history.json',history)
        write_json(ROOT/'status.json',{'status':'RUNNING','active':identity,'epoch':epoch,'target':target,'updated_unix':time.time()});print(json.dumps({'run':identity,'epoch':epoch,'target':target,'seconds':record['seconds'],'macro_f1':score,'boundary_miou':validation['boundary']['miou']}),flush=True)
    winner=history[max(range(len(history)),key=lambda i:history[i]['score'])]
    summary={'case':case,'seed':seed,'completed_epochs':len(history),'best_epoch':winner['epoch'],'macro_f1':winner['score'],'forest_macro_f1':winner['validation_clean']['whole']['forest_macro_f1'],'miou':winner['validation_clean']['whole']['miou'],'boundary_miou':winner['validation_clean']['boundary']['miou'],'best_sha256':sha256(out/'best.pt'),'last_sha256':sha256(out/'last.pt'),'initial_trunk_sha256':init,'path':str(out)}
    write_json(out/f'summary_{target}.json',summary);del net,opt;torch.cuda.empty_cache();return summary

def launch():
    lock=(ROOT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=verify();assert json.loads((ROOT/'engineering_preflight.json').read_text())['status']=='PASS'
    train_data,val_data=make_data(p,'train'),make_data(p,'validation');assert len(train_data)==3000 and len(val_data)==500
    screen=[]
    for case in cfg['cases']:screen.append(run_trial(p,train_data,val_data,case,20261113,30))
    write_json(ROOT/'screen_30ep.json',screen)
    base=next(x for x in screen if x['case']['id']=='A_w0');perarch={}
    def key(x):
        db=x['boundary_miou']-base['boundary_miou'];forest=x['forest_macro_f1']-base['forest_macro_f1'];miou=x['miou']-base['miou'];return (db>=.01 and forest>=-.01 and miou>=0,db,forest)
    for x in screen:
        a=x['case']['architecture']
        if a not in cfg['candidate_architectures']:continue
        if a not in perarch or key(x)>key(perarch[a]):perarch[a]=x
    promoted=sorted(perarch.values(),key=key,reverse=True)[:2]
    deepids={'A_w0','D_original_w0'}|{x['case']['id'] for x in promoted}|{f"A_w{x['case']['boundary_weight']}" for x in promoted}
    write_json(ROOT/'promotion_decision.json',{'development_selection_not_scientific_gate':True,'rule':cfg['promotion_rule'],'promoted':promoted,'deep_case_ids':sorted(deepids),'screen_reference':base})
    deep=[]
    for seed in cfg['seeds']:
        for case in cfg['cases']:
            if case['id'] in deepids:deep.append(run_trial(p,train_data,val_data,case,seed,100))
    write_json(ROOT/'deep_100ep.json',deep)
    comparisons=[]
    for selected in promoted:
        case=selected['case'];deltas=[]
        for seed in cfg['seeds']:
            x=next(v for v in deep if v['case']['id']==case['id'] and v['seed']==seed);a=next(v for v in deep if v['case']['id']=='A_w0' and v['seed']==seed);lc=next(v for v in deep if v['case']['id']==f"A_w{case['boundary_weight']}" and v['seed']==seed)
            deltas.append({'seed':seed,'boundary_miou_pp':100*(x['boundary_miou']-a['boundary_miou']),'whole_miou_pp':100*(x['miou']-a['miou']),'forest_f1_pp':100*(x['forest_macro_f1']-a['forest_macro_f1']),'architecture_boundary_gain_over_matched_loss_pp':100*(x['boundary_miou']-lc['boundary_miou'])})
        means={k:sum(v[k] for v in deltas)/3 for k in deltas[0] if k!='seed'}
        passed=means['boundary_miou_pp']>=1 and all(v['boundary_miou_pp']>0 for v in deltas) and means['whole_miou_pp']>=0 and means['forest_f1_pp']>=-1
        comparisons.append({'case':case,'paired_deltas':deltas,'means':means,'development_S0_numerical_gate':passed,'three_seed_winner_search_not_confirmatory':True})
    write_json(ROOT/'completion.json',{'status':'COMPLETED_DEVELOPMENT_ONLY','screen_trials':len(screen),'deep_stage_entries':len(deep),'comparisons':comparisons,'cloud_combination_training_started':False,'full_allowed':False,'official_test_allowed':False})
    write_json(ROOT/'status.json',{'status':'COMPLETED_DEVELOPMENT_ONLY','updated_unix':time.time()})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    try:
        if args.preflight:preflight()
        else:launch()
    except BaseException:
        write_json(ROOT/'failure.json',{'status':'STOPPED_TECHNICAL_FAILURE','traceback':traceback.format_exc(),'time':time.time()});raise
