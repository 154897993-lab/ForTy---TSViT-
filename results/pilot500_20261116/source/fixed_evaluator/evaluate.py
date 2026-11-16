"""Unified U-Net validation: four scenes, raw predictions, whole/boundary CMs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import argparse,csv,json,time
import numpy as np
import torch
from src.contracts import GROUPS,model_input,sha
from src.data import manifest,load_all
from src.metrics import boundary_band,confusion,summarize,NAMES
from src.protocol import apply_eval,masks_for,mask_hash
from vendor.forest_unet import ForestUNet,MODEL_VERSION
from safety_v1_2 import finite_state,checked_logits,contract_for,verify_prepared

def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def load_model(checkpoint,device):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    for key in ['model_config','state_dict','provenance','model_version']:
        if key not in saved:raise ValueError('Checkpoint lacks '+key+'; request the full delivery contract')
    if saved['model_version']!=MODEL_VERSION:raise ValueError('Model version mismatch')
    prov=saved['provenance'];group=prov['group']
    if group not in GROUPS or bool(saved['model_config']['use_edge'])!=GROUPS[group][1]:
        raise ValueError('Group and edge configuration disagree')
    if saved['model_config']!={'in_channels':40,'num_classes':8,'base_channels':16,'use_edge':GROUPS[group][1]}:
        raise ValueError('Checkpoint is not the agreed formal U-Net architecture')
    for field in ['train_manifest_sha256','protocol_sha256','train_seed','scope','training_cloud_source_sha256',
                  'model_source_sha256','engine_source_sha256']:
        if field not in prov:raise ValueError('Missing checkpoint provenance: '+field)
    here=Path(__file__).resolve().parent
    if prov['model_source_sha256']!=sha(here/'vendor/forest_unet.py') or prov['engine_source_sha256']!=sha(here/'vendor/cloud_haze.py'):
        raise ValueError('Model or engine source differs from checkpoint contract')
    if prov['training_cloud_source_sha256']!=sha(here/'src/training_clouds.py'):
        raise ValueError('Training cloud wrapper differs from checkpoint contract')
    if prov['protocol_sha256']!=sha(here.parent/'00_协议与规划/protocol_candidate_pilot500_v0.2.json'):
        raise ValueError('Checkpoint protocol differs from this delivery version')
    finite_state(saved['state_dict'])
    net=ForestUNet(**saved['model_config']).to(device)
    net.load_state_dict(saved['state_dict'],strict=True)
    finite_state(net.state_dict());net.eval()
    return net,saved

def evaluate(checkpoint,data_manifest,eval_manifest,out,prepared_root=None,raw_root=None,
             device='cpu',threads=2,input_out=None,boundary_radius=2):
    start=time.perf_counter();torch.set_num_threads(threads)
    net,saved=load_model(checkpoint,device);prov=saved['provenance']
    if prov['train_manifest_sha256']!=sha(data_manifest):raise ValueError('Training/data manifest hash mismatch')
    expected_contract,contract_sha=contract_for(prov,data_manifest,eval_manifest)
    with open(eval_manifest,encoding='utf-8-sig',newline='') as f:scenes=list(csv.DictReader(f))
    rows=[r for r in manifest(data_manifest) if r['split']=='validation'];ids={int(r['sample_id']) for r in rows}
    if len(scenes)!=4*len(ids) or {(int(r['sample_id']),r['scene']) for r in scenes}!={(sid,s) for sid in ids for s in ['S0','S1','S2','S3']}:
        raise ValueError('Evaluation must contain exactly every validation ID in all four scenes')
    if any(r['split']!='validation' for r in scenes):raise ValueError('Official test access prohibited')
    prepared_sha=verify_prepared(rows,prepared_root,data_manifest) if prepared_root else None
    data=load_all(rows,prepared_root,raw_root);out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new evaluation output directory')
    (out/'predictions').mkdir(parents=True,exist_ok=True)
    if input_out:Path(input_out).mkdir(parents=True,exist_ok=True)
    totals={s:{k:np.zeros((9,9),np.int64) for k in ['whole','boundary']} for s in ['S0','S1','S2','S3']}
    errors=[];exported=[]
    with torch.inference_mode():
        for row in scenes:
            sid=int(row['sample_id']);scene=row['scene'];original,target,valid=data[sid]
            augmented=apply_eval(original,row,verify=True)
            x=torch.from_numpy(np.ascontiguousarray(augmented[None])).to(device)
            prediction=(checked_logits(net(model_input(x))).argmax(1)[0].cpu().numpy()+1).astype(np.uint8)
            band=boundary_band(target,boundary_radius)
            totals[scene]['whole']+=confusion(prediction,target)
            totals[scene]['boundary']+=confusion(prediction,target,band)
            name=str(sid)+'_'+scene+'.npz';pred_path=out/'predictions'/name
            np.savez_compressed(pred_path,pred_raw=prediction,sample_id=np.int64(sid),scene=scene,
                                group=prov['group'],train_seed=np.int64(prov['train_seed']))
            if input_out:
                seasons=[int(v) for v in row['seasons'].split(';') if v]
                masks,_,_ = masks_for(int(row['seed']),seasons,float(row['coverage_target']),row['level'],scene=='S3',int(row['overlap_shift'])) if seasons else ({},None,0)
                if mask_hash(masks)!=row['mask_sha256']:raise ValueError('Exported mask mismatch')
                binary=np.zeros((4,128,128),np.uint8);soft=np.zeros((4,128,128),np.float32)
                for season,mask in masks.items():binary[season]=mask['binary'];soft[season]=mask['soft']
                shared=Path(input_out)/name
                if not shared.exists():
                    np.savez_compressed(shared,original_s2=original,eval_s2=augmented,label_raw=target.astype(np.uint8),
                        valid_mask=valid,cloud_binary=binary,cloud_soft=soft,boundary_band=band.astype(np.uint8),
                        sample_id=np.int64(sid),scene=scene,eval_row_json=json.dumps(row,sort_keys=True))
                else:
                    with np.load(shared,allow_pickle=False) as z:
                        if not (np.array_equal(z['eval_s2'],augmented) and np.array_equal(z['label_raw'],target)
                                and np.array_equal(z['cloud_binary'],binary)):
                            raise ValueError('Shared scene input differs across groups')
            effective=target!=0;wrong=(prediction!=target)&effective
            errors.append(dict(sample_id=sid,scene=scene,valid_pixels=int(effective.sum()),
                wrong_pixels=int(wrong.sum()),error_rate=float(wrong.sum()/max(int(effective.sum()),1)),
                forest_wrong_pixels=int((wrong&(target<=3)).sum()),coverage_target=row['coverage_target'],
                mask_sha256=row['mask_sha256']))
            exported.append(dict(sample_id=sid,scene=scene,prediction='predictions/'+name,sha256=sha(pred_path)))
    metrics=[];per_class=[];cms={}
    for scene in totals:
        cm=totals[scene];whole=summarize(cm['whole']);boundary=summarize(cm['boundary'])
        metrics.append(dict(group=prov['group'],train_seed=prov['train_seed'],scene=scene,scope=prov['scope'],
            n_samples=len(ids),forest_f1=whole['forest_f1'],planted_f1=whole['planted_f1'],miou=whole['miou'],
            macro_f1=whole['macro_f1'],boundary_forest_f1=boundary['forest_f1'],
            boundary_planted_f1=boundary['planted_f1'],boundary_miou=boundary['miou'],
            valid_pixels=whole['valid_pixels'],boundary_pixels=boundary['valid_pixels']))
        for region,summary in [('whole',whole),('boundary',boundary)]:
            for c,name in enumerate(NAMES):
                per_class.append(dict(scene=scene,region=region,class_id=c+1,class_name=name,
                    iou=summary['iou'][c],f1=summary['f1'][c],support=summary['support'][c]))
        cms[scene]={region:value.tolist() for region,value in cm.items()}
    write_csv(out/'scene_metrics.csv',metrics);write_csv(out/'per_class.csv',per_class)
    write_csv(out/'sample_errors.csv',sorted(errors,key=lambda r:-r['error_rate']))
    write_csv(out/'prediction_manifest.csv',exported)
    (out/'confusion_matrices.json').write_text(json.dumps(cms),encoding='utf-8')
    result=dict(prov)
    result.update(checkpoint_sha256=sha(checkpoint),data_manifest_sha256=sha(data_manifest),
        eval_manifest_sha256=sha(eval_manifest),boundary_radius=boundary_radius,
        boundary_rule='GT 4-neighbor transitions, Chebyshev dilation radius 2, exclude nodata radius 2',
        units='ratio_0_1',absent_classes='zero_if_denominator_zero; empty_region=null',
        official_test_accessed=False,predictions=len(exported),seconds=time.perf_counter()-start)
    result.update(evaluation_contract_sha256=contract_sha,prepared_binding_sha256=prepared_sha,
                  safety_code_version='first-batch-safety-v1.2')
    (out/'provenance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--eval-manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--prepared-root');p.add_argument('--raw-root');p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=2);p.add_argument('--input-out')
    a=p.parse_args();print(json.dumps(evaluate(a.checkpoint,a.manifest,a.eval_manifest,a.out,a.prepared_root,
        a.raw_root,a.device,a.threads,a.input_out),ensure_ascii=False))
if __name__=='__main__':main()
