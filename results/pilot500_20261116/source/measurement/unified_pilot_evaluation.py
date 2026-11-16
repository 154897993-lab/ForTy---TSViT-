"""Call the unchanged fixed evaluator for all pilot best/last checkpoints."""
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

BASE=Path('/root/autodl-tmp/paper1_forty_unet')
HERE=Path(__file__).parent
PACKAGE=BASE/'data/forty_train500_val500_v1_0_20261115'
EVALUATOR=HERE/'delivery_inputs/source/fixed_evaluator/01_代码'
RESULT=HERE/'unified_evaluation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def evaluate_job(entry):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    sys.path.insert(0,str(EVALUATOR))
    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    from evaluate import evaluate
    output=RESULT/entry['run']/entry['selection']
    begin=time.perf_counter()
    value=evaluate(entry['checkpoint_path'],
                   PACKAGE/'02_固定清单/manifest_train500_validation500_v0.2.csv',
                   PACKAGE/'02_固定清单/eval_manifest_v1.0.csv',output,
                   prepared_root=PACKAGE/'prepared',device='cuda',threads=2)
    assert value['predictions']==2000 and value['checkpoint_sha256']==entry['checkpoint_sha256']
    runtime={'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),
             'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'threads':2,
             'deterministic_algorithms':True,'cudnn_benchmark':False,'cudnn_allow_tf32':False,
             'matmul_allow_tf32':False,'inference_batch':1,
             'evaluator_source_sha256':{p.relative_to(EVALUATOR).as_posix():sha(p)
                                      for p in EVALUATOR.rglob('*.py')},
             'driver_sha256':sha(Path(__file__))}
    (output/'runtime.json').write_text(json.dumps(runtime,indent=2))
    return {'run':entry['run'],'selection':entry['selection'],'seconds':time.perf_counter()-begin,'predictions':2000,
            'checkpoint_sha256':value['checkpoint_sha256']}

if __name__=='__main__':
    original=json.loads((HERE/'delivery_inputs/handoff_checkpoints.json').read_text())['checkpoints']
    entries=[dict(e,selection=selection,checkpoint_path=e[selection+'_checkpoint_path'],
                  checkpoint_sha256=e[selection+'_checkpoint_sha256']) for e in original for selection in ('best','last')]
    RESULT.mkdir(exist_ok=False)
    begin=time.perf_counter();results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(evaluate_job,e) for e in entries]
        for future in concurrent.futures.as_completed(futures):
            value=future.result();results.append(value);print(json.dumps(value),flush=True)
    summary={'runs':results,'total_predictions':sum(x['predictions'] for x in results),
             'total_wall_seconds':time.perf_counter()-begin,'all_same_inference_runtime':True,
             'official_test_accessed':False,'selection':'existing macro-F1 best and epoch100 last checkpoints; no reselection'}
    (RESULT/'completion.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'completed_runs':len(results),'predictions':summary['total_predictions']}),flush=True)
