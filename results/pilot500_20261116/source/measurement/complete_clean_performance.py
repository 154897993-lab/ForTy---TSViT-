"""Measure clean A/D complete epochs using the actual training entry."""
import contextlib
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

HERE=Path(__file__).parent
BASE=Path('/root/autodl-tmp/paper1_forty_unet')
sys.path.insert(0,str(HERE/'runtime'))
from benchmark_epoch import EpochLog
from pipeline import sha256,write_json,source_hashes
from train import train
import numpy as np
import torch

def main():
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    root=HERE/'clean_performance';root.mkdir(exist_ok=False)
    base=json.loads((BASE/'results/sweep500_cloud_20261115/configs/A_b32_e100.json').read_text())
    records=[]
    for group in ('A','D'):
        cfg=copy.deepcopy(base);cfg['version']='clean-performance-v1.0-'+group
        cfg['training'].update(epochs=3,measure_train=False)
        cfg['clouds']['sha256']=sha256(HERE/'runtime/ruan_cloud_provider.py')
        cfg['delivery']['scope']='PERFORMANCE_ACCEPTANCE_NOT_PAPER_RESULTS'
        config=root/(group+'.json');write_json(config,cfg)
        start=time.perf_counter()
        with (root/(group+'.log')).open('w') as f:
            log=EpochLog(f,group)
            with contextlib.redirect_stdout(log):train(config,group,20261113,root/group,device='cuda')
        intervals=[b['finished']-a['finished'] for a,b in zip(log.events,log.events[1:])]
        records.append({'group':group,'cpu_workers':0,'epochs':3,'train_samples':500,'validation_samples':500,
                        'run_wall_seconds_including_setup':time.perf_counter()-start,
                        'steady_complete_epoch_seconds':intervals,
                        'mean_steady_complete_epoch_seconds':sum(intervals)/len(intervals),
                        'config_sha256':sha256(config)})
    result={'measurements':records,'source_sha256':source_hashes(),'official_test_accessed':False,
            'driver_sha256':sha256(Path(__file__))}
    write_json(root/'clean_epoch_benchmark.json',result)
    write_json(HERE/'environment_lock.json',{'python':sys.version,'platform':platform.platform(),
        'torch':str(torch.__version__),'numpy':np.__version__,'cuda':torch.version.cuda,
        'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0),
        'dependencies':sorted([{'name':d.metadata['Name'],'version':d.version}
                               for d in importlib.metadata.distributions()],key=lambda x:x['name'].lower())})
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
