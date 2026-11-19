"""Fill the two missing depthwise loss levels while preserving v1 runs."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'D2_direction_screen_20261118_v1'
sys.path.insert(0, str(PARENT))
import d2_develop as d2

PLAN = ROOT / 'factorial_plan.json'
CASES = ('depthwise_zero_w0', 'depthwise_zero_w1')
SEEDS = (20261113, 20261114, 20261115)
METRICS = ('boundary_miou', 'miou', 'forest_macro_f1', 'macro_f1')

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    with (ROOT / 'factorial.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(PLAN)
        assert plan['runner_sha256'] == d2.sha256(Path(__file__))
        assert plan['parent_protocol_sha256'] == d2.sha256(d2.CONFIG)
        assert plan['parent_deep_sha256'] == d2.sha256(PARENT / 'deep_100ep.json')
        assert plan['parent_screen_sha256'] == d2.sha256(PARENT / 'screen_30ep.json')
        assert plan['cases'] == list(CASES) and plan['seeds'] == list(SEEDS)
        assert plan['epochs'] == 100 and plan['official_test_allowed'] is False
        assert read(ROOT/'parent_audit.json')['status'] == 'PASS'
        assert read(ROOT/'frozen_s0_s3.json')['status'] == 'COMPLETED_DEVELOPMENT_DIAGNOSTIC'
        protocol = d2.verify()
        train = d2.make_data(protocol, 'train')
        validation = d2.make_data(protocol, 'validation')
        assert len(train)==3000 and len(validation)==500
        cases = {case['id']:case for case in d2.cfg['cases']}
        d2.ROOT = ROOT / 'bridge'
        d2.ROOT.mkdir(exist_ok=True)
        d2.write_json(ROOT/'runtime.json', {
            'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'hostname':subprocess.check_output(['hostname'],text=True).strip(),
            'gpu':d2.torch.cuda.get_device_name(),
            'torch':str(d2.torch.__version__),
            'cuda':d2.torch.version.cuda,
            'factorial_plan_sha256':d2.sha256(PLAN),
            'parent_protocol_sha256':d2.sha256(d2.CONFIG),
        })
        d2.write_json(ROOT/'status.json', {'status':'RUNNING_BRIDGE'})
        bridge = []
        for case_id in ('A_w0','A_w1','A_w3'):
            seed = 20261114
            d2.run_trial(protocol,train,validation,cases[case_id],seed,1)
            name=f'{case_id}_seed{seed}'
            source = read(PARENT/'runs'/name/'history.json')[0]
            retest = read(d2.ROOT/'runs'/name/'history.json')[0]
            assert source['train_order_sha256'] == retest['train_order_sha256']
            bridge.append({
                'case':case_id,'seed':seed,
                'train_order_match':True,
                'validation_confusion_exact':source['validation_clean']['confusion']==retest['validation_clean']['confusion'],
                'boundary_confusion_exact':source['validation_clean']['boundary_confusion']==retest['validation_clean']['boundary_confusion'],
                'macro_f1_difference_pp':100*(retest['score']-source['score']),
                'boundary_miou_difference_pp':100*(retest['validation_clean']['boundary']['miou']-source['validation_clean']['boundary']['miou']),
                'loss_difference':retest['loss']-source['loss'],
            })
        d2.write_json(ROOT/'bridge_first_epoch.json',bridge)
        if any(abs(x[k])>.5 for x in bridge for k in ['macro_f1_difference_pp','boundary_miou_difference_pp']):
            print('Hardware migration bridge differs by more than 0.5 pp. Rerunning same-device A and depthwise controls.',flush=True)
            task_ids = ['A_w0','A_w1','A_w3','depthwise_zero_w0','depthwise_zero_w1','depthwise_zero_w3']
            reuse_screen = False
        else:
            task_ids = list(CASES)
            reuse_screen = True
        d2.ROOT = ROOT
        screen = {row['case']['id']:row for row in read(PARENT/'screen_30ep.json')}
        results = []
        for seed in SEEDS:
            for case_id in task_ids:
                case = cases[case_id]
                name = f'{case_id}_seed{seed}'
                run = ROOT/'runs'/name
                if seed==20261113 and reuse_screen and not run.exists():
                    src = PARENT/'runs'/name
                    s = screen[case_id]
                    assert d2.sha256(src/'best.pt') == s['best_sha256']
                    assert d2.sha256(src/'last.pt') == s['last_sha256']
                    assert [x['epoch'] for x in read(src/'history.json')]==list(range(1,31))
                    temp = run.with_name(run.name + '.importing')
                    temp.mkdir(parents=True,exist_ok=False)
                    for n in ('config.json','history.json','best.pt','last.pt'):
                        shutil.copyfile(src/n,temp/n)
                    d2.write_json(temp/'continuation_origin.json',{'parent_run':str(src),'parent_epochs':30,'source_last_sha256':s['last_sha256'],'source_best_sha256':s['best_sha256']})
                    temp.rename(run)
                results.append(d2.run_trial(protocol,train,validation,case,seed,100))
                d2.write_json(ROOT/'factorial_progress.json', results)
        old = {(x['case']['id'],x['seed']):x for x in read(PARENT/'deep_100ep.json')}
        combined = {**old,**{(x['case']['id'],x['seed']):x for x in results}}
        paired = []
        for w in (0,1,3):
            for seed in SEEDS:
                d = combined[f'depthwise_zero_w{w}',seed]
                a = combined[f'A_w{w}',seed]
                a0 = combined['A_w0',seed]
                order_a=read(Path(a['path'])/'history.json')
                order_d=read(Path(d['path'])/'history.json')
                assert [v['train_order_sha256'] for v in order_a]==[v['train_order_sha256'] for v in order_d]
                assert d['initial_trunk_sha256']==a['initial_trunk_sha256']==a0['initial_trunk_sha256']
                paired.append({'weight':w,'seed':seed,
                    'architecture_pp':{k:100*(d[k]-a[k]) for k in METRICS},
                    'loss_only_pp':{k:100*(a[k]-a0[k]) for k in METRICS},
                    'combined_pp':{k:100*(d[k]-a0[k]) for k in METRICS}})
        summary = []
        for w in (0,1,3):
            ss = [x for x in paired if x['weight']==w]
            row = {'weight':w}
            for key in ('architecture_pp','loss_only_pp','combined_pp'):
                row[key] = {m:{'mean':statistics.mean(x[key][m] for x in ss),'sd':statistics.stdev(x[key][m] for x in ss)} for m in METRICS}
            row['architecture_boundary_all_positive'] = all(x['architecture_pp']['boundary_miou']>0 for x in ss)
            row['combined_boundary_all_positive'] = all(x['combined_pp']['boundary_miou']>0 for x in ss)
            summary.append(row)
        d2.write_json(ROOT/'factorial_completion.json', {'status':'COMPLETED_DEVELOPMENT_ONLY','same_device_controls_retrained':not reuse_screen,'jobs':results,'paired':paired,'summary':summary,'official_test_allowed':False})
        d2.write_json(ROOT/'status.json', {'status':'COMPLETED_DEVELOPMENT_ONLY'})

if __name__=='__main__':
    try:
        main()
    except BaseException:
        d2.write_json(ROOT/'factorial_failure.json',{'status':'STOPPED_TECHNICAL_FAILURE','traceback':traceback.format_exc()})
        raise

