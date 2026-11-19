"""D2 closing validation: 8-class bias vs class-balanced global residual on frozen A and B_ind."""
import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
DEV = HERE.parent
ROOT = DEV.parent
FIRST = DEV / 'D2_direction_screen_20261118_v1'
REF = DEV / 'D2_class_balanced_refine_20261119_v1'
EVAL = ROOT / 'contracts/P2_formal_eval_20261118_v2'
TRAIN = ROOT / 'ruan_delivery_20261117/training_494ebd3'
sys.path[:0] = [str(REF), str(FIRST), str(EVAL / 'code'), str(TRAIN)]

from factorial_scene_diagnostic import (  # noqa: E402
    CAP, RADIUS, SCENES, SEEDS, read, configure_runtime, read_contract, read_csv,
    load_prepared, boundary_band, load_scene_rows, load_masks, apply_scene,
    confusion, summarize, sha, write_json,
)
from d2_models import build_dev, Refine  # noqa: E402
from forty_adapter import flatten_s2, encode_labels, IGNORE_INDEX  # noqa: E402
from pipeline import boundary_mask, load_protocol, make_data, loader, seed_everything  # noqa: E402

EPOCHS = 15
BATCH = 32
UPDATES = 94
PLAN = HERE / 'run_plan.json'
STATUS = HERE / 'status.json'
LOCK = HERE / 'bias_pair.lock'
HEADS = HERE / 'heads'
VARIANTS = ('bias', 'D2')
GROUPS = ('A', 'B')
STAGES = ('frozen', 'bias', 'D2')


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class BiasHead(nn.Module):
    """Eight class constants; correction = CAP * tanh(bias_k)."""
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(8))

    def forward(self, baseline):
        return baseline + CAP * torch.tanh(self.bias).view(1, 8, 1, 1)


def get_features(net, x):
    features = {}
    def hook(name):
        def capture(module, inputs, outputs):
            features[name] = outputs
        return capture
    handlers = [net.enc1.register_forward_hook(hook('shallow')),
                net.up1.register_forward_hook(hook('full'))]
    try:
        scores = net(x)
        return torch.cat((features['full'], features['shallow']), 1), scores
    finally:
        for h in handlers:
            h.remove()


def d2_adjust(head, features):
    return CAP * torch.tanh(head.head(head.body(features)))


def weighted_ce_terms(pred, target, weight):
    valid = target != IGNORE_INDEX
    scores = pred.movedim(1, -1)[valid]
    labels = target[valid]
    valid_weight = weight[valid]
    ce = torch.logsumexp(scores, dim=1) - scores.gather(1, labels[:, None])[:, 0]
    return (ce * valid_weight).sum(), valid_weight.sum()


def head_param_count(head):
    return sum(p.numel() for p in head.parameters())


def snapshot_state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def state_unchanged(net, snapshot):
    now = net.state_dict()
    return set(now) == set(snapshot) and all(torch.equal(now[k].detach().cpu(), snapshot[k]) for k in snapshot)


def source_paths():
    return {
        'runner': Path(__file__),
        'factorial_scene_diagnostic': REF / 'factorial_scene_diagnostic.py',
        'd2_models': FIRST / 'd2_models.py',
        'forest_unet': TRAIN / 'forest_unet.py',
        'pipeline': TRAIN / 'pipeline.py',
        'forty_adapter': TRAIN / 'forty_adapter.py',
        'eval_common': EVAL / 'code/common.py',
        'eval_contracts': EVAL / 'code/contracts_v2.py',
        'eval_metrics': EVAL / 'code/metrics_v2.py',
        'eval_scenes': EVAL / 'code/scenes_v2.py',
        'eval_evaluate': EVAL / 'code/evaluate_v2.py',
    }


def collect_plan():
    ref_plan = read(REF / 'pilot_plan.json')
    ref_audit = read(REF / 'pilot_audit.json')
    ref_completion = read(REF / 'pilot_completion.json')
    deep = read(FIRST / 'deep_100ep.json')
    handoff = read(ROOT / 'results/P2_core_merged_handoff_atomic10_20261118.json')
    contract = read(EVAL / 'P2_eval_contract_FROZEN_20261118_v2.json')

    assert ref_audit['status'] == 'PASS'
    assert ref_audit['frozen_baseline_all_36_confusions_exact_across_pilots'] is True
    assert ref_completion['status'] == 'COMPLETED_DEVELOPMENT_ONLY'
    assert ref_completion['official_test_accessed'] is False
    assert ref_plan['epochs'] == EPOCHS and ref_plan['seeds'] == list(SEEDS)
    assert ref_plan['residual_logit_cap'] == CAP and ref_plan['radius'] == RADIUS
    assert ref_plan['evaluation_contract_sha256'] == sha(EVAL / 'P2_eval_contract_FROZEN_20261118_v2.json')

    a_rows = {int(j['seed']): j for j in deep if j['case']['id'] == 'A_w0'}
    assert sorted(a_rows) == list(SEEDS)
    a_base = {str(s): a_rows[s]['best_sha256'] for s in SEEDS}
    for s in SEEDS:
        p = Path(a_rows[s]['path']) / 'best.pt'
        assert sha(p) == a_base[str(s)] == ref_plan['base_checkpoints'][str(s)]

    b_rows = {int(j['seed']): j for j in handoff['checkpoints'] if j.get('delivery_group') == 'B_ind'}
    assert sorted(b_rows) == list(SEEDS)
    b_base = {str(s): b_rows[s]['sha256'] for s in SEEDS}
    for s in SEEDS:
        assert sha(Path(b_rows[s]['path'])) == b_base[str(s)]

    a_d2 = {f'global_seed{s}.pt': ref_audit['checkpoints'][f'global_seed{s}.pt'] for s in SEEDS}
    for name, value in a_d2.items():
        assert sha(REF / name) == value

    data_manifest = contract['data']['manifest']
    data_index = contract['data']['index']
    scene_manifest = contract['scenes']['manifest']
    scene_audit = contract['scenes']['audit']
    assert sha(Path(data_manifest)) == contract['data']['manifest_sha256']
    assert sha(Path(data_index)) == contract['data']['index_sha256']
    assert sha(Path(scene_manifest)) == contract['scenes']['manifest_sha256']
    assert sha(Path(scene_audit)) == contract['scenes']['audit_sha256']
    assert contract['data']['counts'] == {'train': 3000, 'validation': 500}
    assert contract['official_test_allowed'] is False

    sources = {name: {'path': str(p), 'sha256': sha(p)} for name, p in source_paths().items()}
    plan = {
        'version': 'D2_bias_and_B_pair_20261119_v1',
        'created_utc': now_utc(),
        'scope': 'D2 closing development validation: 8-class bias vs class-balanced global residual on frozen A and B_ind. Original train3000/validation500 and frozen S0-S3 scenes only; official test sealed.',
        'official_test_allowed': False,
        'seeds': list(SEEDS),
        'stages': list(GROUPS),
        'matrix': {
            'A': ['A_frozen', 'A_bias', 'A_D2'],
            'B': ['B_frozen', 'B_bias', 'B_D2'],
            'new_heads': [f'A_bias_seed{s}' for s in SEEDS] + [f'B_{v}_seed{s}' for v in VARIANTS for s in SEEDS],
            'reused_heads': [f'A_frozen_seed{s}' for s in SEEDS] + [f'A_D2_seed{s}' for s in SEEDS],
        },
        'evaluation_groups': [f'{g}_{v}' for g in GROUPS for v in STAGES],
        'parameters': {
            'epochs': EPOCHS,
            'batch_size': BATCH,
            'updates_per_epoch': UPDATES,
            'optimizer': {'name': 'Adam', 'lr': 0.001},
            'gradient_clip_norm': 10.0,
            'precision': 'float32',
            'residual_logit_cap': CAP,
            'boundary_weight': 1,
            'boundary_radius': RADIUS,
            'class_weight': 'sqrt(mean(weight_mass)/weight_mass).clamp(0.5,3)',
            'checkpoint_selection': 'fixed final epoch 15, no selection',
            'bias_head': {'parameters': 8, 'initialization': 'zeros', 'formula': '0.75*tanh(b_k)'},
            'd2_head': {'parameters': 856, 'architecture': 'depthwise_zero global', 'initialization': 'zero final conv'},
        },
        'data': {
            'manifest': data_manifest,
            'manifest_sha256': contract['data']['manifest_sha256'],
            'index': data_index,
            'index_sha256': contract['data']['index_sha256'],
            'prepared_root': contract['data']['prepared_root'],
            'scenes_manifest': scene_manifest,
            'scenes_manifest_sha256': contract['scenes']['manifest_sha256'],
            'scenes_audit': scene_audit,
            'scenes_audit_sha256': contract['scenes']['audit_sha256'],
            'train': 3000,
            'validation': 500,
            'scenes': list(SCENES),
        },
        'contract_sha256': sha(EVAL / 'P2_eval_contract_FROZEN_20261118_v2.json'),
        'parent_development_protocol_sha256': sha(FIRST / 'development_protocol.json'),
        'parent_deep_100ep_sha256': sha(FIRST / 'deep_100ep.json'),
        'A_base_checkpoints': a_base,
        'B_ind_handoff_sha256': sha(ROOT / 'results/P2_core_merged_handoff_atomic10_20261118.json'),
        'B_base_checkpoints': b_base,
        'prior_class_balanced': {
            'plan_sha256': sha(REF / 'pilot_plan.json'),
            'audit_sha256': sha(REF / 'pilot_audit.json'),
            'completion_sha256': sha(REF / 'pilot_completion.json'),
            'global_heads': a_d2,
        },
        'reuse': {
            'A_frozen': 'A_w0 best checkpoints from deep_100ep.json',
            'A_D2': 'class-balanced global_seed*.pt heads from D2_class_balanced_refine_20261119_v1',
            'B_frozen': 'B_ind best checkpoints from P2_core_merged_handoff_atomic10_20261118.json',
        },
        'sources': sources,
        'stop_conditions': [
            'any source, data, contract, base checkpoint, reused head, or plan hash mismatch',
            'any nonfinite loss, incorrect update count, nonzero unknown-pixel weight, or head correction above 0.75',
            'any frozen-backbone parameter or BatchNorm statistic change',
            'any failure to load or strictly reload a checkpoint',
            'any missing group/seed/scene/region result',
            'official test access or any full/test evaluation',
        ],
        'runner_sha256': sha(Path(__file__)),
    }
    return plan


def verify_plan(plan):
    assert plan['version'] == 'D2_bias_and_B_pair_20261119_v1'
    assert plan['runner_sha256'] == sha(Path(__file__))
    assert plan['seeds'] == list(SEEDS) and plan['stages'] == list(GROUPS)
    assert plan['parameters']['epochs'] == EPOCHS
    assert plan['parameters']['batch_size'] == BATCH
    assert plan['parameters']['updates_per_epoch'] == UPDATES
    assert plan['parameters']['residual_logit_cap'] == CAP
    assert plan['parameters']['boundary_radius'] == RADIUS
    assert plan['official_test_allowed'] is False
    for name, item in plan['sources'].items():
        assert sha(Path(item['path'])) == item['sha256'], name
    assert sha(EVAL / 'P2_eval_contract_FROZEN_20261118_v2.json') == plan['contract_sha256']
    assert sha(FIRST / 'development_protocol.json') == plan['parent_development_protocol_sha256']
    assert sha(FIRST / 'deep_100ep.json') == plan['parent_deep_100ep_sha256']
    assert sha(ROOT / 'results/P2_core_merged_handoff_atomic10_20261118.json') == plan['B_ind_handoff_sha256']
    assert sha(Path(plan['data']['manifest'])) == plan['data']['manifest_sha256']
    assert sha(Path(plan['data']['index'])) == plan['data']['index_sha256']
    assert sha(Path(plan['data']['scenes_manifest'])) == plan['data']['scenes_manifest_sha256']
    assert sha(Path(plan['data']['scenes_audit'])) == plan['data']['scenes_audit_sha256']
    assert sha(REF / 'pilot_plan.json') == plan['prior_class_balanced']['plan_sha256']
    assert sha(REF / 'pilot_audit.json') == plan['prior_class_balanced']['audit_sha256']
    assert sha(REF / 'pilot_completion.json') == plan['prior_class_balanced']['completion_sha256']
    for s in SEEDS:
        assert sha(REF / f'global_seed{s}.pt') == plan['prior_class_balanced']['global_heads'][f'global_seed{s}.pt']
    for s in SEEDS:
        deep = read(FIRST / 'deep_100ep.json')
        row = next(j for j in deep if j['case']['id'] == 'A_w0' and j['seed'] == s)
        assert sha(Path(row['path']) / 'best.pt') == plan['A_base_checkpoints'][str(s)]
        handoff = read(ROOT / 'results/P2_core_merged_handoff_atomic10_20261118.json')
        b = next(j for j in handoff['checkpoints'] if j.get('delivery_group') == 'B_ind' and j['seed'] == s)
        assert sha(Path(b['path'])) == plan['B_base_checkpoints'][str(s)]


def load_A(seed, plan):
    deep = read(FIRST / 'deep_100ep.json')
    row = next(j for j in deep if j['case']['id'] == 'A_w0' and j['seed'] == seed)
    checkpoint = Path(row['path']) / 'best.pt'
    assert sha(checkpoint) == row['best_sha256'] == plan['A_base_checkpoints'][str(seed)]
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['seed'] == seed and saved['case']['id'] == 'A_w0'
    net = build_dev('A').cuda().eval().requires_grad_(False)
    net.load_state_dict(saved['state_dict'], strict=True)
    return net, row['best_sha256']


def load_B(seed, plan):
    handoff = read(ROOT / 'results/P2_core_merged_handoff_atomic10_20261118.json')
    row = next(j for j in handoff['checkpoints']
               if j.get('delivery_group') == 'B_ind' and j['seed'] == seed)
    checkpoint = Path(row['path'])
    assert sha(checkpoint) == row['sha256'] == plan['B_base_checkpoints'][str(seed)]
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['group'] == 'B_ind' and saved['seed'] == seed
    assert saved['model_config'] == {'in_channels': 40, 'num_classes': 8,
                                     'base_channels': 16, 'use_edge': False}
    net = build_dev('A').cuda().eval().requires_grad_(False)
    net.load_state_dict(saved['state_dict'], strict=True)
    return net, row['sha256']


def load_A_D2(seed, plan):
    path = REF / f'global_seed{seed}.pt'
    assert sha(path) == plan['prior_class_balanced']['global_heads'][f'global_seed{seed}.pt']
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['mode'] == 'global' and saved['seed'] == seed
    assert saved['base_sha256'] == plan['A_base_checkpoints'][str(seed)]
    assert saved['plan_sha256'] == plan['prior_class_balanced']['plan_sha256']
    head = Refine('depthwise_zero').cuda().eval()
    head.load_state_dict(saved['state_dict'], strict=True)
    return head


def cache_train(stage, seed, net, data, cfg):
    n = len(data)
    x = torch.empty((n, 32, 128, 128), device='cuda', dtype=torch.float32)
    z = torch.empty((n, 8, 128, 128), device='cuda', dtype=torch.float32)
    target = torch.empty((n, 128, 128), device='cuda', dtype=torch.int64)
    weight = torch.empty((n, 128, 128), device='cuda', dtype=torch.float32)
    pos = 0
    with torch.no_grad():
        for batch in loader(data, cfg, training=False):
            raw = batch['label'].cuda()
            features, scores = get_features(net, flatten_s2(batch['s2'].cuda()))
            ix = slice(pos, pos + len(raw))
            x[ix] = features
            z[ix] = scores
            target[ix] = encode_labels(raw)
            weight[ix] = (raw != 0).float() * (1 + boundary_mask(raw, RADIUS).float())
            pos += len(raw)
    assert pos == 3000
    valid = target != IGNORE_INDEX
    masses = torch.stack([weight[target == cls].sum() for cls in range(8)])
    pixels = torch.stack([(target == cls).sum() for cls in range(8)])
    assert torch.isfinite(masses).all() and bool((masses > 0).all())
    scales = torch.sqrt(masses.mean() / masses).clamp(.5, 3)
    weight *= scales[target.clamp_min(0)]
    assert bool((weight[~valid] == 0).all())
    prior = read(REF / f'loss_balance_seed{seed}.json')
    assert torch.allclose(masses.cpu(), torch.tensor(prior['train_only_weight_mass'], dtype=torch.float32), rtol=1e-6, atol=1e-6)
    assert torch.equal(pixels.cpu(), torch.tensor(prior['train_only_pixels'], dtype=torch.int64))
    assert torch.allclose(scales.cpu(), torch.tensor(prior['scales'], dtype=torch.float32), rtol=1e-6, atol=1e-6)
    balance = {'train_only_weight_mass': masses.tolist(),
               'train_only_pixels': pixels.tolist(),
               'scales': scales.tolist()}
    write_json(HERE / f'loss_balance_{stage}_seed{seed}.json', balance)
    return x, z, target, weight, balance


def train_head(stage, variant, seed, x, z, target, weight, base_hash, plan_hash, balance):
    assert variant in VARIANTS
    seed_everything(seed, True)
    head = BiasHead().cuda() if variant == 'bias' else Refine('depthwise_zero').cuda()
    with torch.no_grad():
        head.eval()
        if variant == 'bias':
            assert torch.equal(z[:4], head(z[:4]))
        else:
            assert torch.equal(z[:4], z[:4] + d2_adjust(head, x[:4]))
    expected_params = 8 if variant == 'bias' else 856
    assert head_param_count(head) == expected_params
    opt = torch.optim.Adam(head.parameters(), lr=.001)
    head.train()
    history = []
    for epoch in range(1, EPOCHS + 1):
        begin = time.perf_counter()
        permutation = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed + epoch))
        order_sha = hashlib.sha256(permutation.numpy().tobytes()).hexdigest()
        loss_num = 0.0
        valid_weight = 0.0
        steps = 0
        max_adjust = 0.0
        for ids in permutation.split(BATCH):
            ids = ids.cuda()
            im, baseline, labels, weights = x[ids], z[ids], target[ids], weight[ids]
            opt.zero_grad(set_to_none=True)
            if variant == 'bias':
                adjust = CAP * torch.tanh(head.bias).view(1, 8, 1, 1)
                pred = baseline + adjust
            else:
                adjust = d2_adjust(head, im)
                pred = baseline + adjust
            numerator, denominator = weighted_ce_terms(pred, labels, weights)
            if not bool(denominator > 0):
                continue
            loss = numerator / denominator
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 10, error_if_nonfinite=True)
            opt.step()
            loss_num += float(numerator.detach())
            valid_weight += float(denominator)
            max_adjust = max(max_adjust, float(adjust.detach().abs().max()))
            steps += 1
        assert steps == UPDATES
        assert max_adjust <= CAP + 1e-7
        history.append({'epoch': epoch, 'weighted_ce': loss_num / valid_weight,
                        'updates': steps, 'skipped': 0, 'seconds': time.perf_counter() - begin,
                        'order_sha256': order_sha, 'max_abs_adjust': max_adjust})
        write_json(STATUS, {'status': 'RUNNING', 'phase': 'training', 'stage': stage,
                            'variant': variant, 'seed': seed, 'epoch': epoch,
                            'target': EPOCHS, 'history': history[-1]})
        print(json.dumps({'stage': stage, 'variant': variant, 'seed': seed,
                          'epoch': epoch, 'seconds': history[-1]['seconds'],
                          'loss': history[-1]['weighted_ce']}), flush=True)
    head.eval()
    payload = {
        'format': 'D2-bias-B-pair-v1',
        'development_only': True,
        'stage': stage,
        'variant': variant,
        'seed': seed,
        'epochs': EPOCHS,
        'base_sha256': base_hash,
        'plan_sha256': plan_hash,
        'loss_balance': balance,
        'history': history,
        'state_dict': head.state_dict(),
    }
    HEADS.mkdir(exist_ok=True)
    path = HEADS / f'{stage}_{variant}_seed{seed}.pt'
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.rename(path)
    digest = sha(path)
    reloaded = torch.load(path, map_location='cpu', weights_only=True)
    head.load_state_dict(reloaded['state_dict'], strict=True)
    assert reloaded['stage'] == stage and reloaded['variant'] == variant and reloaded['seed'] == seed
    print(json.dumps({'stage': stage, 'variant': variant, 'seed': seed,
                      'head_sha256': digest, 'params': expected_params}), flush=True)
    return head, {'stage': stage, 'variant': variant, 'seed': seed,
                  'head_sha256': digest, 'params': expected_params,
                  'history': history, 'loss_balance': balance}


def evaluate(models, contract, manifest):
    val = [row for row in manifest if row['split'] == 'validation']
    index = {(int(row['sample_id']), row['split']): row for row in read_csv(contract['data']['index'])}
    samples = {int(row['sample_id']): load_prepared(row, contract['data']['prepared_root'], index)[:2]
               for row in val}
    assert len(samples) == 500
    rows = load_scene_rows(contract['scenes']['manifest'], set(samples), contract['stage'], SCENES)
    assert len(rows) == 2000
    bands = {sid: boundary_band(y, RADIUS) for sid, (x, y) in samples.items()}
    totals = collections.defaultdict(lambda: np.zeros((9, 9), np.int64))
    transitions = collections.defaultdict(collections.Counter)
    with torch.inference_mode():
        for count, row in enumerate(rows, 1):
            sid, scene = int(row['sample_id']), row['scene']
            x, y = samples[sid]
            band = bands[sid]
            valid = y != 0
            _, soft = load_masks(row, contract['scenes']['manifest'])
            x = apply_scene(x, row, soft)
            tensor = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2).reshape(1, 40, 128, 128))).cuda()
            for stage in GROUPS:
                for seed, (net, heads) in models[stage].items():
                    features, baseline = get_features(net, tensor)
                    old = (baseline.argmax(1)[0].cpu().numpy() + 1).astype(np.uint8)
                    preds = {'frozen': old}
                    if 'bias' in heads:
                        preds['bias'] = ((baseline + CAP * torch.tanh(heads['bias'].bias).view(1, 8, 1, 1)).argmax(1)[0].cpu().numpy() + 1).astype(np.uint8)
                    if 'D2' in heads:
                        preds['D2'] = ((baseline + d2_adjust(heads['D2'], features)).argmax(1)[0].cpu().numpy() + 1).astype(np.uint8)
                    for variant, pred in preds.items():
                        name = f'{stage}_{variant}'
                        for region, mask in [('whole', valid), ('boundary', band), ('interior', valid & ~band)]:
                            totals[name, seed, scene, region] += confusion(pred, y, mask)
                            if variant != 'frozen':
                                c = transitions[name, seed, scene, region]
                                c['fixed'] += int(((old != y) & (pred == y) & mask).sum())
                                c['broken'] += int(((old == y) & (pred != y) & mask).sum())
                                c['changed'] += int(((old != pred) & mask).sum())
            if count % 250 == 0:
                write_json(STATUS, {'status': 'RUNNING', 'phase': 'evaluation',
                                    'scene_rows': count, 'target': len(rows)})
                print(json.dumps({'evaluation_rows': count, 'target': len(rows)}), flush=True)
    results = []
    for (name, seed, scene, region), cm in sorted(totals.items()):
        stage, variant = name.split('_', 1)
        results.append({'group': stage, 'variant': variant, 'seed': seed, 'scene': scene,
                        'region': region, 'metrics': summarize(cm),
                        'confusion': cm.tolist(),
                        'transitions': dict(transitions[name, seed, scene, region])})
    return results


def flat_metrics(metrics):
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = float(value)
        elif isinstance(value, list) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            for i, v in enumerate(value):
                out[f'{key}.{i}'] = float(v)
    return out


def prior_alignment(results):
    prior = read(REF / 'pilot_completion.json')
    old = {(r['mode'], r['seed'], r['scene'], r['region']): r for r in prior['results']}
    new = {(r['variant'], r['seed'], r['scene'], r['region']): r for r in results if r['group'] == 'A'}
    report = {}
    for label, prior_label in [('frozen', 'A_frozen'), ('D2', 'global')]:
        diffs = []
        exact = 0
        support_exact = 0
        compared = 0
        examples = []
        for seed in SEEDS:
            for scene in SCENES:
                for region in ('whole', 'boundary', 'interior'):
                    a = new[label, seed, scene, region]
                    b = old[prior_label, seed, scene, region]
                    fa, fb = flat_metrics(a['metrics']), flat_metrics(b['metrics'])
                    keys = set(fa) & set(fb)
                    diff = max((abs(fa[k] - fb[k]) for k in keys), default=0.0)
                    diffs.append(diff)
                    same = a['confusion'] == b['confusion']
                    exact += int(same)
                    support_exact += int(a['metrics'].get('support') == b['metrics'].get('support'))
                    compared += 1
                    if region == 'boundary' and (seed, scene) in ((20261113, 'S0'), (20261115, 'S2')):
                        examples.append({'seed': seed, 'scene': scene, 'region': region,
                                         'new_variant': label, 'prior_mode': prior_label,
                                         'confusion_exact': same,
                                         'new_sum': int(np.asarray(a['confusion']).sum()),
                                         'prior_sum': int(np.asarray(b['confusion']).sum())})
        report[label] = {'compared': compared,
                         'max_abs_metric_difference': max(diffs) if diffs else None,
                         'confusions_exact': exact,
                         'support_exact': support_exact,
                         'status': 'EXACT' if compared == 36 and exact == 36 and (max(diffs) <= 1e-12 if diffs else False) else 'DIFFERENT',
                         'examples': examples}
    return {'prior_completion_sha256': sha(REF / 'pilot_completion.json'),
            'A_frozen_vs_A_frozen': report['frozen'],
            'A_D2_vs_global': report['D2'],
            'note': 'A_frozen and A_D2 are repeated on the current machine; paired comparisons below use this replay. Differences, if any, are recorded rather than hidden.'}


def paired_analysis(results):
    index = {(r['group'], r['variant'], r['seed'], r['scene'], r['region']): r for r in results}
    comparisons = [('bias_minus_frozen', 'bias', 'frozen'),
                   ('D2_minus_frozen', 'D2', 'frozen'),
                   ('D2_minus_bias', 'D2', 'bias')]
    boundary_metrics = ('boundary_miou', 'miou', 'forest_macro_f1', 'planted_f1', 'macro_f1')
    whole_metrics = ('miou', 'forest_macro_f1', 'planted_f1', 'macro_f1')
    rows = []
    for group in GROUPS:
        for comp, left, right in comparisons:
            for scene in SCENES:
                for region, metrics in (('boundary', boundary_metrics), ('whole', whole_metrics)):
                    for metric in metrics:
                        values = []
                        na = False
                        for seed in SEEDS:
                            a = index[group, left, seed, scene, region]['metrics'].get(metric)
                            b = index[group, right, seed, scene, region]['metrics'].get(metric)
                            if a is None or b is None:
                                na = True
                                values.append(None)
                            else:
                                values.append(100 * (a - b))
                        if na:
                            mean = sd = None
                            all_positive = None
                        else:
                            mean = statistics_mean(values)
                            sd = statistics_sd(values)
                            all_positive = all(v > 0 for v in values)
                        rows.append({'group': group, 'comparison': comp, 'scene': scene,
                                     'metric': metric, 'region': region,
                                     'per_seed': values, 'mean': mean, 'sd': sd,
                                     'all_positive': all_positive, 'na': na,
                                     'neutral': None if na else all(abs(v) < 1e-12 for v in values)})

    def lookup(group, comp, scene, metric, region):
        return next((r for r in rows if r['group'] == group and r['comparison'] == comp
                     and r['scene'] == scene and r['metric'] == metric and r['region'] == region), None)

    gates = {}
    for group in GROUPS:
        gates[group] = {}
        for comp, _, _ in comparisons:
            s0 = lookup(group, comp, 'S0', 'boundary_miou', 'boundary')
            s2 = lookup(group, comp, 'S2', 'boundary_miou', 'boundary')
            s3 = lookup(group, comp, 'S3', 'boundary_miou', 'boundary')
            s0_whole = lookup(group, comp, 'S0', 'miou', 'whole')
            s0_forest = lookup(group, comp, 'S0', 'forest_macro_f1', 'whole')
            s2_forest = lookup(group, comp, 'S2', 'forest_macro_f1', 'whole')
            s3_forest = lookup(group, comp, 'S3', 'forest_macro_f1', 'whole')
            s0_planted = lookup(group, comp, 'S0', 'planted_f1', 'whole')
            s2_planted = lookup(group, comp, 'S2', 'planted_f1', 'whole')
            s3_planted = lookup(group, comp, 'S3', 'planted_f1', 'whole')
            gate = None
            if group == 'A' and comp in ('bias_minus_frozen', 'D2_minus_frozen'):
                gate = bool(s0 and s0['mean'] is not None and s0['mean'] >= 1.0
                            and s0['all_positive'] is True
                            and s0_whole and s0_whole['mean'] is not None and s0_whole['mean'] >= 0
                            and s0_forest and s0_forest['mean'] is not None and s0_forest['mean'] >= -1)
            if group == 'B' and comp in ('bias_minus_frozen', 'D2_minus_frozen'):
                gate = bool(s2 and s3 and s2['mean'] is not None and s3['mean'] is not None
                            and s2['mean'] >= 1 and s3['mean'] >= 1
                            and s2['all_positive'] is True and s3['all_positive'] is True
                            and all(x and x['mean'] is not None and x['mean'] >= -1
                                    for x in (s0_forest, s2_forest, s3_forest)))
            gates[group][comp] = {
                's0_boundary': s0, 's2_boundary': s2, 's3_boundary': s3,
                's0_whole_miou': s0_whole, 's0_whole_forest': s0_forest,
                's2_whole_forest': s2_forest, 's3_whole_forest': s3_forest,
                's0_planted_f1': s0_planted,
                's2_planted_f1': s2_planted, 's3_planted_f1': s3_planted,
                'reference_gate_pass': gate,
                'reference_gate_scope': ('A S0 boundary mean>=1, all seeds>0, whole miou mean>=0, forest mean>=-1'
                                         if group == 'A' else
                                         'B S2/S3 boundary mean>=1 all seeds>0; S0/S2/S3 whole forest mean>=-1'),
            }
    return rows, gates


def statistics_mean(values):
    return sum(values) / len(values)


def statistics_sd(values):
    mean = statistics_mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def preflight(plan):
    verify_plan(plan)
    configure_runtime('cuda', 4)
    proto = read(FIRST / 'development_protocol.json')
    assert sha(FIRST / 'development_protocol.json') == plan['parent_development_protocol_sha256']
    assert sha(Path(proto['parent_protocol'])) == proto['parent_protocol_sha256']
    cfg, _ = load_protocol(proto['parent_protocol'])
    data = make_data(cfg, 'train')
    assert len(data) == 3000
    contract, digest, manifest = read_contract(EVAL / 'P2_eval_contract_FROZEN_20261118_v2.json')
    assert digest == plan['contract_sha256']
    val = [r for r in manifest if r['split'] == 'validation']
    assert len(val) == 500
    checks = {}
    for stage in GROUPS:
        net, base_hash = (load_A(SEEDS[0], plan) if stage == 'A' else load_B(SEEDS[0], plan))
        x, z, target, weight, balance = cache_train(stage, SEEDS[0], net, data, cfg)
        checks[stage] = {'balance': balance}
        for variant in VARIANTS:
            seed_everything(SEEDS[0], True)
            head = BiasHead().cuda() if variant == 'bias' else Refine('depthwise_zero').cuda()
            with torch.no_grad():
                head.eval()
                if variant == 'bias':
                    identical = torch.equal(z[:4], head(z[:4]))
                    adjust = CAP * torch.tanh(head.bias).view(1, 8, 1, 1)
                else:
                    identical = torch.equal(z[:4], z[:4] + d2_adjust(head, x[:4]))
                    adjust = d2_adjust(head, x[:4])
            assert identical, (stage, variant)
            assert float(adjust.abs().max()) <= CAP + 1e-7
            assert head_param_count(head) == (8 if variant == 'bias' else 856)
            opt = torch.optim.Adam(head.parameters(), lr=.001)
            head.train()
            opt.zero_grad(set_to_none=True)
            pred = head(z[:BATCH]) if variant == 'bias' else z[:BATCH] + d2_adjust(head, x[:BATCH])
            numerator, denominator = weighted_ce_terms(pred, target[:BATCH], weight[:BATCH])
            assert bool(denominator > 0)
            (numerator / denominator).backward()
            backbone_grads = [p.grad for p in net.parameters() if p.grad is not None]
            head_grads = [p.grad for p in head.parameters() if p.grad is not None]
            assert len(backbone_grads) == 0, (stage, variant, len(backbone_grads))
            assert len(head_grads) > 0, (stage, variant)
            assert bool((weight[target == IGNORE_INDEX] == 0).all())
            max_bound = float((CAP * torch.tanh(torch.tensor([-100.0, 0.0, 100.0], device='cuda'))).abs().max())
            assert max_bound <= CAP + 1e-7
            checks[stage][variant] = {
                'zero_init_identity_exact': True,
                'params': head_param_count(head),
                'max_abs_adjust_init': float(adjust.abs().max()),
                'backbone_grads': len(backbone_grads),
                'head_grads': len(head_grads),
                'unknown_weight_zero': True,
                'tanh_bound_check': max_bound,
            }
        del x, z, target, weight
        torch.cuda.empty_cache()
    payload = {'status': 'PASS', 'checks': checks,
               'official_test_accessed': False,
               'plan_sha256': sha(PLAN), 'runner_sha256': sha(Path(__file__))}
    write_json(HERE / 'preflight.json', payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def run(plan):
    verify_plan(plan)
    configure_runtime('cuda', 4)
    started = now_utc()
    begin = time.perf_counter()
    write_json(STATUS, {'status': 'RUNNING', 'phase': 'setup', 'started_utc': started})
    contract, digest, manifest = read_contract(EVAL / 'P2_eval_contract_FROZEN_20261118_v2.json')
    assert digest == plan['contract_sha256']
    proto = read(FIRST / 'development_protocol.json')
    assert sha(FIRST / 'development_protocol.json') == plan['parent_development_protocol_sha256']
    assert sha(Path(proto['parent_protocol'])) == proto['parent_protocol_sha256']
    cfg, _ = load_protocol(proto['parent_protocol'])
    data = make_data(cfg, 'train')
    assert len(data) == 3000
    models = {'A': {}, 'B': {}}
    head_summaries = []
    for stage in GROUPS:
        for seed in SEEDS:
            net, base_hash = (load_A(seed, plan) if stage == 'A' else load_B(seed, plan))
            snapshot = snapshot_state(net)
            x, z, target, weight, balance = cache_train(stage, seed, net, data, cfg)
            heads = {}
            made = []
            if stage == 'A':
                bias, summary = train_head('A', 'bias', seed, x, z, target, weight, base_hash, sha(PLAN), balance)
                heads['bias'] = bias
                made.append(summary)
                heads['D2'] = load_A_D2(seed, plan)
            else:
                for variant in VARIANTS:
                    head, summary = train_head('B', variant, seed, x, z, target, weight, base_hash, sha(PLAN), balance)
                    heads[variant] = head
                    made.append(summary)
            unchanged = state_unchanged(net, snapshot)
            assert unchanged, (stage, seed)
            for summary in made:
                summary['backbone_unchanged'] = True
                head_summaries.append(summary)
            if stage == 'B':
                assert [h['order_sha256'] for h in made[0]['history']] == [h['order_sha256'] for h in made[1]['history']]
                for summary in made:
                    summary['pair_order_match'] = True
            models[stage][seed] = (net, heads)
            del x, z, target, weight
            torch.cuda.empty_cache()
    write_json(HERE / 'heads.json', head_summaries)
    results = evaluate(models, contract, manifest)
    alignment = prior_alignment(results)
    pairs, gates = paired_analysis(results)
    completion = {
        'status': 'COMPLETED_DEVELOPMENT_ONLY',
        'official_test_accessed': False,
        'official_test_allowed': False,
        'scope': plan['scope'],
        'version': plan['version'],
        'started_utc': started,
        'completed_utc': now_utc(),
        'seconds': time.perf_counter() - begin,
        'contract_sha256': digest,
        'plan_sha256': sha(PLAN),
        'runner_sha256': sha(Path(__file__)),
        'heads': head_summaries,
        'results': results,
        'prior_alignment': alignment,
        'pairs': pairs,
        'gates': gates,
        'result_counts': {'groups': len(plan['evaluation_groups']), 'seeds': len(SEEDS),
                          'scenes': len(SCENES), 'regions': 3, 'rows': len(results)},
    }
    write_json(HERE / 'bias_pair_completion.json', completion)
    write_json(STATUS, {'status': 'COMPLETED_DEVELOPMENT_ONLY', 'completed_utc': completion['completed_utc']})
    print(json.dumps({'status': completion['status'], 'rows': len(results),
                      'prior_alignment': alignment, 'seconds': completion['seconds']},
                     ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        HEADS.mkdir(exist_ok=True)
        plan = collect_plan()
        write_json(PLAN, plan)
        print(json.dumps({'prepared': str(PLAN), 'runner_sha256': plan['runner_sha256']}))
        return
    plan = read(PLAN)
    if args.preflight:
        preflight(plan)
        return
    if args.run:
        with LOCK.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                run(plan)
            except BaseException:
                write_json(HERE / 'failure.json', {'status': 'STOPPED_TECHNICAL_FAILURE',
                                                   'traceback': traceback.format_exc()})
                write_json(STATUS, {'status': 'FAILED'})
                raise
        return
    parser.error('choose --prepare, --preflight, or --run')


if __name__ == '__main__':
    main()
