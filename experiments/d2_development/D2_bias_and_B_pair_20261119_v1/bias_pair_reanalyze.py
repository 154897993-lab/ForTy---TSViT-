"""Recompute paired tables from raw results using the evaluator's region-local miou key."""
import json
import math
import shutil
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMP = HERE / 'bias_pair_completion.json'
RAW = HERE / 'bias_pair_completion_original.json'
OUT = HERE / 'bias_pair_completion.json'
SEEDS = (20261113, 20261114, 20261115)
SCENES = ('S0', 'S1', 'S2', 'S3')
GROUPS = ('A', 'B')
COMPARISONS = (('bias_minus_frozen', 'bias', 'frozen'),
               ('D2_minus_frozen', 'D2', 'frozen'),
               ('D2_minus_bias', 'D2', 'bias'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def mean(values):
    return sum(values) / len(values)


def sd(values):
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def main():
    data = read(COMP)
    if not RAW.exists():
        shutil.copyfile(COMP, RAW)
    index = {(r['group'], r['variant'], r['seed'], r['scene'], r['region']): r for r in data['results']}

    def source_metric(group, variant, seed, scene, region, metric):
        row = index[group, variant, seed, scene, region]['metrics']
        if metric == 'boundary_miou' and region == 'boundary':
            return row.get('miou')
        return row.get(metric)

    rows = []
    for group in GROUPS:
        for comp, left, right in COMPARISONS:
            for scene in SCENES:
                for region, metrics in (('boundary', ('boundary_miou', 'forest_macro_f1', 'planted_f1', 'macro_f1')),
                                        ('whole', ('miou', 'forest_macro_f1', 'planted_f1', 'macro_f1'))):
                    for metric in metrics:
                        values = []
                        na = False
                        for seed in SEEDS:
                            a = source_metric(group, left, seed, scene, region, metric)
                            b = source_metric(group, right, seed, scene, region, metric)
                            if a is None or b is None:
                                na = True
                                values.append(None)
                            else:
                                values.append(100 * (a - b))
                        if na:
                            m = s = pos = None
                        else:
                            m = mean(values)
                            s = sd(values)
                            pos = all(v > 0 for v in values)
                        rows.append({'group': group, 'comparison': comp, 'scene': scene,
                                     'metric': metric, 'region': region,
                                     'per_seed': values, 'mean': m, 'sd': s,
                                     'all_positive': pos, 'na': na,
                                     'neutral': None if na else all(abs(v) < 1e-12 for v in values)})

    def lookup(group, comp, scene, metric, region):
        return next((r for r in rows if r['group'] == group and r['comparison'] == comp
                     and r['scene'] == scene and r['metric'] == metric and r['region'] == region), None)

    gates = {}
    for group in GROUPS:
        gates[group] = {}
        for comp, _, _ in COMPARISONS:
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

    original_sha = sha(RAW)
    data['pairs'] = rows
    data['gates'] = gates
    data['analysis'] = {
        'version': 'D2_bias_and_B_pair_analysis_v2_20261119',
        'corrected_from_sha256': original_sha,
        'metric_mapping': {
            'boundary_miou': 'metrics.miou at region=boundary',
            'whole_miou': 'metrics.miou at region=whole',
            'forest_macro_f1': 'metrics.forest_macro_f1',
            'planted_f1': 'metrics.planted_f1',
        },
        'reason': 'The evaluator names the region-local mIoU as miou. The first analysis looked for boundary_miou and produced NA rows; this reanalysis only recomputes pairs/gates from unchanged raw results. Training/evaluation were not rerun.',
        'official_test_accessed': False,
    }
    data['status'] = 'COMPLETED_DEVELOPMENT_ONLY'
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': data['status'], 'pairs': len(rows), 'original_sha256': original_sha,
                      'corrected_sha256': sha(OUT), 'gates': gates}, ensure_ascii=False))


if __name__ == '__main__':
    main()
