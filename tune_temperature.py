"""Select a fixed temperature scale using 2015 validation scores only.

No evaluate.py calls, test predictions, or test metrics are used by this runner.
Every candidate retains the same number of RMS-normalized temperature fields.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def temperatures(base, multiplier):
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Temperature scale must be finite and positive')
    return [float(t) * multiplier for t in base]


def rank_trials(trials, groups, seeds, multipliers):
    expected = {(g, s) for g in groups for s in seeds}
    ranking = []
    for multiplier in multipliers:
        rows = [t for t in trials if t['multiplier'] == multiplier]
        if len(rows) != len(expected) or {(r['group'], r['seed']) for r in rows} != expected:
            raise ValueError('Refuse selection from an incomplete or duplicated trial grid')
        if not all(math.isfinite(r['validation_energy']) for r in rows):
            raise ValueError('Non-finite validation score')
        group_means = {str(g): float(np.mean([r['validation_energy'] for r in rows if r['group']==g]))
                       for g in groups}
        ranking.append({'multiplier': multiplier, 'mean_validation_energy': float(np.mean(list(group_means.values()))),
                        'per_group_validation_energy': group_means})
    # Stable order resolves exact ties in favour of the first listed candidate.
    return sorted(ranking, key=lambda x: x['mean_validation_energy'])


def reusable_trials(selection, base):
    """Index a complete earlier validation sweep; never import test results."""
    exp = selection['experiment']
    if exp['base_config'] != base or selection['test_evaluated_by_this_sweep'] is not False:
        raise ValueError('Earlier sweep has another base config or used test evaluation')
    trials = selection['trials']
    if len(trials) != len(exp['groups'])*len(exp['seeds'])*len(exp['multipliers']):
        raise ValueError('Earlier sweep does not contain exactly its complete trial grid')
    if selection['ranking'] != rank_trials(trials,exp['groups'],exp['seeds'],exp['multipliers']):
        raise ValueError('Earlier sweep validation ranking does not match its trials')
    for trial in trials:
        if trial['temperatures'] != temperatures(base['temperatures'],trial['multiplier']):
            raise ValueError('Earlier trial temperature values are inconsistent')
    return {(r['group'],r['seed'],r['multiplier']):r for r in trials}


def fit_record(work, cfg, seed, data_hash, *, method='riesz_softmax'):
    state = torch.load(work/'latest.pt', map_location='cpu', weights_only=False)
    manifest = json.loads((work/'manifest.json').read_text())
    for m in (manifest, state['manifest']):
        if (m['config'] != cfg or m['seed'] != seed or m['method'] != method
                or m['data_sha256'] != data_hash or m['smoke_only']):
            raise ValueError(f'Incompatible fit: {work}')
    complete = state['epoch'] >= cfg['epochs'] or state['bad'] >= cfg['patience']
    if not complete:
        return None
    best = torch.load(work/'best.pt', map_location='cpu', weights_only=False)
    if best['manifest'] != state['manifest'] or best['best'] != state['best']:
        raise ValueError(f'Best/latest checkpoint mismatch: {work}')
    if not math.isfinite(best['best']):
        raise ValueError('Non-finite validation result')
    return {'validation_energy': best['best'], 'best_epoch': best['epoch'],
            'stopped_epoch': state['epoch'], 'checkpoint': str(work/'best.pt'),
            'checkpoint_sha256': digest(work/'best.pt')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups', nargs='+', type=int, default=[0,1,2])
    p.add_argument('--seeds', nargs='+', type=int, default=[42,43,44])
    p.add_argument('--multipliers', nargs='+', type=float, default=[1., .25, 4., 16.])
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--config', default='configs/temperature10.yaml')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--source', default='data/temperature_data_cgm_std.feather')
    p.add_argument('--existing-runs', default='runs/temperature10_groups0_9')
    p.add_argument('--group0-run', default='runs/temperature10_group0')
    p.add_argument('--reuse-sweep', help='Earlier sweep directory or selection.json; reuse compatible frozen fits')
    p.add_argument('--workdir', default='runs/temperature_validation_sweep')
    a = p.parse_args()
    def absolute(path):
        path = Path(path)
        return path if path.is_absolute() else ROOT/path
    base = yaml.safe_load(absolute(a.config).read_text())
    if (base['stations'] != 10 or len(set(a.groups)) != len(a.groups)
            or not all(0 <= g < 100 for g in a.groups)
            or len(set(a.seeds)) != len(a.seeds)
            or not set(a.seeds).issubset(base['seeds'])
            or len(set(a.multipliers)) != len(a.multipliers)):
        p.error('Use distinct 10-station groups, seeds from the base config, and distinct multipliers')
    for multiplier in a.multipliers:
        temperatures(base['temperatures'], multiplier)
    previous, reuse_index, reuse_hash = None, {}, None
    if a.reuse_sweep:
        reuse_path = absolute(a.reuse_sweep)
        if reuse_path.is_dir(): reuse_path = reuse_path/'selection.json'
        previous = json.loads(reuse_path.read_text())
        reuse_index = reusable_trials(previous,base)
        reuse_hash = digest(reuse_path)
    root = absolute(a.workdir); root.mkdir(parents=True, exist_ok=True)
    data_dir = absolute(a.data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    spec = {'base_config': base, 'groups': a.groups, 'seeds': a.seeds, 'multipliers': a.multipliers,
            'selection': 'equal-group mean of per-seed best 2015 validation training-form energy score',
            'test_use': 'none in sweep; 2016 was already inspected in the prior baseline experiment',
            'field_count': len(base['temperatures'])}
    if reuse_hash is not None:
        spec['reuse_sweep_sha256'] = reuse_hash
    spec_path = root/'experiment.json'
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError('Sweep workdir belongs to another grid/config; use a new workdir')
    spec_path.write_text(json.dumps(spec, indent=2))
    trials, hashes = [], {}
    for group in a.groups:
        data = data_dir/f'temperature10_group{group}.npz'
        if not data.exists():
            subprocess.run([sys.executable, '-u', 'prepare_data.py', '--download', '--group', str(group),
                            '--source', str(absolute(a.source)), '--output', str(data)], cwd=ROOT, check=True)
        meta = json.loads(data.with_suffix('.json').read_text())
        if meta['station_group'] != group or len(meta['stations']) != 10:
            raise ValueError('Incorrect prepared station group')
        hashes[str(group)] = data_hash = digest(data)
        for multiplier in a.multipliers:
            label = f'scale_{multiplier:g}'.replace('.', 'p')
            cfg = dict(base, group=group, temperatures=temperatures(base['temperatures'], multiplier))
            cfg_path = root/f'group{group:02d}_{label}.yaml'
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            for seed in a.seeds:
                work = root/label/f'group{group:02d}'/f'seed{seed}'
                prior = reuse_index.get((group,seed,multiplier))
                if prior is not None:
                    if previous['data_hashes'][str(group)] != data_hash:
                        raise ValueError('Prepared data differs from the earlier sweep')
                    work = absolute(prior['checkpoint']).parent
                    if digest(work/'best.pt') != prior['checkpoint_sha256']:
                        raise ValueError('Earlier checkpoint changed after its sweep was frozen')
                elif multiplier == 1:
                    original = (absolute(a.group0_run) if group == 0 else
                                absolute(a.existing_runs)/f'group{group:02d}')/'riesz_softmax'/f'seed{seed}'
                    if (original/'latest.pt').exists():
                        work = original
                record = fit_record(work, cfg, seed, data_hash) if (work/'latest.pt').exists() else None
                if prior is not None:
                    if record is None or record['validation_energy'] != prior['validation_energy']:
                        raise ValueError('Earlier fit no longer matches the frozen validation trial')
                    print(f'Group {group}, {label}, seed {seed}: reusing earlier sweep fit',flush=True)
                if record is None:
                    work.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, '-u', 'train.py', '--data', str(data), '--config', str(cfg_path),
                               '--method', 'riesz_softmax', '--seed', str(seed), '--device', a.device,
                               '--workdir', str(work)]
                    if (work/'latest.pt').exists():
                        command.append('--resume')
                    print(f'Group {group}, {label}, seed {seed}: fitting (validation only)', flush=True)
                    with open(work/'sweep_runner.log', 'a') as f:
                        subprocess.run(command, cwd=ROOT, check=True, stdout=f, stderr=subprocess.STDOUT)
                    record = fit_record(work, cfg, seed, data_hash)
                    if record is None:
                        raise ValueError('Fit did not complete its training/early-stopping budget')
                trial = dict(record, group=group, seed=seed, multiplier=multiplier, temperatures=cfg['temperatures'])
                trials.append(trial)
                print(f'Group {group}, {label}, seed {seed}: val ES {record["validation_energy"]:.4f}', flush=True)
                tmp = root/'progress.tmp'; tmp.write_text(json.dumps({'experiment':spec,'trials':trials},indent=2))
                tmp.replace(root/'progress.json')
    ranking = rank_trials(trials, a.groups, a.seeds, a.multipliers)
    winner = ranking[0]['multiplier']
    selection = {'experiment': spec, 'data_hashes': hashes, 'ranking': ranking,
                 'selected_multiplier': winner, 'selected_temperatures': temperatures(base['temperatures'], winner),
                 'trials': trials, 'test_evaluated_by_this_sweep': False,
                 'selection_scope': 'three training seeds / three groups by default; exploratory validation tuning'}
    frozen = root/'selection.json'
    if frozen.exists() and json.loads(frozen.read_text()) != selection:
        raise ValueError('Frozen selection would change; use a new workdir and document the new search')
    frozen.write_text(json.dumps(selection, indent=2))
    (root/'selected_config.yaml').write_text(yaml.safe_dump(
        dict(base, temperatures=selection['selected_temperatures']), sort_keys=False))
    print('\nScale      Mean 2015 validation ES (lower is better)', flush=True)
    for row in ranking:
        print(f'{row["multiplier"]:<10g} {row["mean_validation_energy"]:.6f}', flush=True)
    print(f'Selected temperatures: {selection["selected_temperatures"]}', flush=True)
    print(f'Selection: {frozen}\nNo test evaluation was run.', flush=True)


if __name__ == '__main__':
    main()
