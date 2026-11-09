"""Validation-only temperature comparison with LVS and repulsion-weight diagnostics.

Training stays Riesz softmax, with the existing ES-based checkpoint selection.
Candidate temperatures are ranked by held-out 2015 anomaly LVS; ES is reported
alongside it. This runner does not load 2016 targets, forecasts, or test scores.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from evaluate_learning_rate import validate_lr_selection
from evaluate_selected import validate_selection
from run_groups import ROOT, read_json, run
from train import load_split, predict, validation_sample_count
from tune_temperature import digest, fit_record, temperatures
from weather.local_variogram import (local_variogram, nearest_other_stations,
                                    recover_coordinates, seasonal_climatology)
from weather.metrics import scores
from weather.model import TemperatureCGM
from weather.objectives import training_energy_score


@torch.no_grad()
def repulsion_statistics(pred, target, taus):
    """Match the upstream loss's detached batch scaling and softmax exactly.

    ESS measures weights before unit-direction averaging and RMS normalization.
    It is not a count of independent weather outcomes.
    """
    gen, pos = pred.detach().float(), target.detach().float()[:, None]
    scale = torch.cdist(gen, torch.cat([gen, pos], 1)).mean() / (1+1e-6)
    coordinate_scale = (scale / math.sqrt(gen.shape[-1])).clamp_min(1e-3)
    distances = torch.cdist(gen/coordinate_scale, gen/coordinate_scale)
    diagonal = torch.eye(gen.shape[1], dtype=torch.bool, device=gen.device)[None]
    result = []
    for tau in taus:
        weight = (-distances/(math.sqrt(gen.shape[-1])*tau)).masked_fill(diagonal, -torch.inf).softmax(-1)
        ess = weight.square().sum(-1).reciprocal()
        result.append({'temperature': tau, 'effective_neighbors': ess.mean().item(),
                       'fraction_of_uniform_ess': (ess/(gen.shape[1]-1)).mean().item(),
                       'maximum_neighbor_weight': weight.max(-1).values.mean().item()})
    return result


def rank_candidates(trials, groups, seeds, multipliers):
    expected = {(g, s, m) for g in groups for s in seeds for m in multipliers}
    actual = [(r['group'], r['seed'], r['multiplier']) for r in trials]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Require the full, nonduplicated validation grid')
    if any(not math.isfinite(r[key]) for r in trials
           for key in ('validation_lvs_anomaly', 'validation_energy_empirical')):
        raise ValueError('Nonfinite validation scores')
    result = []
    for multiplier in multipliers:
        selected = [r for r in trials if r['multiplier'] == multiplier]
        per_group = {str(g): float(np.mean([r['validation_lvs_anomaly'] for r in selected if r['group']==g]))
                     for g in groups}
        weight_means = [{key: float(np.mean([r['repulsion_weights'][i][key] for r in selected]))
                         for key in ('temperature', 'effective_neighbors', 'fraction_of_uniform_ess',
                                     'maximum_neighbor_weight')}
                        for i in range(len(selected[0]['repulsion_weights']))]
        result.append({'multiplier': multiplier, 'mean_validation_lvs_anomaly': float(np.mean(list(per_group.values()))),
                       'per_group_validation_lvs_anomaly': per_group,
                       'mean_validation_energy_empirical': float(np.mean([r['validation_energy_empirical'] for r in selected])),
                       'repulsion_weights': weight_means})
    return sorted(result, key=lambda r: r['mean_validation_lvs_anomaly'])


def score_validation(checkpoint, data, cfg, device):
    """Fixed-noise per-seed predictions, not the final ten-model forecast ensemble."""
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = TemperatureCGM(cfg['stations'], cfg['latent_dim']).to(device)
    model.load_state_dict(state['model']); model.eval()
    inputs = load_split(data, 'val', device)
    generator = torch.Generator(device=device).manual_seed(cfg['validation_seed'])
    mean, std, _ = seasonal_climatology(data['train_target'], data['train_dates'], data['val_dates'])
    neighbors = nearest_other_stations(*recover_coordinates(data, state['manifest']['data_metadata']))
    predictions, weight_sums, training_es = [], None, 0.
    samples = validation_sample_count(cfg)
    with torch.no_grad():
        for idx in torch.arange(len(inputs['target']), device=device).split(cfg['batch_size']):
            pred = predict(model, inputs, idx, samples, generator)
            target = inputs['target'][idx]
            predictions.append(pred.cpu().numpy())
            training_es += training_energy_score(pred, target).sum().item()
            stats = repulsion_statistics(pred, target, cfg['temperatures'])
            if weight_sums is None:
                weight_sums = [dict(s, effective_neighbors=0., fraction_of_uniform_ess=0., maximum_neighbor_weight=0.) for s in stats]
            for accumulated, stat in zip(weight_sums, stats):
                for key in ('effective_neighbors', 'fraction_of_uniform_ess', 'maximum_neighbor_weight'):
                    accumulated[key] += stat[key]*len(idx)
    if not np.isclose(training_es/len(inputs['target']), state['best'], rtol=2e-5, atol=2e-6):
        raise ValueError('Fixed-noise validation no longer reproduces selected checkpoint score')
    pred = np.concatenate(predictions)
    target = data['val_target'].astype(float)
    lvs = local_variogram((pred-mean[:, None])/std[:, None], (target-mean)/std, neighbors)
    empirical = scores(torch.from_numpy(pred), torch.from_numpy(target))
    for stat in weight_sums:
        for key in ('effective_neighbors', 'fraction_of_uniform_ess', 'maximum_neighbor_weight'):
            stat[key] /= len(target)
    return {'validation_lvs_anomaly': float(lvs.mean()),
            'validation_energy_empirical': empirical['energy_score'].mean().item(),
            'validation_crps': empirical['mean_crps'].mean().item(),
            'repulsion_weights': weight_sums, 'validation_cases': len(target), 'validation_samples': samples}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', default='runs/temperature_learning_rate_sweep/lr_selection.json')
    p.add_argument('--groups', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--multipliers', nargs='+', type=float, default=[1, 4, 16, 64, 512])
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--workdir', default='runs/temperature_lvs_validation_sweep')
    a = p.parse_args()
    def absolute(path):
        path = Path(path)
        return (path if path.is_absolute() else ROOT/path).resolve()
    selection_path = absolute(a.selection)
    selection = read_json(selection_path)
    base, configs = validate_lr_selection(selection)
    inherited = configs['riesz_softmax']
    temperature_path = selection_path.parent/'frozen_temperature_selection.json'
    if digest(temperature_path) != selection['experiment']['temperature_selection_sha256']:
        raise ValueError('Underlying frozen temperature selection changed')
    temp_base, _, chosen = validate_selection(read_json(temperature_path))
    if temp_base != base or chosen != inherited['temperatures']:
        raise ValueError('Underlying temperature provenance mismatch')
    if (base['stations'] != 10 or not a.groups or len(set(a.groups)) != len(a.groups)
            or not all(0 <= g < 100 for g in a.groups) or not a.seeds
            or len(set(a.seeds)) != len(a.seeds) or not set(a.seeds).issubset(base['seeds'])
            or not a.multipliers or len(set(a.multipliers)) != len(a.multipliers)):
        p.error('Require distinct valid groups, seeds, and multipliers')
    for multiplier in a.multipliers:
        temperatures(base['temperatures'], multiplier)
    if len({f'{m:.17g}' for m in a.multipliers}) != len(a.multipliers):
        p.error('Multiplier labels must be distinct')
    torch.set_num_threads(2)
    root = absolute(a.workdir)
    spec = {'kind': 'temperature_lvs', 'lr_selection_sha256': digest(selection_path),
            'base_config': base, 'fixed_learning_rate': inherited['learning_rate'],
            'groups': a.groups, 'seeds': a.seeds, 'multipliers': a.multipliers,
            'checkpoint_selection': 'unchanged per-seed fixed-noise validation energy score',
            'candidate_selection': 'equal-group mean of per-seed 2015 anomaly LVS, 50 samples',
            'lvs': {'neighbors': 5, 'exclude_center': True, 'power': .5, 'window': 21},
            'test_use': 'none; 2016 was previously inspected in earlier experiments'}
    if (root/'experiment.json').exists() and read_json(root/'experiment.json') != spec:
        raise ValueError('Workdir belongs to another frozen setup; choose a new workdir')
    root.mkdir(parents=True, exist_ok=True)
    (root/'experiment.json').write_text(json.dumps(spec, indent=2))
    for source, name in ((selection_path, 'frozen_lr_selection.json'), (temperature_path, 'frozen_temperature_selection.json')):
        saved = root/name
        if saved.exists() and saved.read_bytes() != source.read_bytes():
            raise ValueError('Frozen selection snapshot differs')
        saved.write_bytes(source.read_bytes())
    prior = {(r['group'], r['seed']): r for r in selection['trials']
             if r['method']=='riesz_softmax' and r['learning_rate']==inherited['learning_rate']}
    trials, hashes = [], {}
    for group in a.groups:
        path = absolute(a.data_dir)/f'temperature10_group{group}.npz'
        data_hash = digest(path); hashes[str(group)] = data_hash
        if str(group) in selection['data_hashes'] and selection['data_hashes'][str(group)] != data_hash:
            raise ValueError('Prepared data differs from frozen LR sweep')
        with np.load(path, allow_pickle=False) as data:
            meta = read_json(path.with_suffix('.json'))
            if meta['station_group'] != group or len(meta['stations']) != 10:
                raise ValueError('Prepared station group mismatch')
            for multiplier in a.multipliers:
                cfg = dict(inherited, group=group, temperatures=temperatures(base['temperatures'], multiplier))
                if cfg['train_samples'] != 50 or validation_sample_count(cfg) != 50:
                    raise ValueError('This temperature-only comparison requires 50 train/validation particles')
                label = f'scale_{multiplier:.17g}'.replace('.', 'p')
                config_path = root/f'group{group:02d}_{label}.yaml'
                config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                for seed in a.seeds:
                    work = root/label/f'group{group:02d}'/f'seed{seed}'
                    old = prior.get((group, seed)) if cfg['temperatures']==inherited['temperatures'] else None
                    if old:
                        work = absolute(old['checkpoint']).parent
                        if digest(work/'best.pt') != old['checkpoint_sha256']:
                            raise ValueError('Frozen reference checkpoint changed')
                    record = fit_record(work, cfg, seed, data_hash) if (work/'latest.pt').exists() else None
                    if old and (record is None or record['validation_energy'] != old['validation_energy']):
                        raise ValueError('Frozen reference is incomplete or changed')
                    if record is None:
                        work.mkdir(parents=True, exist_ok=True)
                        cmd = ['train.py', '--data', str(path), '--config', str(config_path), '--method', 'riesz_softmax',
                               '--seed', str(seed), '--device', a.device, '--workdir', str(work)]
                        if (work/'latest.pt').exists(): cmd.append('--resume')
                        print(f'Group {group}, scale {multiplier:g}, seed {seed}: training Riesz softmax', flush=True)
                        run(cmd, work/'lvs_sweep_runner.log')
                        record = fit_record(work, cfg, seed, data_hash)
                        if record is None: raise ValueError('Fit did not complete')
                    cache = root/label/f'group{group:02d}'/f'validation_seed{seed}.json'
                    identity = {'checkpoint_sha256': record['checkpoint_sha256'], 'data_sha256': data_hash,
                                'config': cfg, 'seed': seed, 'protocol': spec['lvs'], 'device': a.device}
                    if cache.exists():
                        cached = read_json(cache)
                        if cached['identity'] != identity: raise ValueError('Validation cache identity changed')
                        scored = cached['scores']
                    else:
                        scored = score_validation(work/'best.pt', data, cfg, a.device)
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        cache.write_text(json.dumps({'identity': identity, 'scores': scored}, indent=2))
                    trials.append(dict(record, **scored, group=group, seed=seed, multiplier=multiplier,
                                       temperatures=cfg['temperatures'], reused_frozen_fit=bool(old)))
                    print(f'Group {group}, scale {multiplier:g}, seed {seed}: val LVS {scored["validation_lvs_anomaly"]:.6f}; '
                          f'ES {scored["validation_energy_empirical"]:.6f}; '
                          f'ESS {[round(r["effective_neighbors"], 2) for r in scored["repulsion_weights"]]} / 49', flush=True)
                    (root/'progress.json').write_text(json.dumps({'experiment': spec, 'trials': trials}, indent=2))
    ranking = rank_candidates(trials, a.groups, a.seeds, a.multipliers)
    winner = ranking[0]['multiplier']
    result = {'kind': 'temperature_lvs', 'experiment': spec, 'data_hashes': hashes, 'ranking': ranking,
              'selected_multiplier': winner, 'selected_temperatures': temperatures(base['temperatures'], winner),
              'trials': trials, 'test_evaluated_by_this_sweep': False,
              'scope': 'Exploratory validation-LVS tuning; different selection criterion from previous ES sweep. '
                       'Per-seed fifty-member validation scores are not the final ten-model ensemble protocol.'}
    frozen = root/'selection.json'
    if frozen.exists() and read_json(frozen) != result: raise ValueError('Frozen selection would change')
    frozen.write_text(json.dumps(result, indent=2))
    (root/'selected_config.yaml').write_text(yaml.safe_dump(dict(inherited, temperatures=result['selected_temperatures']), sort_keys=False))
    print('\nScale      Val anomaly LVS    Val ES        Effective neighbors per temperature / 49')
    for row in ranking:
        print(f'{row["multiplier"]:<10g} {row["mean_validation_lvs_anomaly"]:.6f}           '
              f'{row["mean_validation_energy_empirical"]:.6f}      '
              f'{[round(r["effective_neighbors"], 2) for r in row["repulsion_weights"]]}')
    print(f'Selected temperatures: {result["selected_temperatures"]}\nSelection: {frozen}\nNo test evaluation was run.')


if __name__ == '__main__':
    main()
