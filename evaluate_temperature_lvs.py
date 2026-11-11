"""Complete/test a frozen validation-LVS temperature choice; never reselect on test."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from compare_uniform import checked_reference
from evaluate_learning_rate import check_evaluation, validate_lr_selection
from evaluate_lvs import load_forecast
from evaluate_selected import validate_selection
from run_groups import ROOT, read_json, run
from train import validation_sample_count
from tune_temperature import digest, fit_record, temperatures
from tune_temperature_lvs import rank_candidates
from weather.local_variogram import (local_variogram, nearest_other_stations,
                                    recover_coordinates, seasonal_climatology)
from weather.metrics import scores


def validate_lvs_selection(selection, lr_selection):
    """Require the actual complete validation winner, with the inherited fixed LR."""
    if (selection.get('kind') != 'temperature_lvs'
            or selection.get('test_evaluated_by_this_sweep') is not False):
        raise ValueError('Require a frozen validation-only LVS temperature selection')
    exp = selection['experiment']
    if exp.get('train_particles',50)!=50:
        raise ValueError('This evaluator is for 50-particle sweeps; lower-count choices require particle-matched references')
    base, configs = validate_lr_selection(lr_selection)
    if exp['base_config'] != base or exp['fixed_learning_rate'] != configs['riesz_softmax']['learning_rate']:
        raise ValueError('LVS sweep changed the inherited base configuration or fixed LR')
    if exp['lvs'] != {'neighbors': 5, 'exclude_center': True, 'power': .5, 'window': 21}:
        raise ValueError('LVS sweep uses a different evaluation protocol')
    groups, seeds, multipliers = exp['groups'], exp['seeds'], exp['multipliers']
    if (not groups or len(set(groups)) != len(groups) or not all(0 <= g < 100 for g in groups)
            or not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(base['seeds'])
            or not multipliers or len(set(multipliers)) != len(multipliers)):
        raise ValueError('Invalid frozen sweep grid')
    for trial in selection['trials']:
        if trial['temperatures'] != temperatures(base['temperatures'], trial['multiplier']):
            raise ValueError('Frozen trial temperature mismatch')
    ranking = rank_candidates(selection['trials'], groups, seeds, multipliers)
    winner = ranking[0]['multiplier']
    chosen = temperatures(base['temperatures'], winner)
    if (selection['ranking'] != ranking or selection['selected_multiplier'] != winner
            or selection['selected_temperatures'] != chosen):
        raise ValueError('Frozen choice does not match the complete validation LVS ranking')
    cfg = dict(configs['riesz_softmax'], temperatures=chosen)
    return base, configs, cfg


def score_group(data, meta, paths, data_hash):
    """Use existing forecast integrity checks and identical LVS for all methods."""
    neighbors = nearest_other_stations(*recover_coordinates(data, meta))
    mean, std, _ = seasonal_climatology(data['train_target'], data['train_dates'], data['test_dates'])
    target = data['test_target'].astype(float)
    result, inputs, per_case = {}, {}, {'dates': data['test_dates'], 'climatology_mean': mean,
                                      'climatology_std': std, 'neighbor_indices': neighbors}
    for label, (path, method) in paths.items():
        pred, evaluation = load_forecast(path, data, data_hash, method)
        physical = local_variogram(pred, target, neighbors)
        anomaly = local_variogram((pred-mean[:, None])/std[:, None], (target-mean)/std, neighbors)
        result[label] = dict(evaluation['metrics'], local_variogram_anomaly=float(anomaly.mean()),
                             local_variogram_physical=float(physical.mean()))
        inputs[label] = {'directory': str(path), 'predictions_sha256': digest(path/'predictions.npz'),
                         'metrics_sha256': digest(path/'metrics.json')}
        per_case[label+'_lvs_anomaly'], per_case[label+'_lvs_physical'] = anomaly, physical
    raw = data['test_raw']
    raw_metrics = {k: v.mean().item() for k, v in scores(torch.from_numpy(raw), torch.from_numpy(target)).items()}
    physical = local_variogram(raw, target, neighbors)
    anomaly = local_variogram((raw-mean[:, None])/std[:, None], (target-mean)/std, neighbors)
    result['raw'] = dict(raw_metrics, local_variogram_anomaly=float(anomaly.mean()),
                         local_variogram_physical=float(physical.mean()))
    per_case['raw_lvs_anomaly'], per_case['raw_lvs_physical'] = anomaly, physical
    return result, inputs, per_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', default='runs/temperature_lvs_validation_sweep/selection.json')
    p.add_argument('--reference-runs', default='runs/temperature_learning_rate_selected_test')
    p.add_argument('--uniform-runs', default='runs/temperature_uniform_comparison')
    p.add_argument('--skip-uniform', action='store_true')
    p.add_argument('--groups', nargs='+', type=int, default=list(range(10)))
    p.add_argument('--data-dir', default='data')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--workdir', default='runs/temperature_lvs_selected_test')
    a = p.parse_args()
    def absolute(path):
        path = Path(path)
        return (path if path.is_absolute() else ROOT/path).resolve()
    if not a.groups or len(set(a.groups)) != len(a.groups) or not all(0 <= g < 100 for g in a.groups):
        p.error('Choose distinct official station groups, 0-99')
    torch.set_num_threads(2)
    selection_path = absolute(a.selection)
    selection = read_json(selection_path)
    lr_path = selection_path.parent/'frozen_lr_selection.json'
    if digest(lr_path) != selection['experiment']['lr_selection_sha256']:
        raise ValueError('Underlying LR selection changed')
    lr_selection = read_json(lr_path)
    base, configs, selected_cfg = validate_lvs_selection(selection, lr_selection)
    temp_path = selection_path.parent/'frozen_temperature_selection.json'
    if digest(temp_path) != lr_selection['experiment']['temperature_selection_sha256']:
        raise ValueError('Underlying ES-temperature selection changed')
    temp_base, _, chosen = validate_selection(read_json(temp_path))
    if temp_base != base or chosen != configs['riesz_softmax']['temperatures']:
        raise ValueError('Inherited temperature provenance mismatch')
    if (base['stations'] != 10 or len(base['seeds']) != 10 or len(set(base['seeds'])) != 10
            or base['eval_samples_per_model'] != 5 or selected_cfg['train_samples'] != 50
            or validation_sample_count(selected_cfg) != 50):
        raise ValueError('Require ten stations, ten models and fifty training/evaluation scenarios')
    root = absolute(a.workdir)
    spec = {'selection_sha256': digest(selection_path), 'lr_selection_sha256': digest(lr_path),
            'selected_config': selected_cfg, 'groups': a.groups,
            'reference_runs': str(absolute(a.reference_runs)),
            'uniform_runs': None if a.skip_uniform else str(absolute(a.uniform_runs)),
            'data_dir': str(absolute(a.data_dir)),
            'selection_policy': 'frozen validation LVS winner; no test reselection'}
    if (root/'experiment.json').exists() and read_json(root/'experiment.json') != spec:
        raise ValueError('Workdir belongs to a different frozen experiment')
    # Check baseline input availability before fitting any additional models.
    for group in a.groups:
        paths = [absolute(a.data_dir)/f'temperature10_group{group}.npz',
                 absolute(a.data_dir)/f'temperature10_group{group}.json']
        refs = [absolute(a.reference_runs)/f'group{group:02d}/{m}/evaluation' for m in ('energy','riesz_softmax')]
        if not a.skip_uniform: refs.append(absolute(a.uniform_runs)/f'group{group:02d}/evaluation')
        paths += [d/f for d in refs for f in ('metrics.json', 'predictions.npz')]
        for path in paths:
            if not path.is_file(): raise FileNotFoundError(f'Missing prepared data/reference forecast: {path}')
    root.mkdir(parents=True, exist_ok=True)
    (root/'experiment.json').write_text(json.dumps(spec, indent=2))
    for source, name in ((selection_path,'frozen_lvs_selection.json'), (lr_path,'frozen_lr_selection.json'),
                         (temp_path,'frozen_temperature_selection.json')):
        saved = root/name
        if saved.exists() and saved.read_bytes() != source.read_bytes(): raise ValueError('Frozen snapshot changed')
        saved.write_bytes(source.read_bytes())
    selected_trials = {(r['group'], r['seed']): r for r in selection['trials']
                       if r['multiplier']==selection['selected_multiplier']}
    rows = []
    print(f'Frozen Riesz temperatures: {selected_cfg["temperatures"]}; LR {selected_cfg["learning_rate"]:g}.', flush=True)
    for group in a.groups:
        data_path = absolute(a.data_dir)/f'temperature10_group{group}.npz'
        data_hash = digest(data_path); meta = read_json(data_path.with_suffix('.json'))
        if meta['station_group'] != group or len(meta['stations']) != 10: raise ValueError('Data group mismatch')
        if str(group) in selection['data_hashes'] and selection['data_hashes'][str(group)] != data_hash:
            raise ValueError('Prepared data differs from frozen sweep')
        cfg = dict(selected_cfg, group=group)
        ref_root = absolute(a.reference_runs)/f'group{group:02d}'
        paths = {'energy': (ref_root/'energy/evaluation','energy'),
                 'riesz_before': (ref_root/'riesz_softmax/evaluation','riesz_softmax')}
        reference_provenance = {}
        for label, method in (('energy','energy'), ('riesz_before','riesz_softmax')):
            _, reference_provenance[label] = checked_reference(paths[label][0]/'metrics.json',
                                                              dict(configs[method], group=group), data_hash, method)
        if not a.skip_uniform:
            paths['riesz_uniform'] = (absolute(a.uniform_runs)/f'group{group:02d}/evaluation','riesz_uniform')
            _, reference_provenance['riesz_uniform'] = checked_reference(paths['riesz_uniform'][0]/'metrics.json',
                dict(configs['riesz_softmax'], group=group), data_hash, 'riesz_uniform')
        group_root = root/f'group{group:02d}'
        model_root = group_root/'riesz_softmax'; model_root.mkdir(parents=True, exist_ok=True)
        config_path = model_root/'config.yaml'; config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        provenance, checkpoints = [], []
        old_by_seed = {r['seed']: r for r in reference_provenance['riesz_before']}
        for seed in base['seeds']:
            trial = selected_trials.get((group, seed))
            if trial is not None:
                work = absolute(trial['checkpoint']).parent
                if digest(work/'best.pt') != trial['checkpoint_sha256']: raise ValueError('Selected sweep checkpoint changed')
                record = fit_record(work, cfg, seed, data_hash)
                if record is None or record['validation_energy'] != trial['validation_energy']:
                    raise ValueError('Selected sweep checkpoint is incomplete or inconsistent')
                origin = 'frozen_lvs_sweep'
            elif selected_cfg == configs['riesz_softmax']:
                work = absolute(old_by_seed[seed]['checkpoint']).parent
                record = fit_record(work, cfg, seed, data_hash)
                if record is None: raise ValueError('Reference fit incomplete')
                origin = 'unchanged_reference_choice'
            else:
                work = model_root/f'seed{seed}'
                record = fit_record(work, cfg, seed, data_hash) if (work/'latest.pt').exists() else None
                if record is None:
                    work.mkdir(parents=True, exist_ok=True)
                    cmd = ['train.py','--data',str(data_path),'--config',str(config_path),'--method','riesz_softmax',
                           '--seed',str(seed),'--device',a.device,'--workdir',str(work)]
                    if (work/'latest.pt').exists(): cmd.append('--resume')
                    print(f'Group {group}, selected Riesz seed {seed}: training', flush=True)
                    run(cmd, work/'selected_lvs_runner.log')
                    record = fit_record(work, cfg, seed, data_hash)
                    if record is None: raise ValueError('Fit did not finish')
                origin = 'additional_selected_fit'
            provenance.append(dict(record, seed=seed, origin=origin)); checkpoints.append(str(work/'best.pt'))
            print(f'Group {group}, selected Riesz seed {seed}: ready ({origin})', flush=True)
        pf = model_root/'checkpoint_provenance.json'
        if pf.exists() and read_json(pf) != provenance: raise ValueError('Selected checkpoint identities changed')
        pf.write_text(json.dumps(provenance, indent=2))
        evaluation = model_root/'evaluation'
        if not (evaluation/'metrics.json').exists():
            run(['evaluate.py','--data',str(data_path),'--device',a.device,'--checkpoints',*checkpoints,'--output',str(evaluation)])
        check_evaluation(read_json(evaluation/'metrics.json'), 'riesz_softmax', data_hash, base['seeds'], provenance)
        paths['riesz_selected'] = (evaluation,'riesz_softmax')
        with np.load(data_path, allow_pickle=False) as data:
            result, inputs, arrays = score_group(data, meta, paths, data_hash)
        row = {'group': group, 'stations': meta['stations'], 'data_sha256': data_hash,
               'metrics': result, 'inputs': inputs, 'reference_checkpoints': reference_provenance}
        old_result = group_root/'comparison.json'
        if old_result.exists() and read_json(old_result) != row: raise ValueError('Saved comparison identity changed')
        if not old_result.exists():
            np.savez_compressed(group_root/'per_case_lvs.npz', **arrays)
            old_result.write_text(json.dumps(row, indent=2))
        rows.append(row)
        methods = list(result)
        means = {m: {k: float(np.mean([r['metrics'][m][k] for r in rows])) for k in result[m]} for m in methods}
        summary = {'complete': len(rows)==len(a.groups), 'requested_groups': a.groups, 'groups': rows,
                   'group_mean_scores': means, 'selected_temperatures': selected_cfg['temperatures'],
                   'previous_temperatures': configs['riesz_softmax']['temperatures'],
                   'learning_rates': {m: c['learning_rate'] for m,c in configs.items()},
                   'selection_sha256': spec['selection_sha256'],
                   'models_per_method_per_group': 10, 'scenarios_per_case': 50,
                   'lvs_protocol': selection['experiment']['lvs'],
                   'paired_group_differences_vs_energy': {m: {k: [r['metrics'][m][k]-r['metrics']['energy'][k] for r in rows]
                                                             for k in result[m]} for m in methods if m!='energy'},
                   'note': 'Frozen validation-LVS winner; no test-based reselection. Same ten-model x five-sample protocol. '
                           'Training and checkpoint selection unchanged. Repeated/exploratory use of 2016 test year; '
                           'overlapping groups are not independent replicates. No significance claim.'}
        temporary = root/'summary.tmp'; temporary.write_text(json.dumps(summary, indent=2)); temporary.replace(root/'summary.json')
        print(f'Group {group}: energy LVS {result["energy"]["local_variogram_anomaly"]:.6f}; '
              f'previous Riesz {result["riesz_before"]["local_variogram_anomaly"]:.6f}; '
              f'selected Riesz {result["riesz_selected"]["local_variogram_anomaly"]:.6f}', flush=True)
    print('\nMethod                  ES        VS      CRPS   LVS anomaly   LVS physical')
    for method in ('energy','riesz_before','riesz_selected','riesz_uniform','raw'):
        if method not in means: continue
        r = means[method]
        print(f'{method:<20} {r["energy_score"]:8.4f} {r["variogram_score"]:9.4f} {r["mean_crps"]:9.4f} '
              f'{r["local_variogram_anomaly"]:13.6f} {r["local_variogram_physical"]:14.6f}')
    print(f'Summary: {root / "summary.json"}')


if __name__ == '__main__':
    main()
