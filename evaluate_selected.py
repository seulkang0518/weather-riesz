"""Test a frozen validation-selected setting against the completed baselines.

Reuses selected sweep checkpoints, fits only missing seeds, and never reselects
temperatures using test scores. Original fits/results are read without modification.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from run_groups import ROOT, METRICS, aggregate, read_json, run, verify_evaluation
from tune_temperature import digest, fit_record, rank_trials, temperatures


def validate_selection(selection):
    exp = selection['experiment']
    if selection['test_evaluated_by_this_sweep'] is not False:
        raise ValueError('Selection must come from a validation-only sweep')
    if len(selection['trials']) != len(exp['groups'])*len(exp['seeds'])*len(exp['multipliers']):
        raise ValueError('Selection trial grid is incomplete or contains extra records')
    ranking = rank_trials(selection['trials'], exp['groups'], exp['seeds'], exp['multipliers'])
    winner = ranking[0]['multiplier']
    selected = temperatures(exp['base_config']['temperatures'], winner)
    if (selection['ranking'] != ranking or selection['selected_multiplier'] != winner
            or selection['selected_temperatures'] != selected):
        raise ValueError('Selection does not match its complete validation ranking')
    return exp['base_config'], winner, selected


def verify_baseline(root, cfg, data_hash, method):
    result = read_json(root/method/'evaluation/metrics.json')
    verify_evaluation(result, method, data_hash, cfg['seeds'])
    for seed in cfg['seeds']:
        manifest = read_json(root/method/f'seed{seed}'/'manifest.json')
        if (manifest['config'] != cfg or manifest['method'] != method
                or manifest['seed'] != seed or manifest['data_sha256'] != data_hash
                or manifest['smoke_only']):
            raise ValueError(f'Baseline config/data mismatch: {root}/{method}/seed{seed}')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--selection', default='runs/temperature_validation_sweep/selection.json')
    p.add_argument('--groups', nargs='+', type=int, default=list(range(10)))
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--source', default='data/temperature_data_cgm_std.feather')
    p.add_argument('--existing-runs', default='runs/temperature10_groups0_9')
    p.add_argument('--group0-run', default='runs/temperature10_group0')
    p.add_argument('--workdir', default='runs/temperature_selected_test')
    a = p.parse_args()
    def absolute(path):
        path = Path(path)
        return (path if path.is_absolute() else ROOT/path).resolve()
    if len(set(a.groups)) != len(a.groups) or not all(0<=g<100 for g in a.groups):
        p.error('Choose distinct official station group indices, 0-99')
    selection_path = absolute(a.selection)
    selection = read_json(selection_path)
    base, multiplier, chosen = validate_selection(selection)
    if (base['stations'] != 10 or len(base['seeds']) != 10 or len(set(base['seeds'])) != 10
            or base['eval_samples_per_model'] != 5):
        raise ValueError('Require the ten-station, ten-model, fifty-scenario protocol')
    root = absolute(a.workdir); root.mkdir(parents=True, exist_ok=True)
    selection_hash = digest(selection_path)
    spec = {'selection_sha256':selection_hash,'groups':a.groups,'base_config':base,
            'selected_temperatures':chosen}
    if (root/'experiment.json').exists() and read_json(root/'experiment.json') != spec:
        raise ValueError('Workdir has another frozen selection/config/group set; choose a new workdir')
    (root/'experiment.json').write_text(json.dumps(spec,indent=2))
    snapshot = root/'frozen_selection.json'
    if snapshot.exists() and snapshot.read_bytes() != selection_path.read_bytes():
        raise ValueError('Frozen selection snapshot differs')
    snapshot.write_bytes(selection_path.read_bytes())
    selected_trials = {(r['group'],r['seed']):r for r in selection['trials']
                       if r['multiplier']==multiplier}
    print(f'Frozen temperatures: {chosen}; no further selection will occur.',flush=True)
    rows = []
    for index, group in enumerate(a.groups):
        print(f'Group {group} ({index+1}/{len(a.groups)})',flush=True)
        data = absolute(a.data_dir)/f'temperature10_group{group}.npz'
        if not data.exists():
            run(['prepare_data.py','--download','--group',str(group),'--source',str(absolute(a.source)),
                 '--output',str(data)])
        data_hash = digest(data); meta = read_json(data.with_suffix('.json'))
        if meta['station_group'] != group or len(meta['stations']) != 10:
            raise ValueError('Prepared station group mismatch')
        if str(group) in selection['data_hashes'] and selection['data_hashes'][str(group)] != data_hash:
            raise ValueError('Prepared data differs from validation sweep')
        original = absolute(a.group0_run) if group==0 else absolute(a.existing_runs)/f'group{group:02d}'
        cfg_original = dict(base,group=group)
        energy = verify_baseline(original,cfg_original,data_hash,'energy')
        untuned = verify_baseline(original,cfg_original,data_hash,'riesz_softmax')
        cfg = dict(cfg_original,temperatures=chosen)
        group_root = root/f'group{group:02d}'; group_root.mkdir(parents=True,exist_ok=True)
        cfg_path = group_root/'config.yaml'; cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        checkpoints, provenance = [], []
        for seed in base['seeds']:
            trial = selected_trials.get((group,seed))
            if trial is not None:
                work = absolute(trial['checkpoint']).parent
                if digest(work/'best.pt') != trial['checkpoint_sha256']:
                    raise ValueError('A selected sweep checkpoint changed after validation selection')
                record = fit_record(work,cfg,seed,data_hash)
                if record is None or record['validation_energy'] != trial['validation_energy']:
                    raise ValueError('Selected sweep fit no longer matches frozen trial')
                origin = 'frozen_validation_sweep'
            elif multiplier == 1:
                work = original/'riesz_softmax'/f'seed{seed}'
                record = fit_record(work,cfg,seed,data_hash)
                if record is None:
                    raise ValueError('Original fit is not complete')
                origin = 'original_setting_unchanged'
            else:
                work = group_root/f'seed{seed}'
                record = fit_record(work,cfg,seed,data_hash) if (work/'latest.pt').exists() else None
                if record is None:
                    work.mkdir(parents=True,exist_ok=True)
                    command = ['train.py','--data',str(data),'--config',str(cfg_path),'--method','riesz_softmax',
                               '--seed',str(seed),'--device',a.device,'--workdir',str(work)]
                    if (work/'latest.pt').exists(): command.append('--resume')
                    print(f'  Selected Riesz seed {seed}: fitting',flush=True)
                    run(command,work/'selected_runner.log')
                    record = fit_record(work,cfg,seed,data_hash)
                    if record is None: raise ValueError('Selected fit did not finish')
                origin = 'additional_selected_setting_fit'
            checkpoint = str(work/'best.pt')
            checkpoints.append(checkpoint)
            provenance.append(dict(record,seed=seed,origin=origin))
            print(f'  Selected Riesz seed {seed}: ready ({origin})',flush=True)
        evaluation = group_root/'evaluation'
        provenance_file = group_root/'checkpoint_provenance.json'
        if provenance_file.exists() and read_json(provenance_file) != provenance:
            raise ValueError('Selected checkpoint identities changed after they were recorded')
        provenance_file.write_text(json.dumps(provenance,indent=2))
        if not (evaluation/'metrics.json').exists():
            run(['evaluate.py','--data',str(data),'--device',a.device,'--checkpoints',*checkpoints,
                 '--output',str(evaluation)])
        result = read_json(evaluation/'metrics.json')
        verify_evaluation(result,'riesz_softmax',data_hash,base['seeds'])
        # Refuse stale evaluations if a checkpoint was replaced after evaluation.
        identity = {r['seed']:(str(absolute(r['checkpoint'])),r['best_epoch'],r['validation_energy']) for r in provenance}
        for model in result['models']:
            if (model['checkpoint'],model['epoch'],model['val_energy_training_form']) != identity[model['seed']]:
                raise ValueError('Existing evaluation does not match selected checkpoints')
        for reference in (energy,untuned):
            if not np.allclose([reference['raw_ensemble_metrics'][m] for m in METRICS],
                               [result['raw_ensemble_metrics'][m] for m in METRICS],
                               rtol=1e-10,atol=1e-10):
                raise ValueError('Raw forecast scores differ; evaluation protocols do not match')
        rows.append({'group':group,'stations':meta['stations'],'counts':meta['counts'],'data_sha256':data_hash,
                     'energy':energy['metrics'],'original_riesz_softmax':untuned['metrics'],
                     'riesz_softmax':result['metrics'],'raw':result['raw_ensemble_metrics']})
        summary = aggregate(rows,a.groups)
        summary.update(selection_sha256=selection_hash,selected_temperatures=chosen,
                       note='Riesz scores use the frozen validation-selected setting. Previously inspected 2016 test year.')
        summary['group_mean_scores']['original_riesz_softmax'] = {
            metric:float(np.mean([r['original_riesz_softmax'][metric] for r in rows])) for metric in METRICS}
        temporary = root/'summary.tmp'; temporary.write_text(json.dumps(summary,indent=2))
        temporary.replace(root/'summary.json')
        print(f'  Energy {energy["metrics"]["energy_score"]:.4f}; selected Riesz {result["metrics"]["energy_score"]:.4f}',flush=True)
    print('\nMethod                  Energy      Variogram       CRPS',flush=True)
    for key,label in [('energy','energy'),('original_riesz_softmax','original_riesz'),
                      ('riesz_softmax','selected_riesz'),('raw','raw')]:
        score = summary['group_mean_scores'][key]
        print(f'{label:<22} {score["energy_score"]:9.4f} {score["variogram_score"]:13.4f} {score["mean_crps"]:10.4f}',flush=True)
    print(f'Summary: {root / "summary.json"}',flush=True)


if __name__ == '__main__':
    main()
