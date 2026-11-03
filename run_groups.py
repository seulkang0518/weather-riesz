"""Paired station-group comparison, with resumable fits and explicit provenance checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
METHODS = ('energy', 'riesz_softmax')
METRICS = ('energy_score', 'variogram_score', 'mean_crps')


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_evaluation(result, method, data_hash, seeds):
    if (result['method'] != method or result['data_sha256'] != data_hash
            or not result['paper_sampling_protocol'] or result['scenarios'] != 50
            or len(result['models']) != 10
            or {m['seed'] for m in result['models']} != set(seeds)
            or any(m['smoke_only'] for m in result['models'])):
        raise ValueError('Existing evaluation does not match this experiment')
    if not all(np.isfinite(result['metrics'][key]) for key in METRICS):
        raise ValueError('Non-finite evaluation scores')


def aggregate(rows, requested):
    """Equal-weight group means, as in paper; groups can overlap geographically."""
    means = {method: {metric: float(np.mean([row[method][metric] for row in rows]))
                      for metric in METRICS} for method in (*METHODS, 'raw')}
    differences = {metric: [row['riesz_softmax'][metric]-row['energy'][metric] for row in rows]
                   for metric in METRICS}
    return {'requested_groups': requested, 'completed_groups': [row['group'] for row in rows],
            'complete': len(rows) == len(requested), 'station_dimension': 10,
            'models_per_method_per_group': 10, 'final_scenarios_per_case': 50,
            'group_mean_scores': means,
            'riesz_minus_energy': {
                metric: {'mean': float(np.mean(delta)),
                         'std_across_groups': float(np.std(delta, ddof=1)) if len(delta)>1 else None,
                         'riesz_better_groups': int(np.sum(np.array(delta) < 0)),
                         'per_group': delta} for metric, delta in differences.items()},
            'groups': rows,
            'interpretation': 'Lower scores are better; negative paired differences favour Riesz. '
                              'Station groups overlap; descriptive spread is not an independent-sample significance test.'}


def run(command, log=None):
    if log is None:
        subprocess.run([sys.executable, '-u', *command], cwd=ROOT, check=True)
    else:
        with open(log, 'a') as f:
            subprocess.run([sys.executable, '-u', *command], cwd=ROOT, check=True,
                           stdout=f, stderr=subprocess.STDOUT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups', nargs='+', type=int, default=list(range(10)))
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--config', default='configs/temperature10.yaml')
    p.add_argument('--source', default='data/temperature_data_cgm_std.feather')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--workdir', default='runs/temperature10_groups0_9')
    p.add_argument('--group0-run', default='runs/temperature10_group0',
                   help='Reuse the initial group-0 fits, if present')
    a = p.parse_args()
    if len(set(a.groups)) != len(a.groups) or not all(0 <= g < 100 for g in a.groups):
        p.error('Choose distinct official station-group indices from 0 through 99')
    def absolute(path):
        path = Path(path)
        return path if path.is_absolute() else ROOT/path
    cfg = yaml.safe_load(absolute(a.config).read_text())
    if cfg['stations'] != 10 or len(cfg['seeds']) != 10 or len(set(cfg['seeds'])) != 10:
        p.error('This runner requires 10 stations and 10 distinct seeds')
    if cfg['eval_samples_per_model'] != 5:
        p.error('Paper evaluation requires five scenarios per model')
    root, data_dir = absolute(a.workdir), absolute(a.data_dir)
    root.mkdir(parents=True, exist_ok=True); data_dir.mkdir(parents=True, exist_ok=True)
    spec = {'groups': a.groups, 'config': cfg}
    spec_path = root/'experiment.json'
    if spec_path.exists() and read_json(spec_path) != spec:
        raise ValueError('Workdir belongs to a different group list/config; choose a new workdir')
    spec_path.write_text(json.dumps(spec, indent=2))
    rows = []
    for index, group in enumerate(a.groups):
        print(f'Group {group} ({index+1}/{len(a.groups)})', flush=True)
        prepared = data_dir/f'temperature10_group{group}.npz'
        if not prepared.exists():
            run(['prepare_data.py', '--source', str(absolute(a.source)), '--download',
                 '--group', str(group), '--stations', '10', '--output', str(prepared)])
        meta = read_json(prepared.with_suffix('.json'))
        if meta['station_group'] != group or len(meta['stations']) != 10:
            raise ValueError('Prepared dataset station group does not match')
        data_hash = hashlib.sha256(prepared.read_bytes()).hexdigest()
        group_cfg = dict(cfg, group=group)
        cfg_path = root/f'group{group:02d}.yaml'
        cfg_path.write_text(yaml.safe_dump(group_cfg, sort_keys=False))
        group_root = root/f'group{group:02d}'
        initial = absolute(a.group0_run)
        if group == 0 and initial.exists():
            group_root = initial
            print(f'  Reusing initial group 0 at {initial}', flush=True)
        group_root.mkdir(parents=True, exist_ok=True)
        row = {'group': group, 'stations': meta['stations'], 'counts': meta['counts'],
               'data_sha256': data_hash, 'run_directory': str(group_root)}
        for method in METHODS:
            checkpoints = []
            evaluation = group_root/method/'evaluation'
            result_file = evaluation/'metrics.json'
            for seed in cfg['seeds']:
                work = group_root/method/f'seed{seed}'
                manifest_file = work/'manifest.json'
                if manifest_file.exists():
                    manifest = read_json(manifest_file)
                    if (manifest['config'] != group_cfg or manifest['method'] != method
                            or manifest['seed'] != seed or manifest['data_sha256'] != data_hash
                            or manifest['smoke_only']):
                        raise ValueError(f'Incompatible existing fit: {work}')
                elif result_file.exists():
                    raise ValueError(f'Evaluation exists but model manifest is missing: {work}')
                if not result_file.exists():
                    work.mkdir(parents=True, exist_ok=True)
                    command = ['train.py', '--data', str(prepared), '--config', str(cfg_path),
                               '--method', method, '--seed', str(seed), '--device', a.device,
                               '--workdir', str(work)]
                    if (work/'latest.pt').exists():
                        command.append('--resume')
                    print(f'  {method}: seed {seed}', flush=True)
                    run(command, work/'runner.log')
                checkpoints.append(str(work/'best.pt'))
            if not result_file.exists():
                run(['evaluate.py', '--data', str(prepared), '--device', a.device,
                     '--checkpoints', *checkpoints, '--output', str(evaluation)])
            result = read_json(result_file)
            verify_evaluation(result, method, data_hash, cfg['seeds'])
            row[method] = result['metrics']
            if 'raw' in row and row['raw'] != result['raw_ensemble_metrics']:
                raise ValueError('Raw forecast scores differ between methods')
            row['raw'] = result['raw_ensemble_metrics']
            print(f'  {method} test ES: {result["metrics"]["energy_score"]:.4f}', flush=True)
        rows.append(row)
        summary = aggregate(rows, a.groups)
        temporary = root/'summary.tmp'
        temporary.write_text(json.dumps(summary, indent=2)); temporary.replace(root/'summary.json')
    print('\nMethod             Energy      Variogram       CRPS', flush=True)
    for method, values in summary['group_mean_scores'].items():
        print(f'{method:<17} {values["energy_score"]:9.4f} {values["variogram_score"]:13.4f} '
              f'{values["mean_crps"]:10.4f}', flush=True)
    print(f'Summary: {root / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
