"""Score saved forecasts only: no fitting, sampling, or checkpoint modification."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from weather.local_variogram import (local_variogram, nearest_other_stations,
                                    recover_coordinates, seasonal_climatology)
from weather.metrics import scores

ROOT = Path(__file__).resolve().parent
SOURCE_REVISION = '5396f9f5ecb673394b532c80a3e4922d842d1c78'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_forecast(directory, data, data_hash, method):
    """Validate saved case/station alignment and reproduce existing score values."""
    directory = Path(directory)
    metadata = json.loads((directory/'metrics.json').read_text())
    if (metadata['data_sha256'] != data_hash or metadata['method'] != method
            or not metadata['paper_sampling_protocol'] or metadata['scenarios'] != 50
            or len(metadata['models']) != 10
            or len({m['seed'] for m in metadata['models']}) != 10
            or any(m['smoke_only'] for m in metadata['models'])):
        raise ValueError(f'Incompatible evaluation: {directory}')
    with np.load(directory/'predictions.npz', allow_pickle=False) as saved:
        if (not np.array_equal(saved['target'], data['test_target'])
                or not np.array_equal(saved['dates'].astype('datetime64[D]'),
                                      data['test_dates'].astype('datetime64[D]'))):
            raise ValueError(f'Saved target/date alignment mismatch: {directory}')
        pred = saved['predictions'].copy()
    expected = (len(data['test_target']), 50, data['test_target'].shape[1])
    if pred.shape != expected or not np.isfinite(pred).all():
        raise ValueError(f'Invalid saved ensemble shape/values: {directory}')
    actual = scores(torch.from_numpy(pred), torch.from_numpy(data['test_target']))
    for key, value in actual.items():
        if not np.isclose(value.mean().item(), metadata['metrics'][key], rtol=1e-8, atol=1e-9):
            raise ValueError(f'Saved {key} no longer reproduces: {directory}')
    return pred, metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--groups', nargs='+', type=int, default=list(range(10)))
    p.add_argument('--data-dir', default='data')
    p.add_argument('--energy-runs', default='runs/temperature_learning_rate_selected_test')
    p.add_argument('--riesz-runs', default='runs/temperature_learning_rate_selected_test')
    p.add_argument('--uniform-runs', default='runs/temperature_uniform_comparison')
    p.add_argument('--skip-uniform', action='store_true')
    p.add_argument('--neighbors', type=int, default=5)
    p.add_argument('--window', type=int, default=21)
    p.add_argument('--output', default='runs/temperature_lvs_comparison')
    a = p.parse_args()
    if not a.groups or len(set(a.groups)) != len(a.groups) or not all(0 <= g < 100 for g in a.groups):
        p.error('Choose distinct station groups, 0-99')
    torch.set_num_threads(2)

    def absolute(path):
        path = Path(path)
        return (path if path.is_absolute() else ROOT/path).resolve()

    output = absolute(a.output)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory; preserve existing results: {output}')
    # Validate all required files before creating outputs; never trigger training.
    entries = []
    for group in a.groups:
        data_path = absolute(a.data_dir)/f'temperature10_group{group}.npz'
        paths = {
            'energy': absolute(a.energy_runs)/f'group{group:02d}/energy/evaluation',
            'riesz_softmax': absolute(a.riesz_runs)/f'group{group:02d}/riesz_softmax/evaluation',
        }
        if not a.skip_uniform:
            paths['riesz_uniform'] = absolute(a.uniform_runs)/f'group{group:02d}/evaluation'
        for path in [data_path, data_path.with_suffix('.json'),
                     *(d/f for d in paths.values() for f in ('predictions.npz', 'metrics.json'))]:
            if not path.is_file():
                raise FileNotFoundError(f'Missing saved input: {path}. This evaluator does not train models.')
        entries.append((group, data_path, paths))
    rows, arrays = [], {}
    for group, data_path, paths in entries:
        meta = json.loads(data_path.with_suffix('.json').read_text())
        if meta['station_group'] != group or len(meta['stations']) != 10 or len(set(meta['stations'])) != 10:
            raise ValueError('Prepared station order/group mismatch')
        with np.load(data_path, allow_pickle=False) as data:
            lat, lon = recover_coordinates(data, meta)
            neighbors = nearest_other_stations(lat, lon, a.neighbors)
            mean, std, counts = seasonal_climatology(data['train_target'], data['train_dates'],
                                                    data['test_dates'], a.window)
            target = data['test_target'].astype(float)
            row = {'group': group, 'stations': meta['stations'], 'data_sha256': digest(data_path),
                   'latitude': lat.tolist(), 'longitude': lon.tolist(),
                   'neighbors': [[meta['stations'][j] for j in indices] for indices in neighbors],
                   'climatology_min_training_count': int(counts.min()), 'methods': {}, 'inputs': {}}
            arrays[f'g{group:02d}_dates'] = data['test_dates']
            arrays[f'g{group:02d}_climatology_mean'] = mean
            arrays[f'g{group:02d}_climatology_std'] = std
            ensembles = {}
            for method, directory in paths.items():
                pred, evaluation = load_forecast(directory, data, row['data_sha256'], method)
                ensembles[method] = pred
                row['inputs'][method] = {'directory': str(directory),
                                        'predictions_sha256': digest(directory/'predictions.npz'),
                                        'metrics_sha256': digest(directory/'metrics.json')}
                row['methods'][method] = dict(evaluation['metrics'])
            ensembles['raw'] = data['test_raw']
            row['methods']['raw'] = {k: v.mean().item() for k, v in
                                    scores(torch.from_numpy(data['test_raw']), torch.from_numpy(target)).items()}
            for method, pred in ensembles.items():
                physical = local_variogram(pred, target, neighbors)
                anomaly = local_variogram((pred-mean[:, None])/std[:, None],
                                          (target-mean)/std, neighbors)
                row['methods'][method].update(local_variogram_anomaly=float(anomaly.mean()),
                                               local_variogram_physical=float(physical.mean()))
                arrays[f'g{group:02d}_{method}_lvs_anomaly'] = anomaly
                arrays[f'g{group:02d}_{method}_lvs_physical'] = physical
            rows.append(row)
            print('Group', group, ':', ', '.join(f'{m} LVS {r["local_variogram_anomaly"]:.6f}'
                                               for m, r in row['methods'].items()), flush=True)
    methods = list(rows[0]['methods'])
    metrics = list(rows[0]['methods']['energy'])
    summary = {
        'complete': True, 'groups': rows, 'requested_groups': a.groups,
        'source_repository': 'https://github.com/davidlandry93/sidown',
        'source_revision': SOURCE_REVISION,
        'protocol': {'neighbors': a.neighbors, 'central_station_excluded': True, 'power': .5,
                     'pair_reduction': 'sum all ordered pairs; no K^2 division',
                     'aggregation': 'mean cases and centers within each group; equal mean over groups',
                     'climatology': f'2007-2014 observations only; wrapped {a.window}-day ordinal-day window, ddof=0',
                     'calendar': '366-day cycle; actual day-of-year; leap-year shift retained',
                     'coordinates': 'static lat/lon recovered from existing train_context and saved scaler',
                     'scope': 'five nearest OTHER stations within each existing ten-station model group',
                     'differences_from_paper': 'Original uses full EUPPBench network and its reforecast climatology; '
                                               'ours uses Chen/Lerch temperature data, one lead time and train-only climatology.',
                     'retraining': False, 'resampling': False},
        'group_mean_scores': {m: {k: float(np.mean([r['methods'][m][k] for r in rows]))
                                  for k in metrics} for m in methods},
        'paired_group_differences_vs_energy': {
            m: {k: [r['methods'][m][k]-r['methods']['energy'][k] for r in rows]
                for k in ('local_variogram_anomaly', 'local_variogram_physical')}
            for m in methods if m != 'energy'},
        'interpretation': 'Lower is better. Exploratory reuse of previously inspected 2016 forecasts. '
                          'Overlapping groups are not independent replicates. No significance claim. '
                          'Anomaly LVS and physical LVS have different scales; do not compare their raw values.',
    }
    output.mkdir(parents=True)
    np.savez_compressed(output/'per_case_scores.npz', **arrays)
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('\nMethod                 ES       VS      CRPS   LVS anomaly   LVS physical')
    for method, values in summary['group_mean_scores'].items():
        print(f'{method:<20} {values["energy_score"]:8.4f} {values["variogram_score"]:8.4f} '
              f'{values["mean_crps"]:8.4f} {values["local_variogram_anomaly"]:13.6f} '
              f'{values["local_variogram_physical"]:14.6f}')
    print(f'Summary: {output / "summary.json"}')


if __name__ == '__main__':
    main()
