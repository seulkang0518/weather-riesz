"""Download verified public data, select an official station group, fit train-only scalers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.compute as pc
import pyarrow as pa

ROOT = Path(__file__).resolve().parent
FILENAME = 'temperature_data_cgm_std.feather'
MD5 = '76b1615b7a9b3e774c5e8ee98ae39ccc'
URL = 'https://ndownloader.figshare.com/files/36236547'


def digest(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def prepare(source, output, group=0, stations=10):
    if digest(source) != MD5:
        raise ValueError('Dataset checksum mismatch; refuse to train on an unverified file')
    station_file = ROOT / f'third_party/mvpp/t2m_dist_{stations}samples.csv'
    groups = np.loadtxt(station_file, delimiter=',').astype(int)
    if not 0 <= group < len(groups):
        raise ValueError('Group must be between 0 and 99')
    selected = groups[group].tolist()
    # Public file is legacy Feather V1, not an Arrow IPC/Feather V2 dataset.
    table = feather.read_table(source, memory_map=True)
    names = table.schema.names
    if len(names) < 92 or not {'station', 'date', 'obs'}.issubset(names):
        raise ValueError(f'Unexpected source schema: {names}')
    station_values = pa.array(selected, type=table['station'].type)
    df = table.select(names[:92]).filter(pc.is_in(table['station'], value_set=station_values)).to_pandas()
    del table
    df['date'] = pd.to_datetime(df['date'])
    counts = df.groupby('date')['station'].agg(['size', 'nunique'])
    dates = counts.index[(counts['size'] == stations) & (counts['nunique'] == stations)]
    df = df[df.date.isin(dates)].copy()
    # Canonical station identity per vector; preserve official input-file order on the
    # first complete day rather than silently imposing CSV-neighbour order.
    first_date = df.date.min()
    order = df.loc[df.date == first_date, 'station'].astype(int).tolist()
    df['_station_order'] = pd.Categorical(df.station, categories=order, ordered=True)
    df = df.sort_values(['date', '_station_order']).drop(columns='_station_order').reset_index(drop=True)
    expected = np.tile(order, len(dates))
    if not np.array_equal(df.station.to_numpy(), expected):
        raise ValueError('Inconsistent station identities within forecast vectors')
    if df.date.dt.year.min() < 2007 or df.date.dt.year.max() > 2016:
        raise ValueError('Unexpected dates in the public data')
    raw = df.iloc[:, 3:53].to_numpy(dtype=np.float64)
    extra = df.iloc[:, 55:92].to_numpy(dtype=np.float64, copy=True)
    obs = df.obs.to_numpy(dtype=np.float64)
    if not all(np.isfinite(x).all() for x in (raw, extra, obs)):
        raise ValueError('Non-finite data in selected complete forecast cases')
    train_rows = df.date.dt.year.to_numpy() <= 2014
    # Match the official scaler: one pooled observation mean/std across selected
    # stations; standardize ensemble/extra features, retain physical target values.
    obs_mean, obs_std = obs[train_rows].mean(), obs[train_rows].std(ddof=0)
    obs_std = obs_std if obs_std > 0 else 1.0
    extra_mean = extra[train_rows, :-1].mean(0)
    extra_std = extra[train_rows, :-1].std(0, ddof=0)
    extra_std[extra_std == 0] = 1.0
    extra[:, :-1] = (extra[:, :-1] - extra_mean) / extra_std
    norm = (raw - obs_mean) / obs_std
    # Important code-level detail: mean/scale modules receive raw target-ensemble
    # mean/std; decoder gets standardized target ensemble mean/std.
    mean = np.concatenate([raw.mean(1)[:, None], extra[:, 0:32:2]], 1)
    std = np.concatenate([raw.std(1, ddof=1)[:, None], extra[:, 1:32:2]], 1)
    context = np.concatenate([norm.mean(1)[:, None], norm.std(1, ddof=1)[:, None], extra], 1)
    out = {'mean': mean.reshape(-1, stations, 17), 'std': std.reshape(-1, stations, 17),
           'context': context.reshape(-1, stations, 39), 'target': obs.reshape(-1, stations),
           'raw': raw.reshape(-1, stations, 50).transpose(0, 2, 1)}
    case_dates = df.date.to_numpy()[::stations]
    years = pd.DatetimeIndex(case_dates).year
    masks = {'train': years <= 2014, 'val': years == 2015, 'test': years == 2016}
    arrays = {f'{split}_{key}': value[mask].astype(np.float32)
              for split, mask in masks.items() for key, value in out.items()}
    arrays.update({f'{split}_dates': case_dates[mask].astype('datetime64[D]').astype(str)
                   for split, mask in masks.items()})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    meta = {'source_md5': MD5, 'station_group': group, 'stations': order,
            'counts': {s: int(m.sum()) for s, m in masks.items()},
            'observation_scaler': {'mean': float(obs_mean), 'std': float(obs_std)},
            'extra_scaler': {'mean': extra_mean.tolist(), 'std': extra_std.tolist()},
            'input_columns': names[:92], 'target_units': 'unchanged from public dataset',
            'sort': 'date, then first complete date station order',
            'matching': 'official temperature preprocessing, explicit station/date alignment'}
    output.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({'output': str(output), **meta['counts'], 'stations': order}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default=str(ROOT / 'data' / FILENAME))
    p.add_argument('--output', default=str(ROOT / 'data/temperature10_group0.npz'))
    p.add_argument('--group', type=int, default=0)
    p.add_argument('--stations', type=int, choices=[5, 10, 20], default=10)
    p.add_argument('--download', action='store_true')
    a = p.parse_args()
    source = Path(a.source)
    if not source.exists():
        if not a.download:
            p.error('Source missing: use --download (1.36 GB), or provide --source')
        source.parent.mkdir(parents=True, exist_ok=True)
        tmp = source.with_suffix('.part')
        print(f'Downloading {URL} (~1.36 GB)', flush=True)
        urllib.request.urlretrieve(URL, tmp)
        if digest(tmp) != MD5:
            raise ValueError('Downloaded checksum mismatch')
        os.replace(tmp, source)
    prepare(source, a.output, a.group, a.stations)


if __name__ == '__main__':
    main()
