"""LVS arithmetic following Landry's sidown; data preparation is benchmark-specific."""
import numpy as np


def nearest_other_stations(latitude, longitude, k=5):
    """Great-circle ordering; exclude self explicitly, break exact ties by column."""
    lat, lon = np.asarray(latitude, dtype=float), np.asarray(longitude, dtype=float)
    if (lat.ndim != 1 or lon.shape != lat.shape or not np.isfinite([lat, lon]).all()
            or (np.abs(lat) > 90).any() or (np.abs(lon) > 180).any()):
        raise ValueError('Invalid station latitude/longitude')
    if not isinstance(k, int) or not 2 <= k < len(lat):
        raise ValueError('Require 2 <= neighbors < number of stations')
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    cosine = (np.sin(lat[:, None]) * np.sin(lat[None, :])
              + np.cos(lat[:, None]) * np.cos(lat[None, :])
              * np.cos(lon[:, None] - lon[None, :]))
    distance = np.arccos(np.clip(cosine, -1, 1))
    np.fill_diagonal(distance, np.inf)
    return np.argsort(distance, axis=1, kind='stable')[:, :k]


def day_of_year(dates):
    dates = np.asarray(dates, dtype='datetime64[D]')
    if dates.ndim != 1 or np.isnat(dates).any():
        raise ValueError('Require a finite vector of dates')
    return (dates - dates.astype('datetime64[Y]')).astype(int) + 1


def seasonal_climatology(train_target, train_dates, evaluation_dates, window=21):
    """Train-only, pooled-year population mean/std in circular day-of-year windows.

    366-position ordinal-day cycle; actual day-of-year retains the original leap-year
    shift. This is an adaptation of sidown's wrapped 21-day/year rolling statistics,
    using our complete training cases instead of EUPPBench reforecast climatology.
    """
    y = np.asarray(train_target, dtype=float)
    dates = np.asarray(train_dates, dtype='datetime64[D]')
    years = dates.astype('datetime64[Y]').astype(int) + 1970
    if (y.ndim != 2 or len(y) != len(dates) or not np.isfinite(y).all()
            or not ((years >= 2007) & (years <= 2014)).all()):
        raise ValueError('Climatology requires finite 2007-2014 training observations only')
    if not isinstance(window, int) or window < 3 or window > 365 or window % 2 != 1:
        raise ValueError('Climatology window must be odd, between 3 and 365')
    doy, eval_doy = day_of_year(dates), day_of_year(evaluation_dates)
    means, stds = np.empty((366, y.shape[1])), np.empty((366, y.shape[1]))
    counts = np.zeros(366, dtype=int)
    minimum = max(2, int(.2 * window * len(np.unique(years))))
    for d in range(1, 367):
        distance = np.abs(doy - d)
        mask = np.minimum(distance, 366-distance) <= window//2
        counts[d-1] = mask.sum()
        if counts[d-1] < minimum:
            raise ValueError(f'Insufficient training climatology near day {d}: {counts[d-1]}')
        means[d-1], stds[d-1] = y[mask].mean(0), y[mask].std(0, ddof=0)
    if not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError('Climatology contains a nonpositive/nonfinite standard deviation')
    return means[eval_doy-1], stds[eval_doy-1], counts


def local_variogram(predictions, target, neighbors):
    """Return [case, center] VS(p=.5), summing all ordered neighbor pairs.

    No division by K^2; the diagonal contributes zero. Average centers/cases later.
    Inputs may be physical temperatures or already standardized anomalies.
    """
    p, y = np.asarray(predictions, dtype=float), np.asarray(target, dtype=float)
    n = np.asarray(neighbors)
    if (p.ndim != 3 or y.shape != (p.shape[0], p.shape[2]) or p.shape[1] < 1
            or not np.isfinite(p).all() or not np.isfinite(y).all()):
        raise ValueError('Require finite predictions [case,member,station], target [case,station]')
    if (n.ndim != 2 or n.shape[0] != p.shape[2] or n.shape[1] < 2
            or not np.issubdtype(n.dtype, np.integer) or (n < 0).any()
            or (n >= p.shape[2]).any()
            or any(len(set(row)) != len(row) or center in row for center, row in enumerate(n))):
        raise ValueError('Each center needs distinct valid other-station indices')
    result = []
    for row in n:
        a, b = p[:, :, row], y[:, row]
        observed = np.sqrt(np.abs(b[:, :, None]-b[:, None, :]))
        predicted = np.sqrt(np.abs(a[:, :, :, None]-a[:, :, None, :])).mean(1)
        result.append(np.square(observed-predicted).sum((1, 2)))
    return np.stack(result, axis=1)


def recover_coordinates(data, metadata):
    """Recover static coordinates from existing standardized input columns.

    Avoid rewriting prepared NPZs, whose hashes are recorded in trained checkpoints.
    """
    extra_names = metadata['input_columns'][55:92]
    result = []
    for name in ('lat', 'lon'):
        index = extra_names.index(name)
        value = (np.asarray(data['train_context'][:, :, 2+index], dtype=float)
                 * metadata['extra_scaler']['std'][index]
                 + metadata['extra_scaler']['mean'][index])
        if not np.isfinite(value).all() or not np.allclose(value, value[0], rtol=0, atol=1e-5):
            raise ValueError(f'{name} is not static in prepared training inputs')
        result.append(value.mean(0))
    return result
