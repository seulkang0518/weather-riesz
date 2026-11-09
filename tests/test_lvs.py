import numpy as np
import pytest

from evaluate_lvs import load_forecast
from weather.local_variogram import (local_variogram, nearest_other_stations,
                                    recover_coordinates, seasonal_climatology)


def test_local_vs_pair_sum_and_self_exclusion():
    # Each center excludes itself; with K=2 there are two identical ordered pairs.
    y = np.array([[0., 1., 9.]])
    p = np.array([[[0., 1., 4.], [0., 1., 4.]]])
    neighbors = np.array([[1, 2], [0, 2], [0, 1]])
    result = local_variogram(p, y, neighbors)
    np.testing.assert_allclose(result, [[2*(np.sqrt(8)-np.sqrt(3))**2, 2., 0.]])
    np.testing.assert_array_equal(local_variogram(y[:, None], y, neighbors), 0)
    # A common physical temperature shift is invisible to differences.
    np.testing.assert_allclose(local_variogram(p+5, y, neighbors), result)
    with pytest.raises(ValueError, match='other-station'):
        local_variogram(p, y, np.array([[0, 1], [0, 2], [0, 1]]))


def test_geographic_neighbors_exclude_self_and_handle_ties():
    lat, lon = np.zeros(4), np.array([0., 1., 3., 7.])
    neighbors = nearest_other_stations(lat, lon, 2)
    np.testing.assert_array_equal(neighbors, [[1, 2], [0, 2], [1, 0], [2, 1]])
    # Duplicate coordinates must still exclude self deterministically.
    duplicate = nearest_other_stations([0, 0, 0], [0, 0, 1], 2)
    assert all(i not in row for i, row in enumerate(duplicate))
    with pytest.raises(ValueError):
        nearest_other_stations(lat, lon, 4)


def test_climatology_uses_training_only_and_wraps_year_boundary():
    dates = np.arange('2007-01-01', '2015-01-01', dtype='datetime64[D]')
    values = np.stack([np.arange(len(dates)) % 13, np.arange(len(dates)) % 17], 1).astype(float)
    evaluation = np.array(['2016-01-01', '2016-12-31'])
    mean, std, counts = seasonal_climatology(values, dates, evaluation)
    doy = (dates-dates.astype('datetime64[Y]')).astype(int)+1
    for index, day in enumerate((1, 366)):
        delta = np.abs(doy-day)
        selected = values[np.minimum(delta, 366-delta) <= 10]
        np.testing.assert_allclose(mean[index], selected.mean(0))
        np.testing.assert_allclose(std[index], selected.std(0))
        assert counts[day-1] == len(selected)
    with pytest.raises(ValueError, match='training observations'):
        seasonal_climatology(values[-365:], np.arange('2016-01-01', '2016-12-31', dtype='datetime64[D]'), evaluation)


def test_recover_coordinates_preserves_existing_column_order():
    context = np.zeros((3, 3, 39))
    context[:, :, 34] = [1, 2, 3]
    context[:, :, 35] = [3, 2, 1]
    columns = ['x']*92
    columns[87], columns[88] = 'lat', 'lon'
    metadata = {'input_columns': columns, 'extra_scaler': {'mean': [0]*36, 'std': [1]*36}}
    lat, lon = recover_coordinates({'train_context': context}, metadata)
    np.testing.assert_array_equal(lat, [1, 2, 3])
    np.testing.assert_array_equal(lon, [3, 2, 1])
    context[1, 0, 34] += 1
    with pytest.raises(ValueError, match='not static'):
        recover_coordinates({'train_context': context}, metadata)


def test_saved_forecast_rejects_wrong_dataset_and_case_order(tmp_path):
    import json
    metadata = {'data_sha256': 'expected', 'method': 'energy', 'paper_sampling_protocol': True,
                'scenarios': 50, 'models': [{'seed': i, 'smoke_only': False} for i in range(10)]}
    (tmp_path/'metrics.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='Incompatible'):
        load_forecast(tmp_path, {}, 'wrong', 'energy')
    target = np.arange(6).reshape(2, 3)
    np.savez(tmp_path/'predictions.npz', target=target[::-1], dates=['2016-01-01', '2016-01-02'])
    with pytest.raises(ValueError, match='alignment'):
        load_forecast(tmp_path, {'test_target': target, 'test_dates': np.array(['2016-01-01', '2016-01-02'])},
                      'expected', 'energy')
