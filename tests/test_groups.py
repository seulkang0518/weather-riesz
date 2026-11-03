import pytest
from run_groups import aggregate, verify_evaluation


def test_equal_group_weight_and_paired_differences():
    # Unequal case counts must not implicitly reweight groups in the paper mean.
    rows = []
    for group, values in enumerate([(2., 3.), (6., 5.)]):
        row = {'group': group, 'counts': {'test': 2 if group == 0 else 100}}
        for method, value in zip(('energy', 'riesz_softmax'), values):
            row[method] = dict.fromkeys(('energy_score','variogram_score','mean_crps'), value)
        row['raw'] = dict.fromkeys(row['energy'], 10.)
        rows.append(row)
    result = aggregate(rows, [0,1])
    assert result['complete']
    assert result['group_mean_scores']['energy']['energy_score'] == 4.
    assert result['riesz_minus_energy']['energy_score']['per_group'] == [1.,-1.]
    assert result['riesz_minus_energy']['energy_score']['riesz_better_groups'] == 1
    assert not aggregate(rows[:1], [0,1])['complete']


def test_reject_wrong_data_or_sampling_protocol():
    result = {'method':'energy', 'data_sha256':'correct', 'paper_sampling_protocol':True,
              'scenarios':50, 'models':[{'seed':s,'smoke_only':False} for s in range(10)],
              'metrics':dict.fromkeys(('energy_score','variogram_score','mean_crps'), 1.)}
    verify_evaluation(result, 'energy', 'correct', list(range(10)))
    with pytest.raises(ValueError):
        verify_evaluation(result, 'energy', 'wrong', list(range(10)))
    with pytest.raises(ValueError):
        verify_evaluation(dict(result, scenarios=500), 'energy', 'correct', list(range(10)))
