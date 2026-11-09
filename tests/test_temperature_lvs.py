import copy

import pytest
import torch

from tune_temperature_lvs import rank_candidates, repulsion_statistics
from weather.riesz_topk_softmax import _softmax_toward


def test_repulsion_diagnostics_match_upstream_scale_and_softmax():
    torch.manual_seed(31)
    pred, target = torch.randn(3, 8, 10), torch.randn(3, 10)
    taus = [.02, .2, 10.24]
    stats = repulsion_statistics(pred, target, taus)
    scale = torch.cdist(pred, torch.cat([pred, target[:, None]], 1)).mean()/(1+1e-6)
    coordinate_scale = (scale/(10**.5)).clamp_min(1e-3)
    for tau, row in zip(taus, stats):
        _, ess = _softmax_toward(pred/coordinate_scale, pred/coordinate_scale,
                                 torch.ones(3, 8), tau*(10**.5), 1e-8, exclude_self=True)
        assert row['effective_neighbors'] == pytest.approx(ess.mean().item())
        assert 1-1e-6 <= row['effective_neighbors'] <= 7+1e-6
    assert stats[0]['effective_neighbors'] < stats[1]['effective_neighbors'] < stats[2]['effective_neighbors']
    assert stats[2]['fraction_of_uniform_ess'] > .99


def test_lvs_ranking_is_validation_only_complete_and_can_disagree_with_es():
    trials = []
    for group in [0, 1]:
        for scale in [1, 512]:
            trials.append({'group': group, 'seed': 42, 'multiplier': scale,
                           'validation_lvs_anomaly': 1 if scale==1 else 2,
                           'validation_energy_empirical': 2 if scale==1 else 1,
                           'test_lvs': 100 if scale==1 else 0,
                           'repulsion_weights': [{'temperature': float(scale), 'effective_neighbors': 2.,
                                                  'fraction_of_uniform_ess': .5, 'maximum_neighbor_weight': .7}]})
    assert rank_candidates(trials, [0, 1], [42], [1, 512])[0]['multiplier'] == 1
    for bad in [trials[:-1], trials+[trials[0]]]:
        with pytest.raises(ValueError): rank_candidates(bad, [0, 1], [42], [1, 512])
    bad = copy.deepcopy(trials); bad[0]['validation_lvs_anomaly'] = float('nan')
    with pytest.raises(ValueError): rank_candidates(bad, [0, 1], [42], [1, 512])
