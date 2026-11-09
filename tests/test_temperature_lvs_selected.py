from copy import deepcopy

import pytest

from evaluate_temperature_lvs import validate_lvs_selection
from tune_learning_rate import rank_learning_rates
from tune_temperature_lvs import rank_candidates


def example():
    base = {'learning_rate': .001, 'temperatures': [.2,.05,.02], 'seeds': [42]}
    trials = [{'method': m, 'group': 0, 'seed': 42, 'learning_rate': .0005, 'validation_energy': 2.}
              for m in ('energy','riesz_softmax')]
    ranking = rank_learning_rates(trials, [0], [42], [.0005])
    configs = {'energy': dict(base, learning_rate=.0005),
               'riesz_softmax': dict(base, learning_rate=.0005, temperatures=[102.4,25.6,10.24])}
    lr = {'kind': 'learning_rate', 'test_evaluated_by_this_sweep': False, 'trials': trials,
          'experiment': {'base_config': base, 'groups': [0], 'seeds': [42], 'learning_rates': [.0005],
                         'fixed_riesz_temperatures': configs['riesz_softmax']['temperatures']},
          'methods': {m: {'learning_rate': .0005, 'config': c, 'ranking': ranking[m]} for m,c in configs.items()}}
    candidates = [{'group': 0, 'seed': 42, 'multiplier': scale, 'temperatures': taus,
                   'validation_lvs_anomaly': lvs, 'validation_energy_empirical': es,
                   'repulsion_weights': [{'temperature': taus[0], 'effective_neighbors': 49.,
                                          'fraction_of_uniform_ess': 1., 'maximum_neighbor_weight': 1/49}]}
                  for scale,taus,lvs,es in [(64,[12.8,3.2,1.28],1.,2.), (512,[102.4,25.6,10.24],2.,1.)]]
    selection = {'kind': 'temperature_lvs', 'test_evaluated_by_this_sweep': False,
                 'experiment': {'base_config': base, 'fixed_learning_rate': .0005, 'groups': [0],
                                'seeds': [42], 'multipliers': [64,512],
                                'lvs': {'neighbors': 5, 'exclude_center': True, 'power': .5, 'window': 21}},
                 'trials': candidates, 'ranking': rank_candidates(candidates,[0],[42],[64,512]),
                 'selected_multiplier': 64, 'selected_temperatures': [12.8,3.2,1.28]}
    return selection, lr


def test_frozen_lvs_winner_retains_lr_and_rejects_reselection_or_tampering():
    selection, lr = example()
    _, _, cfg = validate_lvs_selection(selection, lr)
    assert cfg['learning_rate'] == .0005  # Not the original .001 base LR.
    assert cfg['temperatures'] == [12.8,3.2,1.28]
    for key,value in [('selected_multiplier',512), ('test_evaluated_by_this_sweep',True),
                      ('selected_temperatures',[1.,1.,1.])]:
        bad = deepcopy(selection); bad[key] = value
        with pytest.raises(ValueError): validate_lvs_selection(bad, lr)
    bad = deepcopy(selection); bad['trials'].pop()
    with pytest.raises(ValueError): validate_lvs_selection(bad, lr)
    bad = deepcopy(selection); bad['experiment']['fixed_learning_rate'] = .001
    with pytest.raises(ValueError): validate_lvs_selection(bad, lr)
    bad = deepcopy(selection); bad['trials'][0]['temperatures'] = [.2,.05,.02]
    with pytest.raises(ValueError): validate_lvs_selection(bad, lr)
