import math
import pytest
from tune_temperature import temperatures, rank_trials, reusable_trials


def test_scaling_preserves_number_and_relative_temperatures():
    base = [.2,.05,.02]
    assert temperatures(base,4) == [.8,.2,.08]
    assert len(temperatures(base,.25)) == len(base)
    for invalid in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            temperatures(base,invalid)


def test_selection_uses_complete_paired_validation_grid_only():
    trials = []
    for scale, scores in [(1,[2,4]), (4,[1,3])]:
        for group, score in enumerate(scores):
            trials.append({'multiplier':scale,'group':group,'seed':42,
                           'validation_energy':score, 'test_energy':100 if scale==4 else 0})
    result = rank_trials(trials,[0,1],[42],[1,4])
    assert result[0]['multiplier'] == 4
    assert result[0]['mean_validation_energy'] == 2
    with pytest.raises(ValueError):
        rank_trials(trials[:-1],[0,1],[42],[1,4])
    with pytest.raises(ValueError):
        rank_trials(trials+[trials[0]],[0,1],[42],[1,4])


def test_reuse_requires_complete_validation_only_sweep_with_identical_config():
    from copy import deepcopy
    base = {'temperatures':[.2,.05,.02],'learning_rate':.001}
    trials = [{'group':0,'seed':42,'multiplier':16,'validation_energy':3.,
               'temperatures':temperatures(base['temperatures'],16)}]
    prior = {'experiment':{'base_config':base,'groups':[0],'seeds':[42],'multipliers':[16]},
             'trials':trials,'ranking':rank_trials(trials,[0],[42],[16]),
             'test_evaluated_by_this_sweep':False}
    assert reusable_trials(prior,base)[(0,42,16)] == trials[0]
    with pytest.raises(ValueError):
        reusable_trials(prior,dict(base,learning_rate=.002))
    bad=deepcopy(prior); bad['test_evaluated_by_this_sweep']=True
    with pytest.raises(ValueError): reusable_trials(bad,base)
    bad=deepcopy(prior); bad['trials'][0]['temperatures']=[1.,1.,1.]
    with pytest.raises(ValueError): reusable_trials(bad,base)
