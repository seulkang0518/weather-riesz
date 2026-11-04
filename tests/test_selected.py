from copy import deepcopy
import pytest
from evaluate_selected import validate_selection
from tune_temperature import rank_trials


def selection():
    trials = [{'group':0,'seed':42,'multiplier':m,'validation_energy':score} for m,score in [(1,3.),(4,2.)]]
    return {'experiment':{'base_config':{'temperatures':[.2,.05,.02]},'groups':[0],
                          'seeds':[42],'multipliers':[1,4]},'trials':trials,
            'ranking':rank_trials(trials,[0],[42],[1,4]),'selected_multiplier':4,
            'selected_temperatures':[.8,.2,.08],'test_evaluated_by_this_sweep':False}


def test_require_frozen_validation_winner_not_a_test_selected_candidate():
    s = selection()
    assert validate_selection(s)[2] == [.8,.2,.08]
    bad = deepcopy(s); bad['selected_multiplier'] = 1
    with pytest.raises(ValueError): validate_selection(bad)
    bad = deepcopy(s); bad['test_evaluated_by_this_sweep'] = True
    with pytest.raises(ValueError): validate_selection(bad)
    bad = deepcopy(s); bad['trials'].pop()
    with pytest.raises(ValueError): validate_selection(bad)
