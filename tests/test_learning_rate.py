import pytest
from tune_learning_rate import rank_learning_rates


def test_each_method_gets_its_own_validation_winner_from_same_complete_grid():
    trials=[]
    for method,values in [('energy',[1.,2.]),('riesz_softmax',[3.,1.])]:
        for lr,value in zip([.001,.002],values):
            for group in [0,1]:
                trials.append({'method':method,'group':group,'seed':42,'learning_rate':lr,
                               'validation_energy':value+group,'test_energy':-value})
    ranked=rank_learning_rates(trials,[0,1],[42],[.001,.002])
    assert ranked['energy'][0]['learning_rate']==.001
    assert ranked['riesz_softmax'][0]['learning_rate']==.002
    with pytest.raises(ValueError): rank_learning_rates(trials[:-1],[0,1],[42],[.001,.002])
    with pytest.raises(ValueError): rank_learning_rates(trials+[trials[0]],[0,1],[42],[.001,.002])
