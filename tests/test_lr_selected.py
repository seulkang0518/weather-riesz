from copy import deepcopy
import pytest
from evaluate_learning_rate import validate_lr_selection
from tune_learning_rate import rank_learning_rates


def test_frozen_lrs_must_match_independent_validation_winners_and_fixed_temperatures():
    base={'learning_rate':.001,'temperatures':[.2,.05,.02]}
    fixed=[3.2,.8,.32]; rates=[.001,.0005]
    trials=[]
    for method,values in [('energy',[2.,1.]),('riesz_softmax',[1.,2.])]:
        for rate,value in zip(rates,values):
            trials.append({'method':method,'group':0,'seed':42,'learning_rate':rate,'validation_energy':value})
    ranking=rank_learning_rates(trials,[0],[42],rates)
    methods={}
    for method in ('energy','riesz_softmax'):
        lr=ranking[method][0]['learning_rate']; cfg=dict(base,learning_rate=lr)
        if method=='riesz_softmax': cfg['temperatures']=fixed
        methods[method]={'learning_rate':lr,'config':cfg,'ranking':ranking[method]}
    s={'kind':'learning_rate','experiment':{'base_config':base,'fixed_riesz_temperatures':fixed,
       'groups':[0],'seeds':[42],'learning_rates':rates},'methods':methods,'trials':trials,'test_evaluated_by_this_sweep':False}
    _,configs=validate_lr_selection(s)
    assert configs['energy']['learning_rate']==.0005 and configs['riesz_softmax']['learning_rate']==.001
    for field,value in [('learning_rate',.002),('config',dict(base))]:
        bad=deepcopy(s); bad['methods']['energy'][field]=value
        with pytest.raises(ValueError): validate_lr_selection(bad)
    bad=deepcopy(s); bad['methods']['riesz_softmax']['config']['temperatures']=[1.,1.,1.]
    with pytest.raises(ValueError): validate_lr_selection(bad)
    bad=deepcopy(s); bad['test_evaluated_by_this_sweep']=True
    with pytest.raises(ValueError): validate_lr_selection(bad)
