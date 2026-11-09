import pytest
from compare_particles import particle_config,summarize,benchmark_loss,cached_lvs,ALL_METHODS,LVS_METRICS
from train import validation_sample_count


def test_particle_change_keeps_validation_fixed_and_reference_config_compatible():
    ref={'train_samples':50,'learning_rate':.0005,'temperatures':[102.4,25.6,10.24]}
    assert particle_config(ref,50)==ref
    for count in (4,8,16):
        cfg=particle_config(ref,count)
        assert cfg['train_samples']==count and validation_sample_count(cfg)==50
        assert {k:v for k,v in cfg.items() if k not in ('train_samples','validation_samples')}=={k:v for k,v in ref.items() if k!='train_samples'}
    assert ref['train_samples']==50 and 'validation_samples' not in ref
    with pytest.raises(ValueError): particle_config(ref,1)
    with pytest.raises(ValueError): validation_sample_count({'train_samples':4,'validation_samples':1})


def test_summary_separates_training_counts_and_uses_equal_group_means():
    rows=[]
    for g in [0,1]:
        for method in ['energy','riesz_uniform']:
            for count in [4,50]:
                rows.append({'group':g,'method':method,'train_particles':count,
                             'metrics':dict.fromkeys(['energy_score','variogram_score','mean_crps'],g+count)})
    s=summarize(rows,[0,1],[4,50]); assert s['complete']
    assert s['group_mean_scores'][0]['energy_score']==4.5
    assert s['test_scenarios_per_case']==50 and s['validation_scenarios_per_model']==50
    assert not summarize(rows[:-1],[0,1],[4,50])['complete']
    with pytest.raises(ValueError): summarize(rows+[rows[0]],[0,1],[4,50])


def test_extended_summary_includes_softmax_lvs_and_rejects_nonfinite_scores():
    metrics=('energy_score','variogram_score','mean_crps')+LVS_METRICS
    rows=[{'group':g,'method':m,'train_particles':n,'metrics':dict.fromkeys(metrics,float(g+n))}
          for g in (0,1) for m in ALL_METHODS for n in (4,50)]
    result=summarize(rows,[0,1],[4,50],ALL_METHODS,metrics)
    assert result['complete'] and len(result['group_mean_scores'])==6
    assert result['group_mean_scores'][2]['method']=='riesz_softmax'
    assert result['group_mean_scores'][2]['local_variogram_anomaly']==4.5
    rows[0]['metrics']['local_variogram_anomaly']=float('nan')
    with pytest.raises(ValueError): summarize(rows,[0,1],[4,50],ALL_METHODS,metrics)


def test_benchmark_uses_actual_softmax_gradient_not_uniform_surrogate():
    import torch
    from weather.objectives import riesz_loss
    torch.manual_seed(39)
    pred=torch.randn(2,7,3,requires_grad=True); target=torch.randn(2,3)
    taus=[.02]
    actual=torch.autograd.grad(benchmark_loss(pred,target,'riesz_softmax',taus),pred,retain_graph=True)[0]
    expected=torch.autograd.grad(riesz_loss(pred,target,taus,weighting='softmax')[0].mean(),pred,retain_graph=True)[0]
    uniform=torch.autograd.grad(benchmark_loss(pred,target,'riesz_uniform',taus),pred)[0]
    torch.testing.assert_close(actual,expected)
    assert not torch.allclose(actual,uniform)
    with pytest.raises(ValueError): benchmark_loss(pred,target,'unknown',taus)


def test_particle_lvs_cache_uses_same_forecasts_and_rejects_changed_identity(tmp_path,monkeypatch):
    import json
    import numpy as np
    import compare_particles as module
    from weather.local_variogram import local_variogram,nearest_other_stations,seasonal_climatology
    rng=np.random.default_rng(8)
    dates=np.arange('2007-01-01','2015-01-01',dtype='datetime64[D]')
    train=rng.normal(size=(len(dates),10))
    pred=rng.normal(size=(3,50,10)); target=rng.normal(size=(3,10))
    test_dates=np.array(['2016-01-01','2016-06-15','2016-12-31'])
    context=np.zeros((len(dates),10,39)); context[:,:,34]=np.arange(10)*.1+49
    context[:,:,35]=np.arange(10)*.2+8
    data=tmp_path/'data.npz'
    np.savez(data,train_target=train,train_dates=dates,test_target=target,test_dates=test_dates,train_context=context)
    columns=['x']*92; columns[87]='lat'; columns[88]='lon'
    data.with_suffix('.json').write_text(json.dumps({'input_columns':columns,'stations':list(range(10)),
            'extra_scaler':{'mean':[0]*36,'std':[1]*36}}))
    directory=tmp_path/'forecasts'; directory.mkdir()
    (directory/'metrics.json').write_text('{}')
    np.savez(directory/'predictions.npz',predictions=pred,target=target,dates=test_dates)
    monkeypatch.setattr(module,'load_forecast',lambda *args:(pred,{}))
    output=tmp_path/'diagnostics/lvs.json'
    actual=cached_lvs(directory,data,'fixed-data-hash','energy',output)
    neighbors=nearest_other_stations(context[0,:,34],context[0,:,35])
    mean,std,_=seasonal_climatology(train,dates,test_dates)
    assert actual['local_variogram_anomaly']==pytest.approx(local_variogram((pred-mean[:,None])/std[:,None],(target-mean)/std,neighbors).mean())
    before=output.stat().st_mtime_ns
    monkeypatch.setattr(module,'load_forecast',lambda *args:pytest.fail('Cache must not regenerate/reevaluate forecasts'))
    assert cached_lvs(directory,data,'fixed-data-hash','energy',output)==actual
    assert output.stat().st_mtime_ns==before
    (directory/'metrics.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='Cached LVS input'):
        cached_lvs(directory,data,'fixed-data-hash','energy',output)
