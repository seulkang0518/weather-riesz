import copy

import pytest
import torch

from tune_temperature_lvs import rank_candidates, repulsion_statistics,trial_config,particle_reference,score_validation
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


def test_eight_particle_trial_keeps_fifty_validation_samples_and_reference_unchanged():
    ref={'train_samples':50,'temperatures':[102.4,25.6,10.24],'learning_rate':.0005}
    cfg=trial_config(ref,[.2,.05,.02],.25,2,8)
    assert cfg==dict(ref,train_samples=8,validation_samples=50,temperatures=[.05,.0125,.005],group=2)
    assert ref['train_samples']==50 and 'validation_samples' not in ref
    assert trial_config(ref,[.2,.05,.02],512,0)==dict(ref,group=0)


def test_particle_reference_follows_read_only_pilot_without_reading_test_scores(tmp_path,monkeypatch):
    import json
    import hashlib
    import tune_temperature_lvs as module
    inherited={'train_samples':50,'temperatures':[102.4,25.6,10.24],'learning_rate':.0005}
    spec={'lr_selection_sha256':'lr','groups':[0],'particles':[50,8,4],
          'methods':['riesz_softmax'],'reference_configs':{'riesz_softmax':inherited},
          'validation_samples':50,'test_samples':50}
    pilot=tmp_path/'pilot'; pilot.mkdir()
    (pilot/'experiment.json').write_text(json.dumps(spec))
    (pilot/'summary.json').write_text('Deliberately invalid: test summary must not be parsed')
    cell=pilot/'group00/riesz_softmax/particles8'; (cell/'seed42').mkdir(parents=True)
    record={'checkpoint':str(cell/'seed42/best.pt'),'checkpoint_sha256':'checkpoint',
            'validation_energy':3.,'best_epoch':2,'stopped_epoch':12}
    provenance=cell/'checkpoint_provenance.json'
    provenance.write_text(json.dumps({'fits':[dict(record,seed=42)]}))
    full=tmp_path/'full'; full.mkdir()
    full_spec=dict(spec,reuse_run={'directory':str(pilot),
                   'experiment_sha256':hashlib.sha256((pilot/'experiment.json').read_bytes()).hexdigest()})
    (full/'experiment.json').write_text(json.dumps(full_spec))
    def checked_fit(work,cfg,seed,data_hash,**kwargs):
        assert cfg==dict(inherited,train_samples=8,validation_samples=50,group=0)
        assert seed==42 and data_hash=='data'
        return dict(record)
    monkeypatch.setattr(module,'fit_record',checked_fit)
    before=provenance.stat().st_mtime_ns
    work,result=particle_reference(full,inherited,0,42,8,'data','lr')
    assert work==cell/'seed42' and result==record and provenance.stat().st_mtime_ns==before
    bad=dict(spec,lr_selection_sha256='changed'); (pilot/'experiment.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='pilot experiment changed'):
        particle_reference(full,inherited,0,42,8,'data','lr')


def test_validation_quality_uses_fifty_predictions_but_diagnostics_use_eight(monkeypatch):
    import numpy as np
    import tune_temperature_lvs as module
    from weather.model import TemperatureCGM
    from weather.objectives import training_energy_score
    from train import load_split,predict
    torch.manual_seed(81)
    model=TemperatureCGM(10,10); model.eval()
    cfg={'stations':10,'latent_dim':10,'train_samples':8,'validation_samples':50,
         'validation_seed':25,'batch_size':64,'temperatures':[.2,.05,.02]}
    data={'val_mean':np.ones((3,10,17),dtype=np.float32),
          'val_std':np.zeros((3,10,17),dtype=np.float32),
          'val_context':np.zeros((3,10,39),dtype=np.float32),
          'val_target':np.zeros((3,10),dtype=np.float32),
          'train_target':np.zeros((3,10)),'train_dates':np.array(['2007-01-01']*3),
          'val_dates':np.array(['2015-01-01']*3)}
    inputs=load_split(data,'val','cpu')
    generator=torch.Generator().manual_seed(cfg['validation_seed'])
    with torch.no_grad():
        expected=predict(model,inputs,torch.arange(3),50,generator)
        best=training_energy_score(expected,inputs['target']).mean().item()
    state={'model':model.state_dict(),'best':best,'manifest':{'data_metadata':{}}}
    monkeypatch.setattr(module.torch,'load',lambda *args,**kwargs:state)
    monkeypatch.setattr(module,'recover_coordinates',lambda *args:(np.arange(10)*.1+49,np.arange(10)*.2+8))
    monkeypatch.setattr(module,'seasonal_climatology',lambda *args:(np.zeros((3,10)),np.ones((3,10)),None))
    original=module.repulsion_statistics; counts=[]
    def capture(pred,target,taus):
        counts.append(pred.shape[1]); return original(pred,target,taus)
    monkeypatch.setattr(module,'repulsion_statistics',capture)
    result=score_validation('unused',data,cfg,'cpu')
    assert result['validation_samples']==50 and result['repulsion_particles']==8
    assert result['repulsion_available_neighbors']==7 and counts==[8]
    assert all(1-1e-5<=r['effective_neighbors']<=7+1e-5 for r in result['repulsion_weights'])
    expected_metrics=module.scores(expected,torch.from_numpy(data['val_target']))
    assert result['validation_energy_empirical']==pytest.approx(expected_metrics['energy_score'].mean().item())
