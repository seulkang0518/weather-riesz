from copy import deepcopy

import pytest

from evaluate_temperature_lvs import validate_lvs_selection, particle_forecast
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
    bad = deepcopy(selection); bad['experiment']['train_particles'] = 8
    with pytest.raises(ValueError,match='particle-matched'):
        validate_lvs_selection(bad, lr)


def test_lower_count_selection_requires_opt_in_and_keeps_validation_at_fifty():
    selection, lr = example()
    for base in (selection['experiment']['base_config'], lr['experiment']['base_config']):
        base['train_samples'] = 50
    for method in lr['methods'].values():
        method['config']['train_samples'] = 50
    selection['experiment']['train_particles'] = 8
    with pytest.raises(ValueError, match='particle-matched'):
        validate_lvs_selection(selection, lr)
    _, configs, cfg = validate_lvs_selection(selection, lr, train_particles=8)
    assert cfg['train_samples'] == configs['energy']['train_samples'] == 8
    assert cfg['validation_samples'] == configs['energy']['validation_samples'] == 50
    assert cfg['learning_rate'] == .0005 and cfg['temperatures'] == [12.8, 3.2, 1.28]
    assert configs['riesz_softmax']['temperatures'] == [102.4, 25.6, 10.24]
    assert lr['methods']['riesz_softmax']['config']['train_samples'] == 50


def test_particle_forecast_checks_all_matched_fits_and_follows_read_only_pilot(tmp_path, monkeypatch):
    import json
    import evaluate_temperature_lvs as module
    from tune_temperature import digest
    reference = {'train_samples': 50, 'seeds': [42, 43], 'temperatures': [102.4, 25.6, 10.24]}
    spec = {'groups': [0], 'particles': [50, 8], 'methods': ['energy'],
            'reference_configs': {'energy': reference}, 'lr_selection_sha256': 'lr',
            'validation_samples': 50, 'test_samples': 50}
    pilot = tmp_path/'pilot'; pilot.mkdir()
    (pilot/'experiment.json').write_text(json.dumps(spec))
    cell = pilot/'group00/energy/particles8'; evaluation = cell/'evaluation'
    evaluation.mkdir(parents=True)
    (evaluation/'predictions.npz').write_bytes(b'saved forecasts')
    records = [{'seed': seed, 'checkpoint_sha256': str(seed)} for seed in reference['seeds']]
    pf = cell/'checkpoint_provenance.json'; pf.write_text(json.dumps({'fits': records}))
    full = tmp_path/'full'; full.mkdir()
    (full/'experiment.json').write_text(json.dumps(dict(spec, reuse_run={
        'directory': str(pilot), 'experiment_sha256': digest(pilot/'experiment.json')})))
    def check(path, cfg, data_hash, method):
        assert path == evaluation/'metrics.json' and method == 'energy' and data_hash == 'data'
        assert cfg == dict(reference, train_samples=8, validation_samples=50, group=0)
        return {}, records
    monkeypatch.setattr(module, 'checked_reference', check)
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob('*') if p.is_file()}
    assert particle_forecast(full, reference, 0, 8, 'data', 'lr', 'energy') == (evaluation, records)
    assert before == {p: p.stat().st_mtime_ns for p in before}
    with pytest.raises(ValueError, match='nonnested'):
        particle_forecast(full, reference, 0, 8, 'data', 'lr', 'energy', output_root=pilot/'new')
    with pytest.raises(ValueError, match='recipe/group/count/protocol'):
        particle_forecast(full, reference, 0, 4, 'data', 'lr', 'energy')
    pf.write_text(json.dumps({'fits': records[:-1]}))
    with pytest.raises(ValueError, match='provenance changed'):
        particle_forecast(full, reference, 0, 8, 'data', 'lr', 'energy')
