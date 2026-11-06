import pytest
from compare_particles import particle_config,summarize
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
