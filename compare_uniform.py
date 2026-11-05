"""Single exact-uniform-weight ablation at the frozen Riesz softmax settings."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from evaluate_learning_rate import validate_lr_selection, check_evaluation
from evaluate_selected import validate_selection
from run_groups import ROOT, METRICS, read_json, run, verify_evaluation
from tune_temperature import digest, fit_record


def checked_reference(path,cfg,data_hash,method):
    result=read_json(path)
    verify_evaluation(result,method,data_hash,cfg['seeds'])
    provenance=[]
    for model in result['models']:
        work=Path(model['checkpoint']).resolve().parent
        record=fit_record(work,cfg,model['seed'],data_hash,method=method)
        if record is None: raise ValueError('Reference fit is incomplete')
        provenance.append(dict(record,seed=model['seed']))
    check_evaluation(result,method,data_hash,cfg['seeds'],provenance)
    return result,provenance


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--selection',default='runs/temperature_learning_rate_sweep/lr_selection.json')
    p.add_argument('--reference-runs',default='runs/temperature_learning_rate_selected_test')
    p.add_argument('--groups',nargs='+',type=int,default=list(range(10)))
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--data-dir',default='data')
    p.add_argument('--source',default='data/temperature_data_cgm_std.feather')
    p.add_argument('--workdir',default='runs/temperature_uniform_comparison')
    a=p.parse_args()
    def absolute(path):
        path=Path(path)
        return (path if path.is_absolute() else ROOT/path).resolve()
    if len(set(a.groups))!=len(a.groups) or not all(0<=g<100 for g in a.groups):
        p.error('Choose distinct station groups, 0-99')
    selection_path=absolute(a.selection); selection=read_json(selection_path)
    base,configs=validate_lr_selection(selection)
    if base['stations']!=10 or len(set(base['seeds']))!=10 or len(base['seeds'])!=10 or base['eval_samples_per_model']!=5:
        raise ValueError('Require ten stations, ten models, fifty evaluation scenarios')
    temperature_path=selection_path.parent/'frozen_temperature_selection.json'
    if digest(temperature_path)!=selection['experiment']['temperature_selection_sha256']:
        raise ValueError('Frozen temperature selection changed')
    temp_base,_,chosen=validate_selection(read_json(temperature_path))
    if temp_base!=base or chosen!=configs['riesz_softmax']['temperatures']:
        raise ValueError('Temperature provenance mismatch')
    root=absolute(a.workdir); root.mkdir(parents=True,exist_ok=True)
    spec={'lr_selection_sha256':digest(selection_path),'groups':a.groups,'reference_configs':configs,
          'changed_component':'softmax neighbour weights replaced by uniform weights; all other settings held fixed'}
    if (root/'experiment.json').exists() and read_json(root/'experiment.json')!=spec:
        raise ValueError('Workdir belongs to a different selection/config/group set')
    (root/'experiment.json').write_text(json.dumps(spec,indent=2))
    snapshot=root/'frozen_lr_selection.json'
    if snapshot.exists() and snapshot.read_bytes()!=selection_path.read_bytes(): raise ValueError('Frozen snapshot differs')
    snapshot.write_bytes(selection_path.read_bytes())
    temp_snapshot=root/'frozen_temperature_selection.json'
    if temp_snapshot.exists() and temp_snapshot.read_bytes()!=temperature_path.read_bytes(): raise ValueError('Frozen temperature snapshot differs')
    temp_snapshot.write_bytes(temperature_path.read_bytes())
    rows=[]
    print(f'Uniform ablation: fixed LR {configs["riesz_softmax"]["learning_rate"]}; fixed three prefactors {chosen}',flush=True)
    for index,group in enumerate(a.groups):
        print(f'Group {group} ({index+1}/{len(a.groups)})',flush=True)
        data=absolute(a.data_dir)/f'temperature10_group{group}.npz'
        if not data.exists():
            run(['prepare_data.py','--download','--group',str(group),'--source',str(absolute(a.source)),'--output',str(data)])
        data_hash=digest(data); meta=read_json(data.with_suffix('.json'))
        if meta['station_group']!=group or len(meta['stations'])!=10: raise ValueError('Prepared group mismatch')
        cfg=dict(configs['riesz_softmax'],group=group)
        reference_root=absolute(a.reference_runs)/f'group{group:02d}'
        softmax,soft_prov=checked_reference(reference_root/'riesz_softmax/evaluation/metrics.json',cfg,data_hash,'riesz_softmax')
        energy,energy_prov=checked_reference(reference_root/'energy/evaluation/metrics.json',dict(configs['energy'],group=group),data_hash,'energy')
        group_root=root/f'group{group:02d}'; group_root.mkdir(parents=True,exist_ok=True)
        cfg_path=group_root/'config.yaml'; cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        checkpoints,provenance=[],[]
        for seed in base['seeds']:
            work=group_root/f'seed{seed}'
            record=fit_record(work,cfg,seed,data_hash,method='riesz_uniform') if (work/'latest.pt').exists() else None
            if record is None:
                work.mkdir(parents=True,exist_ok=True)
                cmd=['train.py','--data',str(data),'--config',str(cfg_path),'--method','riesz_uniform','--seed',str(seed),
                     '--device',a.device,'--workdir',str(work)]
                if (work/'latest.pt').exists(): cmd.append('--resume')
                print(f'  Uniform seed {seed}: fitting',flush=True)
                run(cmd,work/'uniform_runner.log')
                record=fit_record(work,cfg,seed,data_hash,method='riesz_uniform')
                if record is None: raise ValueError('Uniform fit did not finish')
            checkpoints.append(str(work/'best.pt')); provenance.append(dict(record,seed=seed))
            print(f'  Uniform seed {seed}: ready',flush=True)
        frozen={'uniform':provenance,'softmax':soft_prov,'energy':energy_prov}
        pf=group_root/'checkpoint_provenance.json'
        if pf.exists() and read_json(pf)!=frozen: raise ValueError('A compared checkpoint identity changed')
        pf.write_text(json.dumps(frozen,indent=2))
        evaluation=group_root/'evaluation'
        if not (evaluation/'metrics.json').exists():
            run(['evaluate.py','--data',str(data),'--device',a.device,'--checkpoints',*checkpoints,'--output',str(evaluation)])
        uniform=read_json(evaluation/'metrics.json')
        check_evaluation(uniform,'riesz_uniform',data_hash,base['seeds'],provenance)
        for ref in (energy,softmax):
            if not np.allclose([ref['raw_ensemble_metrics'][m] for m in METRICS],
                               [uniform['raw_ensemble_metrics'][m] for m in METRICS],rtol=1e-10,atol=1e-10):
                raise ValueError('Raw evaluation protocols differ')
        rows.append({'group':group,'stations':meta['stations'],'counts':meta['counts'],'data_sha256':data_hash,
                     'energy':energy['metrics'],'riesz_softmax':softmax['metrics'],
                     'riesz_uniform':uniform['metrics'],'raw':uniform['raw_ensemble_metrics']})
        summary={'complete':len(rows)==len(a.groups),'requested_groups':a.groups,'completed_groups':[r['group'] for r in rows],
                 'groups':rows,'fixed_learning_rate':cfg['learning_rate'],'retained_prefactors':chosen,
                 'lr_selection_sha256':spec['lr_selection_sha256'],'models_per_method':10,'scenarios_per_case':50,
                 'group_mean_scores':{method:{m:float(np.mean([r[method][m] for r in rows])) for m in METRICS}
                                     for method in ('energy','riesz_softmax','riesz_uniform','raw')},
                 'uniform_minus_softmax':{m:[r['riesz_uniform'][m]-r['riesz_softmax'][m] for r in rows] for m in METRICS},
                 'note':'Controlled weight-only diagnostic ablation at inherited settings; repeated/exploratory 2016 test use, no uniform-specific tuning.'}
        tmp=root/'summary.tmp'; tmp.write_text(json.dumps(summary,indent=2)); tmp.replace(root/'summary.json')
        print(f'  Softmax ES {softmax["metrics"]["energy_score"]:.4f}; uniform ES {uniform["metrics"]["energy_score"]:.4f}',flush=True)
    print('\nMethod                  Energy      Variogram       CRPS',flush=True)
    for method,values in summary['group_mean_scores'].items():
        print(f'{method:<22} {values["energy_score"]:9.4f} {values["variogram_score"]:13.4f} {values["mean_crps"]:10.4f}',flush=True)
    print(f'Summary: {root / "summary.json"}',flush=True)


if __name__=='__main__': main()
