"""Evaluate independently frozen LR choices on the original ten test groups."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from evaluate_selected import validate_selection, verify_baseline
from run_groups import ROOT, METRICS, aggregate, read_json, run, verify_evaluation
from tune_learning_rate import METHODS, rank_learning_rates
from tune_temperature import digest, fit_record


def validate_lr_selection(selection):
    if selection.get('kind')!='learning_rate' or selection['test_evaluated_by_this_sweep'] is not False:
        raise ValueError('Require a frozen validation-only LR selection')
    exp=selection['experiment']; base=exp['base_config']
    if not all(math.isfinite(lr) and lr>0 for lr in exp['learning_rates']):
        raise ValueError('Invalid learning-rate grid')
    rankings=rank_learning_rates(selection['trials'],exp['groups'],exp['seeds'],exp['learning_rates'])
    configs={}
    for method in METHODS:
        winner=rankings[method][0]['learning_rate']
        expected=dict(base,learning_rate=winner)
        if method=='riesz_softmax': expected['temperatures']=exp['fixed_riesz_temperatures']
        saved=selection['methods'][method]
        if saved['learning_rate']!=winner or saved['ranking']!=rankings[method] or saved['config']!=expected:
            raise ValueError(f'{method} choice does not match the complete validation ranking')
        configs[method]=expected
    return base,configs


def check_evaluation(result,method,data_hash,seeds,provenance):
    verify_evaluation(result,method,data_hash,seeds)
    identity={r['seed']:(str(Path(r['checkpoint']).resolve()),r['best_epoch'],r['validation_energy']) for r in provenance}
    for record in result['models']:
        if (record['checkpoint'],record['epoch'],record['val_energy_training_form'])!=identity[record['seed']]:
            raise ValueError('Evaluation does not match selected checkpoint identities')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--selection',default='runs/temperature_learning_rate_sweep/lr_selection.json')
    p.add_argument('--groups',nargs='+',type=int,default=list(range(10)))
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--data-dir',default='data')
    p.add_argument('--source',default='data/temperature_data_cgm_std.feather')
    p.add_argument('--existing-runs',default='runs/temperature10_groups0_9')
    p.add_argument('--group0-run',default='runs/temperature10_group0')
    p.add_argument('--temperature-test-runs',default='runs/temperature_extended_selected_test')
    p.add_argument('--workdir',default='runs/temperature_learning_rate_selected_test')
    a=p.parse_args()
    def absolute(path):
        path=Path(path)
        return (path if path.is_absolute() else ROOT/path).resolve()
    if len(set(a.groups))!=len(a.groups) or not all(0<=g<100 for g in a.groups):
        p.error('Choose distinct official station groups, 0-99')
    selection_path=absolute(a.selection); selection=read_json(selection_path)
    base,configs=validate_lr_selection(selection)
    if base['stations']!=10 or len(base['seeds'])!=10 or len(set(base['seeds']))!=10 or base['eval_samples_per_model']!=5:
        raise ValueError('Require ten stations, ten seeds per method, fifty final scenarios')
    temperature_path=selection_path.parent/'frozen_temperature_selection.json'
    if digest(temperature_path)!=selection['experiment']['temperature_selection_sha256']:
        raise ValueError('Underlying frozen temperature selection changed')
    temp_base,_,chosen=validate_selection(read_json(temperature_path))
    if temp_base!=base or chosen!=selection['experiment']['fixed_riesz_temperatures']:
        raise ValueError('Fixed temperatures do not match the underlying validation choice')
    root=absolute(a.workdir); root.mkdir(parents=True,exist_ok=True)
    spec={'lr_selection_sha256':digest(selection_path),'groups':a.groups,'selected_configs':configs}
    if (root/'experiment.json').exists() and read_json(root/'experiment.json')!=spec:
        raise ValueError('Workdir belongs to another frozen selection/group set')
    (root/'experiment.json').write_text(json.dumps(spec,indent=2))
    for source,name in [(selection_path,'frozen_lr_selection.json'),(temperature_path,'frozen_temperature_selection.json')]:
        dest=root/name
        if dest.exists() and dest.read_bytes()!=source.read_bytes(): raise ValueError('Frozen snapshot differs')
        dest.write_bytes(source.read_bytes())
    prior={(r['method'],r['group'],r['seed']):r for r in selection['trials']
           if r['learning_rate']==configs[r['method']]['learning_rate']}
    print('Frozen learning rates: '+str({m:c['learning_rate'] for m,c in configs.items()}),flush=True)
    print('Fixed Riesz temperatures: '+str(chosen),flush=True)
    rows=[]
    for index,group in enumerate(a.groups):
        print(f'Group {group} ({index+1}/{len(a.groups)})',flush=True)
        data=absolute(a.data_dir)/f'temperature10_group{group}.npz'
        if not data.exists():
            run(['prepare_data.py','--download','--group',str(group),'--source',str(absolute(a.source)),'--output',str(data)])
        data_hash=digest(data); meta=read_json(data.with_suffix('.json'))
        if meta['station_group']!=group or len(meta['stations'])!=10: raise ValueError('Data group mismatch')
        if str(group) in selection['data_hashes'] and selection['data_hashes'][str(group)]!=data_hash:
            raise ValueError('Prepared data differs from LR validation sweep')
        original=absolute(a.group0_run) if group==0 else absolute(a.existing_runs)/f'group{group:02d}'
        original_energy=verify_baseline(original,dict(base,group=group),data_hash,'energy')
        temp_result=read_json(absolute(a.temperature_test_runs)/f'group{group:02d}/evaluation/metrics.json')
        verify_evaluation(temp_result,'riesz_softmax',data_hash,base['seeds'])
        temp_records={}
        for model in temp_result['models']:
            work=absolute(model['checkpoint']).parent
            record=fit_record(work,dict(base,group=group,temperatures=chosen),model['seed'],data_hash)
            if record is None: raise ValueError('Earlier fixed-temperature fit is incomplete')
            temp_records[model['seed']]=dict(record,seed=model['seed'])
        check_evaluation(temp_result,'riesz_softmax',data_hash,base['seeds'],list(temp_records.values()))
        group_root=root/f'group{group:02d}'; group_root.mkdir(parents=True,exist_ok=True)
        results={}
        for method in METHODS:
            cfg=dict(configs[method],group=group)
            method_root=group_root/method; method_root.mkdir(parents=True,exist_ok=True)
            cfg_path=method_root/'config.yaml'; cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
            checkpoints,provenance=[],[]
            for seed in base['seeds']:
                trial=prior.get((method,group,seed)); record=None
                if trial is not None:
                    work=absolute(trial['checkpoint']).parent
                    if digest(work/'best.pt')!=trial['checkpoint_sha256']: raise ValueError('Frozen LR checkpoint changed')
                    record=fit_record(work,cfg,seed,data_hash,method=method)
                    if record is None or record['validation_energy']!=trial['validation_energy']:
                        raise ValueError('Selected checkpoint differs from LR validation record')
                    origin='frozen_lr_sweep'
                elif cfg['learning_rate']==base['learning_rate']:
                    work=(original/'energy'/f'seed{seed}' if method=='energy'
                          else absolute(temp_records[seed]['checkpoint']).parent)
                    record=fit_record(work,cfg,seed,data_hash,method=method)
                    if record is None: raise ValueError('Cached original fit is incomplete')
                    origin='completed_previous_fit'
                else:
                    work=method_root/f'seed{seed}'; origin='additional_selected_lr_fit'
                    if (work/'latest.pt').exists(): record=fit_record(work,cfg,seed,data_hash,method=method)
                    if record is None:
                        work.mkdir(parents=True,exist_ok=True)
                        cmd=['train.py','--data',str(data),'--config',str(cfg_path),'--method',method,
                             '--seed',str(seed),'--device',a.device,'--workdir',str(work)]
                        if (work/'latest.pt').exists(): cmd.append('--resume')
                        print(f'  {method} seed {seed}: fitting',flush=True)
                        run(cmd,work/'lr_test_runner.log')
                        record=fit_record(work,cfg,seed,data_hash,method=method)
                        if record is None: raise ValueError('Fit did not complete')
                checkpoints.append(str(work/'best.pt')); provenance.append(dict(record,seed=seed,origin=origin))
                print(f'  {method} seed {seed}: ready ({origin})',flush=True)
            pf=method_root/'checkpoint_provenance.json'
            if pf.exists() and read_json(pf)!=provenance: raise ValueError('Checkpoint identities changed')
            pf.write_text(json.dumps(provenance,indent=2))
            evaluation=method_root/'evaluation'
            if not (evaluation/'metrics.json').exists():
                run(['evaluate.py','--data',str(data),'--device',a.device,'--checkpoints',*checkpoints,'--output',str(evaluation)])
            result=read_json(evaluation/'metrics.json')
            check_evaluation(result,method,data_hash,base['seeds'],provenance)
            for reference in (original_energy,temp_result):
                if not np.allclose([reference['raw_ensemble_metrics'][m] for m in METRICS],
                                   [result['raw_ensemble_metrics'][m] for m in METRICS],rtol=1e-10,atol=1e-10):
                    raise ValueError('Raw forecast scores differ across evaluations')
            results[method]=result
        rows.append({'group':group,'stations':meta['stations'],'counts':meta['counts'],'data_sha256':data_hash,
                     'energy':results['energy']['metrics'],'riesz_softmax':results['riesz_softmax']['metrics'],
                     'energy_before_lr':original_energy['metrics'],'riesz_before_lr':temp_result['metrics'],
                     'raw':original_energy['raw_ensemble_metrics']})
        summary=aggregate(rows,a.groups)
        for label in ('energy_before_lr','riesz_before_lr'):
            summary['group_mean_scores'][label]={m:float(np.mean([r[label][m] for r in rows])) for m in METRICS}
        summary.update(lr_selection_sha256=spec['lr_selection_sha256'],selected_learning_rates={m:c['learning_rate'] for m,c in configs.items()},
                       fixed_riesz_temperatures=chosen,note='Independent frozen LR choices; prior test year already inspected; Riesz also has prior temperature tuning.')
        tmp=root/'summary.tmp'; tmp.write_text(json.dumps(summary,indent=2)); tmp.replace(root/'summary.json')
        print(f'  Tuned energy ES {results["energy"]["metrics"]["energy_score"]:.4f}; tuned Riesz ES {results["riesz_softmax"]["metrics"]["energy_score"]:.4f}',flush=True)
    print('\nMethod                  Energy      Variogram       CRPS',flush=True)
    for method,label in [('energy_before_lr','energy_before_lr'),('riesz_before_lr','riesz_before_lr'),
                         ('energy','energy_tuned_lr'),('riesz_softmax','riesz_tuned_lr'),('raw','raw')]:
        score=summary['group_mean_scores'][method]
        print(f'{label:<22} {score["energy_score"]:9.4f} {score["variogram_score"]:13.4f} {score["mean_crps"]:10.4f}',flush=True)
    print(f'Summary: {root / "summary.json"}',flush=True)


if __name__=='__main__':
    main()
