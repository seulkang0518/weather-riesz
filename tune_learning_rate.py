"""Equal-budget validation-only LR search for energy and fixed-temperature Riesz."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

from evaluate_selected import validate_selection
from run_groups import ROOT
from tune_temperature import digest, fit_record

METHODS = ('energy','riesz_softmax')


def rank_learning_rates(trials, groups, seeds, rates):
    expected = {(m,g,s,lr) for m in METHODS for g in groups for s in seeds for lr in rates}
    actual = [(r['method'],r['group'],r['seed'],r['learning_rate']) for r in trials]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Require the complete, nonduplicated, equal-budget LR trial grid')
    if not all(math.isfinite(r['validation_energy']) for r in trials):
        raise ValueError('Nonfinite validation score')
    ranking = {}
    for method in METHODS:
        rows = []
        for lr in rates:
            per_group = {str(g):float(np.mean([r['validation_energy'] for r in trials
                         if r['method']==method and r['learning_rate']==lr and r['group']==g])) for g in groups}
            rows.append({'learning_rate':lr,'mean_validation_energy':float(np.mean(list(per_group.values()))),
                         'per_group_validation_energy':per_group})
        ranking[method] = sorted(rows,key=lambda r:r['mean_validation_energy'])
    return ranking


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--temperature-selection',default='runs/temperature_validation_extended/selection.json')
    p.add_argument('--groups',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--learning-rates',nargs='+',type=float,default=[.001,.0005,.002])
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--data-dir',default='data')
    p.add_argument('--source',default='data/temperature_data_cgm_std.feather')
    p.add_argument('--existing-runs',default='runs/temperature10_groups0_9')
    p.add_argument('--group0-run',default='runs/temperature10_group0')
    p.add_argument('--workdir',default='runs/temperature_learning_rate_sweep')
    a = p.parse_args()
    def absolute(path):
        path=Path(path)
        return (path if path.is_absolute() else ROOT/path).resolve()
    temperature_path=absolute(a.temperature_selection)
    temperature_selection=json.loads(temperature_path.read_text())
    base, multiplier, chosen=validate_selection(temperature_selection)
    if (base['stations']!=10 or len(set(a.groups))!=len(a.groups)
            or not all(0<=g<100 for g in a.groups) or len(set(a.seeds))!=len(a.seeds)
            or not set(a.seeds).issubset(base['seeds']) or len(set(a.learning_rates))!=len(a.learning_rates)
            or not all(math.isfinite(lr) and lr>0 for lr in a.learning_rates)
            or len({f'{lr:.17g}' for lr in a.learning_rates})!=len(a.learning_rates)):
        p.error('Require distinct valid station groups/seeds and distinct positive finite learning rates')
    root=absolute(a.workdir); root.mkdir(parents=True,exist_ok=True)
    spec={'base_config':base,'groups':a.groups,'seeds':a.seeds,'learning_rates':a.learning_rates,
          'fixed_riesz_temperatures':chosen,'temperature_selection_sha256':digest(temperature_path),
          'selection':'per-method equal-group mean of best per-seed 2015 validation energy score',
          'test_use':'none in this sweep; 2016 has already been inspected in earlier experiments'}
    spec_path=root/'experiment.json'
    if spec_path.exists() and json.loads(spec_path.read_text())!=spec:
        raise ValueError('Workdir belongs to a different frozen temperature/grid/config; use a new workdir')
    spec_path.write_text(json.dumps(spec,indent=2))
    snapshot=root/'frozen_temperature_selection.json'
    if snapshot.exists() and snapshot.read_bytes()!=temperature_path.read_bytes():
        raise ValueError('Frozen temperature selection changed')
    snapshot.write_bytes(temperature_path.read_bytes())
    prior={(r['group'],r['seed']):r for r in temperature_selection['trials'] if r['multiplier']==multiplier}
    data_hashes,trials={},[]
    print(f'Fixed Riesz temperatures: {chosen}. Both methods use the same LR grid/groups/seeds.',flush=True)
    for group in a.groups:
        data=absolute(a.data_dir)/f'temperature10_group{group}.npz'
        if not data.exists():
            subprocess.run([sys.executable,'-u','prepare_data.py','--download','--group',str(group),
                            '--source',str(absolute(a.source)),'--output',str(data)],cwd=ROOT,check=True)
        meta=json.loads(data.with_suffix('.json').read_text())
        if meta['station_group']!=group or len(meta['stations'])!=10:
            raise ValueError('Prepared station group mismatch')
        data_hashes[str(group)]=data_hash=digest(data)
        if str(group) in temperature_selection['data_hashes'] and temperature_selection['data_hashes'][str(group)]!=data_hash:
            raise ValueError('Prepared data differs from temperature validation sweep')
        for method in METHODS:
            for lr in a.learning_rates:
                cfg=dict(base,group=group,learning_rate=lr)
                if method=='riesz_softmax': cfg['temperatures']=chosen
                label=f'lr_{lr:.17g}'.replace('.','p')
                cfg_path=root/f'group{group:02d}_{method}_{label}.yaml'
                cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
                for seed in a.seeds:
                    work=root/method/label/f'group{group:02d}'/f'seed{seed}'
                    frozen_trial=None; origin='new_or_resumed_lr_fit'
                    if lr==base['learning_rate']:
                        if method=='energy' or multiplier==1:
                            original=absolute(a.group0_run) if group==0 else absolute(a.existing_runs)/f'group{group:02d}'
                            original=original/method/f'seed{seed}'
                            if (original/'latest.pt').exists():
                                work=original; origin='completed_original_fit'
                        elif (group,seed) in prior:
                            frozen_trial=prior[(group,seed)]
                            work=absolute(frozen_trial['checkpoint']).parent
                            if digest(work/'best.pt')!=frozen_trial['checkpoint_sha256']:
                                raise ValueError('Frozen temperature-sweep checkpoint changed')
                            origin='frozen_temperature_sweep_fit'
                    record=fit_record(work,cfg,seed,data_hash,method=method) if (work/'latest.pt').exists() else None
                    if frozen_trial is not None and (record is None or record['validation_energy']!=frozen_trial['validation_energy']):
                        raise ValueError('Earlier Riesz fit no longer matches frozen validation trial')
                    if record is None:
                        if origin!='new_or_resumed_lr_fit':
                            raise ValueError('Refuse to modify an incomplete original fit; finish it separately')
                        work.mkdir(parents=True,exist_ok=True)
                        command=[sys.executable,'-u','train.py','--data',str(data),'--config',str(cfg_path),
                                 '--method',method,'--seed',str(seed),'--device',a.device,'--workdir',str(work)]
                        if (work/'latest.pt').exists(): command.append('--resume')
                        print(f'Group {group}, {method}, LR {lr:g}, seed {seed}: fitting (validation only)',flush=True)
                        with open(work/'lr_runner.log','a') as f:
                            subprocess.run(command,cwd=ROOT,check=True,stdout=f,stderr=subprocess.STDOUT)
                        record=fit_record(work,cfg,seed,data_hash,method=method)
                        if record is None: raise ValueError('Fit did not complete its training budget')
                    trials.append(dict(record,method=method,group=group,seed=seed,learning_rate=lr,origin=origin))
                    print(f'Group {group}, {method}, LR {lr:g}, seed {seed}: val ES {record["validation_energy"]:.4f} ({origin})',flush=True)
                    tmp=root/'progress.tmp'; tmp.write_text(json.dumps({'experiment':spec,'trials':trials},indent=2))
                    tmp.replace(root/'progress.json')
    rankings=rank_learning_rates(trials,a.groups,a.seeds,a.learning_rates)
    selections={}
    for method in METHODS:
        winner=rankings[method][0]['learning_rate']
        cfg=dict(base,learning_rate=winner)
        if method=='riesz_softmax': cfg['temperatures']=chosen
        selections[method]={'learning_rate':winner,'config':cfg,'ranking':rankings[method]}
        (root/f'selected_{method}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    selection={'kind':'learning_rate','experiment':spec,'data_hashes':data_hashes,
               'methods':selections,'trials':trials,'test_evaluated_by_this_sweep':False,
               'scope':'equal LR budget; Riesz additionally has prior temperature tuning; exploratory follow-up'}
    frozen=root/'lr_selection.json'
    if frozen.exists() and json.loads(frozen.read_text())!=selection:
        raise ValueError('Frozen LR selection would change; use a new workdir')
    frozen.write_text(json.dumps(selection,indent=2))
    print('\nMethod           LR           Mean 2015 validation ES',flush=True)
    for method in METHODS:
        for row in rankings[method]:
            print(f'{method:<16} {row["learning_rate"]:<12g} {row["mean_validation_energy"]:.6f}',flush=True)
        print(f'Selected {method} LR: {selections[method]["learning_rate"]:g}',flush=True)
    print(f'Selection: {frozen}\nNo test evaluation was run.',flush=True)


if __name__=='__main__':
    main()
