"""Fixed-recipe particle ablation: energy vs uniform Riesz, with 50-scenario evaluation."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
import yaml

from compare_uniform import checked_reference
from evaluate_learning_rate import validate_lr_selection, check_evaluation
from evaluate_selected import validate_selection
from run_groups import ROOT, METRICS, read_json, run
from train import load_split, predict, validation_sample_count
from tune_temperature import digest, fit_record
from weather.model import TemperatureCGM
from weather.objectives import training_energy_score, riesz_loss

METHODS=('energy','riesz_uniform')


def particle_config(reference,particles):
    if not isinstance(particles,int) or particles<2:
        raise ValueError('Training particle count must be an integer >=2')
    if reference['train_samples']!=50 or validation_sample_count(reference)!=50:
        raise ValueError('Require reference training/validation with 50 scenarios')
    cfg=dict(reference,train_samples=particles)
    if particles!=50: cfg['validation_samples']=50
    return cfg


def summarize(rows,groups,particles):
    scores=[]
    for method in METHODS:
        for count in particles:
            subset=[r for r in rows if r['method']==method and r['train_particles']==count]
            if subset:
                scores.append({'method':method,'train_particles':count,'groups':len(subset),
                               **{m:float(np.mean([r['metrics'][m] for r in subset])) for m in METRICS}})
    expected={(g,m,n) for g in groups for m in METHODS for n in particles}
    actual=[(r['group'],r['method'],r['train_particles']) for r in rows]
    if len(set(actual))!=len(actual) or not set(actual).issubset(expected):
        raise ValueError('Invalid/duplicate particle comparison cells')
    return {'complete':set(actual)==expected,'requested_groups':groups,'training_particles':particles,
            'models_per_method_per_group':10,'validation_scenarios_per_model':50,'test_scenarios_per_case':50,
            'group_mean_scores':scores,'cells':rows,
            'note':'Fixed inherited LR/settings, no particle-specific tuning. Previously inspected 2016 test year. Group means use equal weights.'}


def benchmark(reference,data_path,method,counts,device,threads,steps):
    """Current-machine forward/loss/backward/Adam timings, excluding validation/I/O."""
    state=torch.load(reference,map_location='cpu',weights_only=False)
    cfg=state['manifest']['config']; arrays=np.load(data_path,allow_pickle=False)
    b=min(cfg['batch_size'],len(arrays['train_target']))
    inputs={k:torch.from_numpy(arrays[f'train_{k}'][:b].copy()).to(device)
            for k in ('mean','std','context','target')}
    def sync():
        if device=='cuda': torch.cuda.synchronize()
    rows=[]
    for count in counts:
        model=TemperatureCGM(cfg['stations'],cfg['latent_dim']).to(device)
        model.load_state_dict(state['model']); model.train()
        opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],eps=cfg['adam_epsilon'])
        gen=torch.Generator(device=device).manual_seed(20261106)
        timings=[]
        if device=='cuda': torch.cuda.reset_peak_memory_stats()
        for step in range(steps+3):
            sync(); start=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            z=torch.randn(b,count,cfg['latent_dim'],device=device,generator=gen)
            pred=model(*(inputs[k] for k in ('mean','std','context')),z)
            loss=(training_energy_score(pred,inputs['target']).mean() if method=='energy' else
                  riesz_loss(pred,inputs['target'],cfg['temperatures'],weighting='uniform')[0].mean())
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite benchmark loss')
            loss.backward(); opt.step(); sync()
            if step>=3: timings.append(time.perf_counter()-start)
        rows.append({'train_particles':count,'median_step_seconds':float(np.median(timings)),
                     'mean_step_seconds':float(np.mean(timings)),
                     'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None})
        del loss,pred,z,opt,model
    baseline=next((r['median_step_seconds'] for r in rows if r['train_particles']==50),None)
    if baseline is not None:
        for r in rows: r['step_speedup_vs_50']=baseline/r['median_step_seconds']
    return {'method':method,'device':device,'device_label':torch.cuda.get_device_name() if device=='cuda' else platform.machine(),
            'torch':torch.__version__,'cpu_threads':threads,'cases_per_batch':b,'warmup_steps':3,'measured_steps':steps,
            'reference_checkpoint':str(Path(reference).resolve()),'reference_sha256':digest(reference),
            'data_sha256':digest(data_path),'rows':rows,
            'scope':'isolated training-step benchmark, same batch/model initialization per count; excludes data loading, validation/checkpoints; not total-fit speedup'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--selection',default='runs/temperature_learning_rate_sweep/lr_selection.json')
    p.add_argument('--energy-runs',default='runs/temperature_learning_rate_selected_test')
    p.add_argument('--uniform-runs',default='runs/temperature_uniform_comparison')
    p.add_argument('--groups',nargs='+',type=int,default=list(range(10)))
    p.add_argument('--particles',nargs='+',type=int,default=[50,16,8,4])
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--benchmark-steps',type=int,default=20)
    p.add_argument('--data-dir',default='data')
    p.add_argument('--source',default='data/temperature_data_cgm_std.feather')
    p.add_argument('--workdir',default='runs/temperature_particle_comparison')
    a=p.parse_args()
    def absolute(path):
        path=Path(path)
        return (path if path.is_absolute() else ROOT/path).resolve()
    if (len(set(a.groups))!=len(a.groups) or not all(0<=g<100 for g in a.groups)
            or len(set(a.particles))!=len(a.particles) or min(a.particles)<2
            or a.benchmark_steps<1 or a.threads<1):
        p.error('Use distinct valid groups/counts >=2, positive timing steps and CPU threads')
    torch.set_num_threads(a.threads)
    selection_path=absolute(a.selection); selection=read_json(selection_path)
    base,selected=validate_lr_selection(selection)
    if base['stations']!=10 or len(base['seeds'])!=10 or len(set(base['seeds']))!=10 or base['eval_samples_per_model']!=5:
        raise ValueError('Require ten stations, ten seeds, fifty final scenarios')
    tp=selection_path.parent/'frozen_temperature_selection.json'
    if digest(tp)!=selection['experiment']['temperature_selection_sha256']: raise ValueError('Temperature selection changed')
    tb,_,chosen=validate_selection(read_json(tp))
    if tb!=base or chosen!=selected['riesz_softmax']['temperatures']: raise ValueError('Temperature provenance mismatch')
    references={'energy':selected['energy'],'riesz_uniform':selected['riesz_softmax']}
    for cfg in references.values(): particle_config(cfg,50)
    root=absolute(a.workdir); root.mkdir(parents=True,exist_ok=True)
    spec={'lr_selection_sha256':digest(selection_path),'groups':a.groups,'particles':a.particles,
          'reference_configs':references,'validation_samples':50,'test_samples':50}
    if (root/'experiment.json').exists() and read_json(root/'experiment.json')!=spec:
        raise ValueError('Workdir has a different frozen recipe/group/count set')
    (root/'experiment.json').write_text(json.dumps(spec,indent=2))
    for source,name in [(selection_path,'frozen_lr_selection.json'),(tp,'frozen_temperature_selection.json')]:
        dest=root/name
        if dest.exists() and dest.read_bytes()!=source.read_bytes(): raise ValueError('Frozen snapshot differs')
        dest.write_bytes(source.read_bytes())
    rows=[]; bench_path=root/f'step_benchmark_{a.device}_{a.threads}threads_{a.benchmark_steps}steps.json'
    print(f'Training particles {a.particles}; validation fixed at 50; evaluation 10 models x 5 scenarios.',flush=True)
    for index,group in enumerate(a.groups):
        print(f'Group {group} ({index+1}/{len(a.groups)})',flush=True)
        data=absolute(a.data_dir)/f'temperature10_group{group}.npz'
        if not data.exists():
            run(['prepare_data.py','--download','--group',str(group),'--source',str(absolute(a.source)),'--output',str(data)])
        data_hash=digest(data); meta=read_json(data.with_suffix('.json'))
        if meta['station_group']!=group or len(meta['stations'])!=10: raise ValueError('Prepared group mismatch')
        paths={'energy':absolute(a.energy_runs)/f'group{group:02d}/energy/evaluation/metrics.json',
               'riesz_uniform':absolute(a.uniform_runs)/f'group{group:02d}/evaluation/metrics.json'}
        cached={}; reference_prov={}
        for method in METHODS:
            cached[method],reference_prov[method]=checked_reference(paths[method],dict(references[method],group=group),data_hash,method)
        if index==0:
            if not bench_path.exists():
                timings={m:benchmark(reference_prov[m][0]['checkpoint'],data,m,a.particles,a.device,a.threads,a.benchmark_steps) for m in METHODS}
                bench_path.write_text(json.dumps(timings,indent=2))
            else:
                timings=read_json(bench_path)
                for m in METHODS:
                    if (timings[m]['reference_sha256']!=reference_prov[m][0]['checkpoint_sha256'] or timings[m]['data_sha256']!=data_hash
                            or [r['train_particles'] for r in timings[m]['rows']]!=a.particles):
                        raise ValueError('Cached step benchmark uses different reference/counts/data')
            print('Step benchmark saved to '+str(bench_path),flush=True)
        for method in METHODS:
            for count in a.particles:
                cfg=particle_config(dict(references[method],group=group),count)
                if count==50:
                    result=cached[method]; origin='completed_50_particle_reference'
                    print(f'  {method}, M=50: reusing completed reference',flush=True)
                else:
                    cell=root/f'group{group:02d}'/method/f'particles{count}'
                    cell.mkdir(parents=True,exist_ok=True); cp=cell/'config.yaml'
                    cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
                    checkpoints,provenance=[],[]
                    for seed in base['seeds']:
                        work=cell/f'seed{seed}'
                        record=fit_record(work,cfg,seed,data_hash,method=method) if (work/'latest.pt').exists() else None
                        if record is None:
                            work.mkdir(parents=True,exist_ok=True)
                            cmd=['train.py','--data',str(data),'--config',str(cp),'--method',method,'--seed',str(seed),
                                 '--device',a.device,'--threads',str(a.threads),'--workdir',str(work)]
                            if (work/'latest.pt').exists(): cmd.append('--resume')
                            print(f'  {method}, M={count}, seed {seed}: fitting',flush=True)
                            run(cmd,work/'particle_runner.log')
                            record=fit_record(work,cfg,seed,data_hash,method=method)
                            if record is None: raise ValueError('Fit did not finish')
                        checkpoints.append(str(work/'best.pt')); provenance.append(dict(record,seed=seed))
                    frozen={'fits':provenance,'reference_50':reference_prov[method]}
                    pf=cell/'checkpoint_provenance.json'
                    if pf.exists() and read_json(pf)!=frozen: raise ValueError('Checkpoint identities changed')
                    pf.write_text(json.dumps(frozen,indent=2))
                    evaluation=cell/'evaluation'
                    if not (evaluation/'metrics.json').exists():
                        run(['evaluate.py','--data',str(data),'--device',a.device,'--checkpoints',*checkpoints,'--output',str(evaluation)])
                    result=read_json(evaluation/'metrics.json')
                    check_evaluation(result,method,data_hash,base['seeds'],provenance)
                    if not np.allclose([result['raw_ensemble_metrics'][m] for m in METRICS],
                                       [cached[method]['raw_ensemble_metrics'][m] for m in METRICS],rtol=1e-10,atol=1e-10):
                        raise ValueError('Raw evaluation protocol differs')
                    origin='new_particle_count_fit'
                rows.append({'group':group,'method':method,'train_particles':count,'metrics':result['metrics'],
                             'data_sha256':data_hash,'stations':meta['stations'],'cases':result['cases'],'origin':origin})
                summary=summarize(rows,a.groups,a.particles); summary['step_benchmark']=str(bench_path)
                summary['fixed_learning_rates']={m:c['learning_rate'] for m,c in references.items()}
                tmp=root/'summary.tmp'; tmp.write_text(json.dumps(summary,indent=2)); tmp.replace(root/'summary.json')
                print(f'  {method}, M={count}: ES {result["metrics"]["energy_score"]:.4f}',flush=True)
    print('\nMethod             Train M    Energy     Variogram       CRPS',flush=True)
    for row in summary['group_mean_scores']:
        print(f'{row["method"]:<18} {row["train_particles"]:>7} {row["energy_score"]:9.4f} {row["variogram_score"]:13.4f} {row["mean_crps"]:10.4f}',flush=True)
    print(f'Summary: {root / "summary.json"}\nTiming: {bench_path}',flush=True)


if __name__=='__main__': main()
