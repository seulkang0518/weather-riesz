"""Fixed-recipe particle ablation, with 50-scenario evaluation and optional LVS."""
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
from evaluate_lvs import load_forecast
from run_groups import ROOT, METRICS, read_json, run
from train import load_split, predict, validation_sample_count
from tune_temperature import digest, fit_record
from weather.model import TemperatureCGM
from weather.objectives import training_energy_score, riesz_loss
from weather.local_variogram import (local_variogram, nearest_other_stations,
                                    recover_coordinates, seasonal_climatology)

METHODS=('energy','riesz_uniform')
ALL_METHODS=('energy','riesz_softmax','riesz_uniform')
LVS_METRICS=('local_variogram_anomaly','local_variogram_physical')


def particle_config(reference,particles):
    if not isinstance(particles,int) or particles<2:
        raise ValueError('Training particle count must be an integer >=2')
    if reference['train_samples']!=50 or validation_sample_count(reference)!=50:
        raise ValueError('Require reference training/validation with 50 scenarios')
    cfg=dict(reference,train_samples=particles)
    if particles!=50: cfg['validation_samples']=50
    return cfg


def summarize(rows,groups,particles,methods=METHODS,metrics=METRICS):
    scores=[]
    for method in methods:
        for count in particles:
            subset=[r for r in rows if r['method']==method and r['train_particles']==count]
            if subset:
                scores.append({'method':method,'train_particles':count,'groups':len(subset),
                               **{m:float(np.mean([r['metrics'][m] for r in subset])) for m in metrics}})
    expected={(g,m,n) for g in groups for m in methods for n in particles}
    actual=[(r['group'],r['method'],r['train_particles']) for r in rows]
    if (len(set(actual))!=len(actual) or not set(actual).issubset(expected)
            or any(not np.isfinite(r['metrics'][key]) for r in rows for key in metrics)):
        raise ValueError('Invalid/duplicate particle comparison cells')
    return {'complete':set(actual)==expected,'requested_groups':groups,'training_particles':particles,
            'models_per_method_per_group':10,'validation_scenarios_per_model':50,'test_scenarios_per_case':50,
            'group_mean_scores':scores,'cells':rows,
            'note':'Fixed inherited LR/settings, no particle-specific tuning. Previously inspected 2016 test year. Group means use equal weights.'}


def validate_reuse_spec(previous,summary,current):
    """Reuse a complete earlier group subset only, with the same frozen recipe."""
    for key in ('lr_selection_sha256','particles','reference_configs','validation_samples','test_samples'):
        if previous[key]!=current[key]: raise ValueError(f'Particle reuse recipe mismatch: {key}')
    for key,default in (('methods',list(METHODS)),('with_lvs',False),('reference_paths',None)):
        if previous.get(key,default)!=current.get(key,default):
            raise ValueError(f'Particle reuse protocol mismatch: {key}')
    groups=previous['groups']; methods=previous.get('methods',list(METHODS))
    metrics=METRICS+LVS_METRICS if previous.get('with_lvs',False) else METRICS
    if not groups or len(set(groups))!=len(groups) or not all(0<=g<100 for g in groups):
        raise ValueError('Invalid prior particle groups')
    reconstructed=summarize(summary['cells'],groups,previous['particles'],methods,metrics)
    if (not summary['complete'] or not reconstructed['complete'] or summary['requested_groups']!=groups
            or summary['training_particles']!=previous['particles']
            or summary['group_mean_scores']!=reconstructed['group_mean_scores']
            or summary['validation_scenarios_per_model']!=50 or summary['test_scenarios_per_case']!=50):
        raise ValueError('Prior particle comparison is incomplete or inconsistent')
    return {(r['group'],r['method'],r['train_particles']):r for r in summary['cells']}


def reused_particle_cell(cell,cfg,data_hash,method,seeds,reference_provenance):
    """Read-only validation: do not resume/refit or rewrite any prior-run file."""
    saved=read_json(cell/'checkpoint_provenance.json')
    records=[]
    for seed in seeds:
        work=cell/f'seed{seed}'
        record=fit_record(work,cfg,seed,data_hash,method=method)
        if record is None: raise ValueError('Prior particle fit is incomplete; refuse to modify it')
        records.append(dict(record,seed=seed))
    if saved!={'fits':records,'reference_50':reference_provenance}:
        raise ValueError('Prior particle checkpoint identity or M=50 reference changed')
    evaluation=cell/'evaluation'
    result=read_json(evaluation/'metrics.json')
    check_evaluation(result,method,data_hash,seeds,records)
    return evaluation,result


def benchmark_loss(pred,target,method,temperatures):
    if method=='energy': return training_energy_score(pred,target).mean()
    if method not in ALL_METHODS: raise ValueError('Unknown particle comparison method')
    return riesz_loss(pred,target,temperatures,
                      weighting='uniform' if method=='riesz_uniform' else 'softmax')[0].mean()


def cached_lvs(directory,data_path,data_hash,method,output):
    """Keep diagnostics separate from existing forecast outputs, including M=50."""
    identity={'data_sha256':data_hash,'predictions_sha256':digest(directory/'predictions.npz'),
              'metrics_sha256':digest(directory/'metrics.json'), 'method':method,
              'protocol':{'neighbors':5,'exclude_center':True,'power':.5,'window':21,
                          'climatology':'train-only 2007-2014; actual day-of-year, circular 366-day cycle, ddof=0'}}
    if output.exists():
        record=read_json(output)
        if record['identity']!=identity: raise ValueError('Cached LVS input or protocol changed')
        if digest(output.with_suffix('.npz'))!=record['per_case_sha256']:
            raise ValueError('Cached per-case LVS changed')
        return record['metrics']
    with np.load(data_path,allow_pickle=False) as data:
        pred,_=load_forecast(directory,data,data_hash,method)
        meta=read_json(data_path.with_suffix('.json'))
        neighbors=nearest_other_stations(*recover_coordinates(data,meta))
        mean,std,_=seasonal_climatology(data['train_target'],data['train_dates'],data['test_dates'])
        target=data['test_target'].astype(float)
        physical=local_variogram(pred,target,neighbors)
        anomaly=local_variogram((pred-mean[:,None])/std[:,None],(target-mean)/std,neighbors)
        metrics={'local_variogram_anomaly':float(anomaly.mean()),'local_variogram_physical':float(physical.mean())}
        output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output.with_suffix('.npz'),dates=data['test_dates'],anomaly=anomaly,physical=physical,
                            neighbors=neighbors,stations=meta['stations'],climatology_mean=mean,climatology_std=std)
    output.write_text(json.dumps({'identity':identity,'metrics':metrics,
                                 'per_case_sha256':digest(output.with_suffix('.npz'))},indent=2))
    return metrics


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
            loss=benchmark_loss(pred,inputs['target'],method,cfg['temperatures'])
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
    p.add_argument('--softmax-runs',default='runs/temperature_learning_rate_selected_test')
    p.add_argument('--methods',nargs='+',choices=ALL_METHODS,default=list(METHODS))
    p.add_argument('--with-lvs',action='store_true',help='Add source-checked anomaly/physical LVS to each comparison')
    p.add_argument('--groups',nargs='+',type=int,default=list(range(10)))
    p.add_argument('--particles',nargs='+',type=int,default=[50,16,8,4])
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--benchmark-steps',type=int,default=20)
    p.add_argument('--data-dir',default='data')
    p.add_argument('--source',default='data/temperature_data_cgm_std.feather')
    p.add_argument('--workdir',default='runs/temperature_particle_comparison')
    p.add_argument('--reuse-run',help='Read-only reuse of a completed compatible pilot; choose a new workdir')
    a=p.parse_args()
    def absolute(path):
        path=Path(path)
        return (path if path.is_absolute() else ROOT/path).resolve()
    if (not a.groups or len(set(a.groups))!=len(a.groups) or not all(0<=g<100 for g in a.groups)
            or len(set(a.particles))!=len(a.particles) or min(a.particles)<2
            or len(set(a.methods))!=len(a.methods) or a.benchmark_steps<1 or a.threads<1):
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
    references={m:selected['energy' if m=='energy' else 'riesz_softmax'] for m in a.methods}
    for cfg in references.values(): particle_config(cfg,50)
    root=absolute(a.workdir)
    spec={'lr_selection_sha256':digest(selection_path),'groups':a.groups,'particles':a.particles,
          'reference_configs':references,'validation_samples':50,'test_samples':50}
    if tuple(a.methods)!=METHODS or a.with_lvs:
        spec.update(methods=a.methods,with_lvs=a.with_lvs,reference_paths={
            m:str(absolute({'energy':a.energy_runs,'riesz_softmax':a.softmax_runs,'riesz_uniform':a.uniform_runs}[m]))
            for m in a.methods})
    prior_root=None; prior_cells={}; prior_groups=[]; prior_summary=None
    if a.reuse_run:
        prior_root=absolute(a.reuse_run)
        if root==prior_root or root.is_relative_to(prior_root) or prior_root.is_relative_to(root):
            raise ValueError('Reuse source and output must be separate, nonnested directories')
        prior_spec=read_json(prior_root/'experiment.json'); prior_summary=read_json(prior_root/'summary.json')
        prior_cells=validate_reuse_spec(prior_spec,prior_summary,spec)
        if digest(prior_root/'frozen_lr_selection.json')!=spec['lr_selection_sha256']:
            raise ValueError('Prior frozen LR snapshot changed')
        if digest(prior_root/'frozen_temperature_selection.json')!=digest(tp):
            raise ValueError('Prior frozen temperature snapshot changed')
        prior_groups=[g for g in a.groups if g in prior_spec['groups']]
        if not prior_groups: raise ValueError('No requested groups overlap with the reuse source')
        spec['reuse_run']={'directory':str(prior_root),'experiment_sha256':digest(prior_root/'experiment.json'),
                           'summary_sha256':digest(prior_root/'summary.json')}
    if (root/'experiment.json').exists() and read_json(root/'experiment.json')!=spec:
        raise ValueError('Workdir has a different frozen recipe/group/count set')
    root.mkdir(parents=True,exist_ok=True)
    (root/'experiment.json').write_text(json.dumps(spec,indent=2))
    for source,name in [(selection_path,'frozen_lr_selection.json'),(tp,'frozen_temperature_selection.json')]:
        dest=root/name
        if dest.exists() and dest.read_bytes()!=source.read_bytes(): raise ValueError('Frozen snapshot differs')
        dest.write_bytes(source.read_bytes())
    if prior_root:
        for name in ('experiment.json','summary.json'):
            dest=root/('frozen_reuse_'+name)
            if dest.exists() and dest.read_bytes()!=(prior_root/name).read_bytes():
                raise ValueError('Frozen pilot snapshot differs')
            dest.write_bytes((prior_root/name).read_bytes())
    suffix='' if tuple(a.methods)==METHODS else '_'+ '-'.join(a.methods)
    rows=[]; bench_path=root/f'step_benchmark_{a.device}_{a.threads}threads_{a.benchmark_steps}steps{suffix}.json'
    print(f'Training particles {a.particles}; validation fixed at 50; evaluation 10 models x 5 scenarios.',flush=True)
    if prior_root: print(f'Read-only reuse of groups {prior_groups} from {prior_root}',flush=True)
    for index,group in enumerate(a.groups):
        print(f'Group {group} ({index+1}/{len(a.groups)})',flush=True)
        data=absolute(a.data_dir)/f'temperature10_group{group}.npz'
        if not data.exists():
            run(['prepare_data.py','--download','--group',str(group),'--source',str(absolute(a.source)),'--output',str(data)])
        data_hash=digest(data); meta=read_json(data.with_suffix('.json'))
        if meta['station_group']!=group or len(meta['stations'])!=10: raise ValueError('Prepared group mismatch')
        paths={'energy':absolute(a.energy_runs)/f'group{group:02d}/energy/evaluation/metrics.json',
               'riesz_softmax':absolute(a.softmax_runs)/f'group{group:02d}/riesz_softmax/evaluation/metrics.json',
               'riesz_uniform':absolute(a.uniform_runs)/f'group{group:02d}/evaluation/metrics.json'}
        cached={}; reference_prov={}
        for method in a.methods:
            cached[method],reference_prov[method]=checked_reference(paths[method],dict(references[method],group=group),data_hash,method)
        if index==0:
            if not bench_path.exists():
                prior_bench=Path(prior_summary['step_benchmark']) if prior_summary else None
                if prior_bench is not None and not prior_bench.is_absolute(): prior_bench=absolute(prior_bench)
                if prior_bench is not None and group in prior_groups and prior_bench.name==bench_path.name:
                    timings=read_json(prior_bench)
                else:
                    timings={m:benchmark(reference_prov[m][0]['checkpoint'],data,m,a.particles,a.device,a.threads,a.benchmark_steps) for m in a.methods}
                for m in a.methods:
                    if (timings[m]['reference_sha256']!=reference_prov[m][0]['checkpoint_sha256'] or timings[m]['data_sha256']!=data_hash
                            or [r['train_particles'] for r in timings[m]['rows']]!=a.particles
                            or timings[m]['device']!=a.device or timings[m]['cpu_threads']!=a.threads
                            or timings[m]['measured_steps']!=a.benchmark_steps):
                        raise ValueError('Reused step benchmark has another reference/count/data/device protocol')
                bench_path.write_text(json.dumps(timings,indent=2))
            else:
                timings=read_json(bench_path)
                for m in a.methods:
                    if (timings[m]['reference_sha256']!=reference_prov[m][0]['checkpoint_sha256'] or timings[m]['data_sha256']!=data_hash
                            or [r['train_particles'] for r in timings[m]['rows']]!=a.particles):
                        raise ValueError('Cached step benchmark uses different reference/counts/data')
            print('Step benchmark saved to '+str(bench_path),flush=True)
        for method in a.methods:
            for count in a.particles:
                cfg=particle_config(dict(references[method],group=group),count)
                prior=prior_cells.get((group,method,count))
                if prior and prior['data_sha256']!=data_hash: raise ValueError('Prepared data differs from pilot')
                if count==50:
                    result=cached[method]; origin='completed_50_particle_reference'
                    evaluation=paths[method].parent
                    print(f'  {method}, M=50: reusing completed reference',flush=True)
                elif prior is not None:
                    old_cell=prior_root/f'group{group:02d}'/method/f'particles{count}'
                    evaluation,result=reused_particle_cell(old_cell,cfg,data_hash,method,base['seeds'],reference_prov[method])
                    origin='reused_completed_pilot'
                    print(f'  {method}, M={count}: reusing completed pilot fits and forecasts',flush=True)
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
                metrics=dict(result['metrics'])
                if a.with_lvs:
                    metrics.update(cached_lvs(evaluation,data,data_hash,method,
                        root/f'group{group:02d}'/method/f'particles{count}'/'lvs.json'))
                if prior is not None:
                    if prior['stations']!=meta['stations'] or prior['cases']!=result['cases']:
                        raise ValueError('Pilot station/case alignment changed')
                    for key,value in metrics.items():
                        if not np.isclose(value,prior['metrics'][key],rtol=1e-10,atol=1e-10):
                            raise ValueError('Reused forecast scores differ from frozen pilot')
                rows.append({'group':group,'method':method,'train_particles':count,'metrics':metrics,
                             'data_sha256':data_hash,'stations':meta['stations'],'cases':result['cases'],'origin':origin})
                summary=summarize(rows,a.groups,a.particles,a.methods,METRICS+LVS_METRICS if a.with_lvs else METRICS)
                summary['step_benchmark']=str(bench_path)
                summary['fixed_learning_rates']={m:c['learning_rate'] for m,c in references.items()}
                summary['fixed_riesz_temperatures']=chosen
                summary['timing_scope']='isolated training steps; fixed validation overhead and differing stopping epochs excluded'
                if a.with_lvs: summary['lvs_protocol']='same five other neighbors and train-only 21-day anomalies as evaluate_lvs.py'
                if prior_root:
                    summary['reuse_run']=spec['reuse_run']; summary['reused_groups']=prior_groups
                    additional=[g for g in a.groups if g not in prior_groups]
                    summary['additional_groups']=additional
                    if additional:
                        summary['additional_groups_comparison']=summarize(
                            [r for r in rows if r['group'] in additional],additional,a.particles,a.methods,
                            METRICS+LVS_METRICS if a.with_lvs else METRICS)
                tmp=root/'summary.tmp'; tmp.write_text(json.dumps(summary,indent=2)); tmp.replace(root/'summary.json')
                print(f'  {method}, M={count}: ES {result["metrics"]["energy_score"]:.4f}',flush=True)
    print('\nMethod             Train M    Energy     Variogram       CRPS'+('   LVS anomaly   LVS physical' if a.with_lvs else ''),flush=True)
    for row in summary['group_mean_scores']:
        extra=f' {row["local_variogram_anomaly"]:13.6f} {row["local_variogram_physical"]:14.6f}' if a.with_lvs else ''
        print(f'{row["method"]:<18} {row["train_particles"]:>7} {row["energy_score"]:9.4f} {row["variogram_score"]:13.4f} {row["mean_crps"]:10.4f}'+extra,flush=True)
    if summary.get('additional_groups'):
        print('\nAdditional groups only (excluding the reused particle pilot): '+str(summary['additional_groups']),flush=True)
        print('Method             Train M    Energy     Variogram       CRPS'+('   LVS anomaly   LVS physical' if a.with_lvs else ''),flush=True)
        for row in summary['additional_groups_comparison']['group_mean_scores']:
            extra=f' {row["local_variogram_anomaly"]:13.6f} {row["local_variogram_physical"]:14.6f}' if a.with_lvs else ''
            print(f'{row["method"]:<18} {row["train_particles"]:>7} {row["energy_score"]:9.4f} {row["variogram_score"]:13.4f} {row["mean_crps"]:10.4f}'+extra,flush=True)
    print('\nIsolated training-step timing (first group; not total training time):',flush=True)
    for method in a.methods:
        for row in timings[method]['rows']:
            speed=f'{row["step_speedup_vs_50"]:.2f}x' if 'step_speedup_vs_50' in row else 'n/a'
            print(f'{method:<18} M={row["train_particles"]:<3} {1000*row["median_step_seconds"]:.3f} ms; relative to M=50: {speed}',flush=True)
    print(f'Summary: {root / "summary.json"}\nTiming: {bench_path}',flush=True)


if __name__=='__main__': main()
