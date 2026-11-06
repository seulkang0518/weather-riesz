import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
import yaml

from weather.model import TemperatureCGM
from weather.objectives import training_energy_score, riesz_loss


def load_split(data, split, device):
    return {k: torch.from_numpy(data[f'{split}_{k}']).to(device)
            for k in ('mean', 'std', 'context', 'target')}


def predict(model, inputs, indices, samples, generator):
    z = torch.randn(len(indices), samples, model.latent_dim,
                    device=inputs['mean'].device, generator=generator)
    return model(*(inputs[k][indices] for k in ('mean', 'std', 'context')), z)


def validation_sample_count(cfg):
    count=cfg.get('validation_samples',cfg['train_samples'])
    if not isinstance(count,int) or count<2:
        raise ValueError('validation_samples must be an integer >=2')
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/temperature10_group0.npz')
    p.add_argument('--config', default='configs/temperature10.yaml')
    p.add_argument('--method', choices=['energy', 'riesz_softmax', 'riesz_uniform'], required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--epochs', type=int)
    p.add_argument('--max-batches', type=int, help='Smoke check only; NOT a benchmark run')
    p.add_argument('--workdir', required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    val_samples=validation_sample_count(cfg)
    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if a.device == 'cuda':
        torch.cuda.manual_seed_all(a.seed)
    data = np.load(a.data, allow_pickle=False)
    inputs = {s: load_split(data, s, a.device) for s in ('train', 'val')}
    if inputs['train']['target'].shape[1] != cfg['stations']:
        raise ValueError('Config/data station count mismatch')
    meta_path = Path(a.data).with_suffix('.json')
    if not meta_path.exists():
        raise ValueError('Missing preparation metadata')
    data_meta = json.loads(meta_path.read_text())
    if data_meta['station_group'] != cfg['group']:
        raise ValueError('Config/data station group mismatch')
    work = Path(a.workdir); work.mkdir(parents=True, exist_ok=True)
    if (work / 'manifest.json').exists() and not a.resume:
        raise FileExistsError('Workdir already has a run; use --resume or a new folder')
    model = TemperatureCGM(cfg['stations'], cfg['latent_dim']).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], eps=cfg['adam_epsilon'])
    # Both methods share initial weights, shuffle seed, and training-noise stream.
    generator = torch.Generator(device=a.device).manual_seed(a.seed + 1000)
    shuffle = torch.Generator().manual_seed(a.seed + 2000)
    manifest = {'config': cfg, 'method': a.method, 'seed': a.seed, 'data': str(Path(a.data).resolve()),
                'data_sha256': hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
                'data_metadata': data_meta, 'torch': torch.__version__, 'python': platform.python_version(),
                'parameters': sum(p.numel() for p in model.parameters()),
                'device': a.device, 'smoke_only': bool(a.max_batches),
                'validation_selection': 'fixed-noise training-form energy score for BOTH objectives',
                'port': 'PyTorch architecture/preprocessing port; not bitwise TensorFlow reproduction'}
    start_epoch, best, bad = 0, float('inf'), 0
    if a.resume:
        previous = json.loads((work / 'manifest.json').read_text())
        for key in ('config', 'method', 'seed', 'data_sha256', 'smoke_only'):
            if previous[key] != manifest[key]:
                raise ValueError(f'Resume mismatch: {key}')
        state = torch.load(work / 'latest.pt', map_location=a.device, weights_only=False)
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        generator.set_state(state['generator'].cpu()); shuffle.set_state(state['shuffle'].cpu())
        start_epoch, best, bad = state['epoch'], state['best'], state['bad']
    else:
        (work / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)
    if bad >= cfg['patience']:
        print('Run already finished by early stopping. Use a new workdir for a different experiment.', flush=True)
        return
    start = time.monotonic()
    for epoch in range(start_epoch, a.epochs or cfg['epochs']):
        model.train(); total, count = 0., 0
        permutation = torch.randperm(len(inputs['train']['target']), generator=shuffle)
        for batch, idx in enumerate(permutation.split(cfg['batch_size'])):
            if a.max_batches and batch >= a.max_batches:
                break
            idx = idx.to(a.device)
            opt.zero_grad(set_to_none=True)
            pred = predict(model, inputs['train'], idx, cfg['train_samples'], generator)
            loss = (training_energy_score(pred, inputs['train']['target'][idx]).mean()
                    if a.method == 'energy' else
                    riesz_loss(pred, inputs['train']['target'][idx], cfg['temperatures'],
                               weighting='uniform' if a.method=='riesz_uniform' else 'softmax')[0].mean())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss, epoch {epoch+1}, batch {batch}')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError('Nonfinite parameter gradient')
            opt.step()
            total += loss.item() * len(idx); count += len(idx)
        model.eval(); val_sum, val_count = 0., 0
        vg = torch.Generator(device=a.device).manual_seed(cfg['validation_seed'])
        with torch.no_grad():
            for idx in torch.arange(len(inputs['val']['target']), device=a.device).split(cfg['batch_size']):
                pred = predict(model, inputs['val'], idx, val_samples, vg)
                val_sum += training_energy_score(pred, inputs['val']['target'][idx]).sum().item()
                val_count += len(idx)
        val = val_sum / val_count
        if not np.isfinite(val):
            raise FloatingPointError('Nonfinite validation score')
        improved = val < best
        best, bad = (val, 0) if improved else (best, bad + 1)
        state = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch+1,
                 'best': best, 'bad': bad, 'generator': generator.get_state(),
                 'shuffle': shuffle.get_state(), 'manifest': manifest}
        temp = work / 'latest.tmp'; torch.save(state, temp); temp.replace(work / 'latest.pt')
        if improved:
            torch.save(state, work / 'best.pt')
        row = {'epoch': epoch+1, 'train_loss': total/count, 'val_energy_training_form': val,
               'best_val': best, 'bad_epochs': bad, 'elapsed_seconds': time.monotonic()-start}
        print(json.dumps(row), flush=True)
        with open(work / 'train.jsonl', 'a') as f:
            f.write(json.dumps(row)+'\n')
        if bad >= cfg['patience']:
            break
    print(f'Finished. Best checkpoint: {work / "best.pt"}', flush=True)


if __name__ == '__main__':
    main()
