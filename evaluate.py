"""Evaluate 10 fitted models x 5 scenarios, or clearly marked pilot subsets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train import load_split, predict
from weather.metrics import scores
from weather.model import TemperatureCGM


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--data', default='data/temperature10_group0.npz')
    p.add_argument('--output', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--samples-per-model', type=int, default=5)
    p.add_argument('--allow-pilot', action='store_true')
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    if a.samples_per_model < 1:
        p.error('samples-per-model must be positive')
    if len(a.checkpoints) != 10 or a.samples_per_model != 5:
        if not a.allow_pilot:
            p.error('Paper protocol requires 10 checkpoints x 5 scenarios. Use --allow-pilot for diagnostics')
    torch.set_num_threads(a.threads)
    data = np.load(a.data, allow_pickle=False)
    inputs = load_split(data, 'test', a.device)
    data_hash = hashlib.sha256(Path(a.data).read_bytes()).hexdigest()
    samples, records, methods, seeds = [], [], set(), set()
    for checkpoint in a.checkpoints:
        state = torch.load(checkpoint, map_location=a.device, weights_only=False)
        manifest = state['manifest']; cfg = manifest['config']
        if manifest['data_sha256'] != data_hash:
            raise ValueError('Checkpoint was trained on different prepared data')
        if manifest['seed'] in seeds:
            raise ValueError('Duplicate training seeds in ensemble')
        if manifest['smoke_only'] and not a.allow_pilot:
            raise ValueError('Smoke checkpoint is not a benchmark model')
        seeds.add(manifest['seed']); methods.add(manifest['method'])
        model = TemperatureCGM(cfg['stations'], cfg['latent_dim']).to(a.device)
        model.load_state_dict(state['model']); model.eval()
        generator = torch.Generator(device=a.device).manual_seed(cfg['test_seed'] + manifest['seed'])
        parts = []
        with torch.no_grad():
            for idx in torch.arange(len(inputs['target']), device=a.device).split(64):
                parts.append(predict(model, inputs, idx, a.samples_per_model, generator).cpu())
        samples.append(torch.cat(parts))
        records.append({'checkpoint': str(Path(checkpoint).resolve()), 'seed': manifest['seed'],
                        'epoch': state['epoch'], 'val_energy_training_form': state['best'],
                        'smoke_only': manifest['smoke_only']})
    if len(methods) != 1:
        raise ValueError('Do not mix methods into one ensemble')
    pred = torch.cat(samples, dim=1)
    target = inputs['target'].cpu()
    result_scores = scores(pred, target)
    raw_scores = scores(torch.from_numpy(data['test_raw']), target)
    summary = {'method': next(iter(methods)), 'cases': len(target), 'stations': target.shape[1],
               'scenarios': pred.shape[1], 'models': records,
               'scope': 'one station group (not the paper average over 100 groups)',
               'paper_sampling_protocol': len(records) == 10 and a.samples_per_model == 5,
               'metrics': {k: v.mean().item() for k, v in result_scores.items()},
               'raw_ensemble_metrics': {k: v.mean().item() for k, v in raw_scores.items()},
               'data_sha256': data_hash}
    root = Path(a.output); root.mkdir(parents=True, exist_ok=True)
    if (root / 'metrics.json').exists():
        raise FileExistsError('Evaluation output already exists; choose a new directory')
    np.savez_compressed(root / 'predictions.npz', predictions=pred.numpy(), target=target.numpy(),
                        dates=data['test_dates'], **{k: v.numpy() for k, v in result_scores.items()})
    (root / 'metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
