import ast
import importlib.util
from pathlib import Path

import numpy as np
import torch

from weather.model import TemperatureCGM
from weather.metrics import scores
from weather.objectives import training_energy_score, riesz_loss

ROOT = Path(__file__).resolve().parents[1]


def official_functions():
    source = (ROOT / 'third_party/mvpp/scoring_rules_supp.py').read_text()
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('es_sample', 'vs_sample')]
    namespace = {'np': np}
    exec(compile(tree, '<official scores>', 'exec'), namespace)
    return namespace


def test_metrics_match_official_and_naive_crps():
    rng = np.random.default_rng(12)
    p, y = rng.normal(size=(4, 7, 3)), rng.normal(size=(4, 3))
    result = scores(torch.tensor(p), torch.tensor(y))
    official = official_functions()
    np.testing.assert_allclose(result['energy_score'].mean(), official['es_sample'](y, p.transpose(0, 2, 1)), rtol=1e-12)
    np.testing.assert_allclose(result['variogram_score'].mean(), official['vs_sample'](y, p.transpose(0, 2, 1)), rtol=1e-12)
    crps = np.abs(p-y[:, None]).mean(1) - .5*np.abs(p[:, :, None]-p[:, None, :]).mean((1, 2))
    np.testing.assert_allclose(result['mean_crps'], crps.mean(1), rtol=1e-12)


def test_training_energy_matches_official_formula_and_gradient():
    torch.manual_seed(4)
    p = torch.randn(3, 5, 4, dtype=torch.double, requires_grad=True)
    y = torch.randn(3, 4, dtype=torch.double)
    # Independent explicit pair differences, same epsilon clipping and denominator.
    expected = (p-y[:, None]).square().sum(-1).clamp(1e-7, 1e10).sqrt().mean(1)
    expected -= (p[:, :, None]-p[:, None, :]).square().sum(-1).clamp(1e-7, 1e10).sqrt().sum((1, 2))/(2*5*4)
    torch.testing.assert_close(training_energy_score(p, y), expected)
    gradient = torch.autograd.grad(training_energy_score(p, y).sum(), p)[0]
    torch.testing.assert_close(gradient, torch.autograd.grad(expected.sum(), p)[0])


def test_riesz_preserves_condition_groups_and_stops_reference_gradient():
    torch.manual_seed(5)
    p = torch.randn(2, 6, 3, requires_grad=True)
    y = torch.randn(2, 3, requires_grad=True)
    loss, _ = riesz_loss(p, y)
    grad = torch.autograd.grad(loss[0], p, retain_graph=True)[0]
    assert grad[0].abs().sum() > 0
    assert grad[1].abs().sum() == 0
    # Global distance/RMS scales share statistics across cases in upstream loss,
    # but are detached; neighbours and positive references remain within each case.
    loss.mean().backward()
    assert y.grad is None and torch.isfinite(p.grad).all()
    perm = torch.randperm(6)
    perm_loss, _ = riesz_loss(p[:, perm], y)
    torch.testing.assert_close(loss, perm_loss)


def test_model_forward_matches_explicit_three_part_architecture():
    torch.manual_seed(6)
    model = TemperatureCGM(3)
    mean, std, context, z = torch.randn(2,3,17), torch.randn(2,3,17), torch.randn(2,3,39), torch.randn(2,5,10)
    actual = model(mean, std, context, z)
    mu = torch.stack([(mean[:, d] * model.mean_weight[d]).sum(-1)+model.mean_bias[d] for d in range(3)], -1)
    delta = torch.exp(std.flatten(1) @ model.noise_scale.weight.T + model.noise_scale.bias)
    predictions = []
    for j in range(5):
        predictions.append(mu + model.decoder(torch.cat([context.flatten(1), delta*z[:,j]], -1)))
    torch.testing.assert_close(actual, torch.stack(predictions, 1))
    actual.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
