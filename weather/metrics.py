import torch


def scores(pred, target):
    """Per-case physical-unit ES, unweighted VS(p=.5), and station-mean CRPS."""
    pred, target = pred.double(), target.double()
    es = torch.linalg.vector_norm(pred - target[:, None], dim=-1).mean(1)
    es -= 0.5 * torch.cdist(pred, pred).mean((1, 2))
    observed = (target[:, :, None] - target[:, None, :]).abs().sqrt()
    predicted = (pred[:, :, :, None] - pred[:, :, None, :]).abs().sqrt().mean(1)
    vs = (observed - predicted).square().sum((1, 2))
    # Sorted ensemble formula avoids [B,M,M,D] storage for marginal CRPS.
    ordered = pred.sort(dim=1).values
    m = pred.shape[1]
    coeff = (2 * torch.arange(1, m + 1, device=pred.device, dtype=pred.dtype) - m - 1)
    crps = ((pred - target[:, None]).abs().mean(1) -
            (ordered * coeff[None, :, None]).sum(1) / m**2).mean(1)
    return {'energy_score': es, 'variogram_score': vs, 'mean_crps': crps}
