import os
os.environ.setdefault('DRIFT_COMPILE', '0')
import torch
from .riesz_topk_softmax import riesz_topk_softmax_loss
from .uniform_riesz import uniform_riesz_loss


def training_energy_score(pred, target):
    """Match official TF loss, including epsilon-clipped self distances.

    pred [B,M,D], target [B,D]. U denominator M(M-1), tiny constant diagonal.
    Evaluation deliberately uses M^2 instead (the empirical-distribution score).
    """
    m = pred.shape[1]
    if m < 2:
        raise ValueError('Energy training requires at least two predictions per case')
    gram = pred @ pred.transpose(1, 2)
    squared = pred.square().sum(-1)
    pair = (squared[:, :, None] + squared[:, None, :] - 2 * gram).clamp(1e-7, 1e10).sqrt()
    attraction = (target.square().sum(-1)[:, None] + squared -
                  2 * (pred * target[:, None]).sum(-1)).clamp(1e-7, 1e10).sqrt()
    return attraction.mean(1) - pair.sum((1, 2)) / (2 * m * (m - 1))


def riesz_loss(pred, target, temperatures=(0.2, 0.05, 0.02), *, weighting='softmax'):
    # B = forecast cases, M = predictions for ONE condition, one observed positive.
    # No pooling unrelated dates, no MAE, no CFG negatives or reference memory bank.
    if weighting=='uniform':
        return uniform_riesz_loss(pred,target,temperatures)
    if weighting!='softmax':
        raise ValueError('weighting must be softmax or uniform')
    return riesz_topk_softmax_loss(
        pred, target[:, None], tau_list=temperatures,
        normalize_per_temperature=True, epsilon=1e-6, direction_epsilon=1e-8)
