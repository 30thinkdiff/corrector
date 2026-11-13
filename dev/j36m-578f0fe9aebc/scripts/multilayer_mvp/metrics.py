"""Layer-mean normalized MSE; zero-safe per-transition cosine metrics."""
import math
import torch
import torch.nn.functional as F


def loss(predictions, targets, scales):
    if set(predictions) != set(targets):
        raise ValueError('prediction/target keys mismatch')
    losses = []
    for name, value in predictions.items():
        if value.shape != targets[name].shape:
            raise ValueError(f'target shape mismatch: {name}')
        item = (value.float() - targets[name].float()).square().mean() / scales[name]**2
        if not torch.isfinite(item):
            raise FloatingPointError(f'nonfinite loss: {name}')
        losses.append(item)
    return torch.stack(losses).mean()


class Metrics:
    def __init__(self, scales):
        self.scales = scales
        self.rows = {}

    def add(self, predictions, targets):
        for name, pred in predictions.items():
            p, y = pred.detach().float().flatten(1), targets[name].detach().float().flatten(1)
            if not (torch.isfinite(p).all() and torch.isfinite(y).all()):
                raise FloatingPointError('nonfinite validation tensor')
            row = self.rows.setdefault(name, dict(sse=0., zero_sse=0., elements=0, samples=0,
                                                 cosine_sum=0., cosine_valid=0, zero_targets=0,
                                                 zero_predictions=0))
            p_norm, y_norm = p.norm(dim=1), y.norm(dim=1)
            valid = (p_norm > 1e-8) & (y_norm > 1e-8)
            row['sse'] += float((p-y).double().square().sum())
            row['zero_sse'] += float(y.double().square().sum())
            row['elements'] += y.numel()
            row['samples'] += len(y)
            row['cosine_sum'] += float(F.cosine_similarity(p[valid], y[valid], dim=1).sum())
            row['cosine_valid'] += int(valid.sum())
            row['zero_targets'] += int((y_norm <= 1e-8).sum())
            row['zero_predictions'] += int((p_norm <= 1e-8).sum())

    def result(self):
        output = {}
        for name, r in self.rows.items():
            mse = r['sse'] / r['elements']
            zero = r['zero_sse'] / r['elements']
            output[name] = {
                'mse': mse, 'normalized_mse': mse / self.scales[name]**2,
                'zero_delta_mse': zero, 'zero_delta_normalized_mse': zero / self.scales[name]**2,
                'relative_mse_vs_zero': mse / zero if zero > 0 else None,
                'cosine': r['cosine_sum'] / r['cosine_valid'] if r['cosine_valid'] else None,
                **{k: r[k] for k in ('samples', 'cosine_valid', 'zero_targets', 'zero_predictions')},
            }
            if not math.isfinite(mse):
                raise FloatingPointError('invalid metrics')
        return output
