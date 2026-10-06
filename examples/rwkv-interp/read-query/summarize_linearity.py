import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1])
assert (root / 'DONE').exists()
with (root / 'scores.tsv').open() as f:
    rows = list(csv.DictReader(f, delimiter='\t'))
assert len(rows) == 976
q = np.memmap(root / 'queries.f32', dtype='<f4', mode='r', shape=(976, 3, 3, 4096))
assert (root / 'queries.f32').stat().st_size == q.nbytes
lookup = {(int(r['case']), int(r['direction']), float(r['alpha'])): j for j, r in enumerate(rows)}
assert len(lookup) == 976
inputs = np.fromfile(root / 'directions.f32', dtype='<f4').reshape(16, 8, 4096)
assert np.isfinite(inputs).all() and np.isfinite(q).all()
assert np.array_equal(inputs[:, 2:6], np.broadcast_to(inputs[0, 2:6], (16, 4, 4096)))
assert np.array_equal(inputs[:, 6], inputs[:, 0] + inputs[:, 1])
assert np.array_equal(inputs[:, 7], inputs[:, 4] + inputs[:, 5])
for r in rows:
    if int(r['direction']) == -1:
        assert float(r['margin']) == float(r['baseline_margin'])
    i, d, a = int(r['case']), int(r['direction']), float(r['alpha'])
    if d >= 0:
        assert np.isclose(float(r['delta_norm']), np.linalg.norm(inputs[i, d].astype(float) * a), rtol=1e-6)

for d in [0, 1]:
    for a in [-1., -.5, -.1, -.03, .03, .1, .5, 1., 2.]:
        j, k = lookup[0, d, a], lookup[0, d+2, a]
        assert np.array_equal(q[j], q[k])
        assert rows[j]['margin'] == rows[k]['margin']

records = []

def metric(actual, predicted, **meta):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    err = actual - predicted
    norm = np.linalg.norm(actual, axis=-1)
    error_norm = np.linalg.norm(err, axis=-1)
    ratio = error_norm / np.maximum(norm, 1e-12)
    record = dict(meta,
                  relative_error=float(np.linalg.norm(err) / max(np.linalg.norm(actual), 1e-12)),
                  effect_rms=float(np.sqrt(np.mean(actual * actual))),
                  error_rms=float(np.sqrt(np.mean(err * err))),
                  median_individual_error=float(np.median(ratio)),
                  p90_individual_error=float(np.quantile(ratio, .9)),
                  fraction_below_10pct=float(np.mean(ratio <= .1)),
                  n=int(np.prod(norm.shape)))
    records.append(record)

scales = [-1., -.5, -.1, -.03, .03, .1, .5, 1., 2.]
groups = {'natural': [0, 1], 'fixed_natural': [2, 3], 'random': [4, 5]}
for target in [(t, l) for t in range(3) for l in range(3)] + ['margin']:
    label = 'margin' if target == 'margin' else f'r_L{[36,48,60][target[1]]}_t{target[0]+9}'
    def get(i, d, a):
        ix = lookup[i, d, a]
        if target == 'margin':
            return np.array([float(rows[ix]['margin'])])
        return np.asarray(q[ix, target[0], target[1]], dtype=float)
    base = np.stack([get(i, -1, 0) for i in range(16)])
    effects = {(d, a): np.stack([get(i, d, a) for i in range(16)]) - base
               for d in range(8) for a in (scales if d < 6 else [.1, .5, 1.])}
    jac = {d: (effects[d, .03] - effects[d, -.03]) / .06 for d in range(6)}
    jac_large = {d: (effects[d, .1] - effects[d, -.1]) / .2 for d in range(6)}
    secant = {d: effects[d, .5] - effects[d, -.5] for d in range(6)}
    # Least-squares coefficients through the unperturbed origin, fit only on small steps.
    small = np.array([-.1, -.03, .03, .1])
    design_inv = np.linalg.pinv(np.stack([small, small**2], axis=1))
    quadratic = {d: np.einsum('ab,bic->aic', design_inv, np.stack([effects[d, a] for a in small])) for d in range(6)}
    for group, directions in groups.items():
        metric(np.stack([jac_large[d] for d in directions]), np.stack([jac[d] for d in directions]),
               target=label, test='derivative_convergence', group=group, alpha=.1)
        for a in scales:
            actual = np.stack([effects[d, a] for d in directions])
            metric(actual, np.stack([a*jac[d] for d in directions]), target=label, test='local_tangent', group=group, alpha=a)
            metric(actual, np.stack([a*secant[d] for d in directions]), target=label, test='half_scale_secant', group=group, alpha=a)
            predicted = np.stack([a*quadratic[d][0] + a*a*quadratic[d][1] for d in directions])
            metric(actual, predicted, target=label, test='local_quadratic', group=group, alpha=a)
    for group, d1, d2, total in [('natural', 0, 1, 6), ('random', 4, 5, 7)]:
        for a in [.1, .5, 1.]:
            metric(effects[total, a], effects[d1, a] + effects[d2, a],
                   target=label, test='superposition', group=group, alpha=a)
    splits = {'colors': (np.arange(8), np.arange(8, 16)),
              'order': (np.array([i for i in range(16) if (i//2)%2 == 0]), np.array([i for i in range(16) if (i//2)%2 == 1]))}
    for split, (train, test) in splits.items():
        for group, directions in list(groups.items())[1:]:
            for a in [.1, .5, 1.]:
                actual = np.stack([effects[d, a][test] for d in directions])
                common = np.stack([np.broadcast_to(a*jac[d][train].mean(axis=0), effects[d, a][test].shape) for d in directions])
                local = np.stack([a*jac[d][test] for d in directions])
                curved = np.stack([np.broadcast_to(a*quadratic[d][0, train].mean(axis=0) + a*a*quadratic[d][1, train].mean(axis=0), effects[d, a][test].shape) for d in directions])
                for method, prediction in [('shared', common), ('recipient_tangent', local), ('shared_quadratic', curved), ('zero', np.zeros_like(actual))]:
                    metric(actual, prediction, target=label, test=f'transfer_{split}_{method}', group=group, alpha=a)

with (root / 'metrics.tsv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0]), delimiter='\t')
    writer.writeheader()
    writer.writerows(records)

def select(test, group, target, alpha):
    return next(r for r in records if r['test'] == test and r['group'] == group and r['target'] == target and r['alpha'] == alpha)

summary = {'runs': 976, 'exact_zero_controls': 16,
           'local_natural_at_one': {t: select('local_tangent', 'natural', t, 1.) for t in ['r_L36_t9', 'r_L48_t9', 'r_L60_t11', 'margin']},
           'superposition_at_one': {t: select('superposition', 'natural', t, 1.) for t in ['r_L36_t9', 'r_L60_t11', 'margin']},
           'transfer_at_one': {t: select('transfer_colors_shared', 'fixed_natural', t, 1.) for t in ['r_L36_t9', 'r_L60_t11', 'margin']}}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
colors = ['#0072B2', '#D55E00', '#009E73']
targets = ['r_L36_t9', 'r_L60_t11', 'margin']
labels = ['Query L36, city token', 'Query L60, colon', 'Answer margin']
for target, label, color in zip(targets, labels, colors):
    for test, style in [('local_tangent', '-'), ('local_quadratic', '--')]:
        x = [.03, .1, .5, 1., 2.]
        axes[0, 0].plot(x, [select(test, 'natural', target, a)['relative_error'] for a in x], style, color=color, marker='o', label=label if style == '-' else None)
    x = [.1, .5, 1.]
    axes[0, 1].plot(x, [select('superposition', 'natural', target, a)['relative_error'] for a in x], color=color, marker='o', label=label)
    for test, style in [('transfer_colors_shared', '-'), ('transfer_colors_recipient_tangent', '--')]:
        axes[1, 0].plot(x, [select(test, 'fixed_natural', target, a)['relative_error'] for a in x], style, color=color, marker='o', label=label if style == '-' else None)
for ax in axes.flat[:3]:
    ax.axhline(.1, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Perturbation scale (1 = natural difference magnitude)')
    ax.set_ylabel('Error / actual effect (L2)')
    ax.legend(fontsize=8)
axes[0, 0].set_title('Local: solid tangent; dashed quadratic (fit <=0.1)')
axes[0, 0].set_xscale('log')
axes[0, 1].set_title('Superposition: natural city + color directions')
axes[1, 0].set_title('Held-out colors: solid shared, dashed recipient tangent')
x = np.arange(3)
for k, (group, label) in enumerate([('natural', 'Recipient natural'), ('fixed_natural', 'Fixed natural'), ('random', 'Random, matched norm')]):
    axes[1, 1].bar(x + (k-1)*.25, [select('local_tangent', group, t, 1.)['relative_error'] for t in targets], width=.25, label=label)
axes[1, 1].set_xticks(x, ['L36 / city', 'L60 / colon', 'Margin'])
axes[1, 1].set_ylabel('Error / actual effect (L2)')
axes[1, 1].set_title('Local tangent error at scale 1')
axes[1, 1].legend(fontsize=8)
mode = 'activation-quantized matvec disabled' if 'no-mmvq' in root.name else 'default kernels'
fig.suptitle(f'Full-layer read perturbations - Vulkan, {mode}')
fig.tight_layout()
fig.savefig(root / 'linearity.png', dpi=170, bbox_inches='tight')
plt.close(fig)
source_dir = Path(__file__).resolve().parent
paths = [source_dir / 'read-linearity.cpp', Path(__file__), source_dir.parent / 'collect.cpp', source_dir / 'dot-plan.tsv', root / 'protocol.txt']
(root / 'provenance.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2) + '\n')
print(json.dumps(summary, indent=2))
