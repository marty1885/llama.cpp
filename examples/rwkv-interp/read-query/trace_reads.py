"""Full-layer read decomposition using observed recurrent operands."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'relations'))
from data import Dataset
from transport import lineage, transported_writes


def relative(error, reference):
    return np.linalg.norm(error, axis=(-2, -1)) / np.maximum(np.linalg.norm(reference, axis=(-2, -1)), 1e-12)


def trace(ds, sid, output):
    chain = lineage(ds, sid)
    vectors = {name: np.concatenate([ds.trajectory(s)['vectors'][name] for s in chain]).astype(np.float64)
               for name in ('w', 'k', 'v', 'a', 'b', 'r', 'wkv_read')}
    n, layers, heads, size = vectors['r'].shape
    keys = np.zeros_like(vectors['k'])
    coefficients = np.zeros((n, n, layers, heads))
    norms = np.zeros((n, n, layers))
    projections = np.zeros_like(norms)
    errors = []
    for t in range(n):
        old = keys[:t]
        projected = np.sum(old * vectors['a'][t], axis=-1)
        old *= vectors['w'][t]
        old += projected[..., None] * vectors['b'][t]
        keys[t] = vectors['k'][t]
        coefficients[t, :t + 1] = np.sum(keys[:t + 1] * vectors['r'][t], axis=-1)
        reads = coefficients[t, :t + 1, ..., None] * vectors['v'][:t + 1]
        observed = vectors['wkv_read'][t]
        errors.append(relative(reads.sum(axis=0) - observed, observed))
        norms[t, :t + 1] = np.linalg.norm(reads, axis=(-2, -1))
        projections[t, :t + 1] = np.sum(reads * observed, axis=(-2, -1)) / np.maximum(np.sum(observed ** 2, axis=(-2, -1)), 1e-24)
    state = np.einsum('olhv,olhk->lhvk', vectors['v'], keys)
    actual = np.asarray(ds.boundary(sid)['wkv'], dtype=np.float64)
    state_error = np.linalg.norm((state - actual).reshape(layers, -1), axis=-1) / np.maximum(np.linalg.norm(actual.reshape(layers, -1), axis=-1), 1e-12)
    errors = np.asarray(errors)
    if not np.isfinite(coefficients).all() or max(errors.max(), state_error.max()) > 1e-4:
        raise ValueError(f'{sid}: reconstruction failed')
    labels = [dict(segment=s, token=t['index'], piece=t['piece'].decode('utf-8', errors='replace')) for s in chain for t in ds.tokens(s)]
    np.savez_compressed(output / f'{sid}.npz', coefficients=coefficients, values=vectors['v'], norms=norms,
                        projections=projections, r=vectors['r'], observed=vectors['wkv_read'], labels=json.dumps(labels))
    return dict(segment=sid, reads_checked=int(errors.size), max_read_relative_error=float(errors.max()), max_state_relative_error=float(state_error.max()))


def contributions(data, start, stop):
    return data['coefficients'][start:stop, :, ..., None] * data['values'][None]


def compare(ds, output):
    results = {name: [] for name in ('query', 'colors', 'cities_fixed_query')}
    source = {name: [] for name in results}
    pairs = []
    for s in range(8):
        pairs.append(('query', f's{s}_q0', f's{s}_q1'))
        if s < 4:
            for q in range(2):
                pairs.append(('colors', f's{s}_q{q}', f's{s + 4}_q{q}'))
        if s % 4 < 2:
            for q in range(2):
                pairs.append(('cities_fixed_query', f's{s}_q{q}', f's{s + 2}_q{1 - q}'))
    for kind, a, b in pairs:
        with np.load(output / f'{a}.npz') as x, np.load(output / f'{b}.npz') as y:
            nstory = int(ds.segments[ds.segments[a]['parent']]['n_tokens'])
            delta = contributions(y, nstory, None) - contributions(x, nstory, None)
            read_delta = y['observed'][nstory:] - x['observed'][nstory:]
            scale = np.linalg.norm(x['observed'][nstory:], axis=(-2, -1))
            r_scale = np.linalg.norm(x['r'][nstory:], axis=(-2, -1))
            results[kind].append(np.stack((np.linalg.norm(read_delta, axis=(-2, -1)) / np.maximum(scale, 1e-12),
                                          np.linalg.norm(y['r'][nstory:] - x['r'][nstory:], axis=(-2, -1)) / np.maximum(r_scale, 1e-12))))
            source[kind].append(np.linalg.norm(delta, axis=(-2, -1)) / np.maximum(scale[:, None], 1e-12))
            if kind == 'query' and np.any(delta[:9]):
                raise ValueError('Identical query prefixes differ')
    arrays = {f'{k}_effects': np.stack(v) for k, v in results.items()}
    arrays.update({f'{k}_sources': np.stack(v) for k, v in source.items()})
    np.savez_compressed(output / 'contrasts.npz', **arrays, pairs=json.dumps(pairs))
    return arrays


def plot(output, arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, kind in enumerate(('query', 'colors', 'cities_fixed_query')):
        effect = arrays[f'{kind}_effects'].mean(axis=0)
        for row, label in enumerate(('Memory read change', 'Read vector r change')):
            ax = axes[row, col]
            im = ax.imshow(effect[row].T, origin='lower', aspect='auto', vmin=0, vmax=1)
            ax.set(title=f'{kind}: {label}', xlabel='Query token', ylabel='Layer')
            fig.colorbar(im, ax=ax, label='Relative full-vector L2 (clipped at 1)')
    fig.savefig(output / 'contrasts.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    limit = float(np.quantile(np.concatenate([arrays[f'{k}_sources'].mean(axis=0)[-1].ravel() for k in ('query', 'colors', 'cities_fixed_query')]), 0.99))
    for ax, kind in zip(axes, ('query', 'colors', 'cities_fixed_query')):
        data = arrays[f'{kind}_sources'].mean(axis=0)[-1].T
        im = ax.imshow(data, origin='lower', aspect='auto', vmin=0, vmax=limit)
        ax.set(title=f'{kind}: colon read contribution changes', ylabel='Layer', xlabel='Write-origin token position')
        ax.axvline(19.5, color='white', linestyle='--')
        fig.colorbar(im, ax=ax, label=f'Relative L2 (clipped at {limit:.3f})')
    fig.savefig(output / 'sources.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if not (args.dataset / 'DONE').exists():
        raise ValueError('Collection is incomplete')
    ds = Dataset(args.dataset)
    args.output.mkdir(exist_ok=False)
    checks = []
    for sid in ds.segments:
        checks.append(trace(ds, sid, args.output))
        print(json.dumps(checks[-1]), flush=True)
    # Cross-check the factorization against the existing explicit matrix implementation.
    with np.load(args.output / 's0_q0.npz') as data:
        for layer in (0, 30, 60):
            for block in range(ds.heads):
                old = transported_writes(ds, 's0_q0', layer, block)['read_contributions']
                new = data['coefficients'][-1, :, layer, block, None] * data['values'][:, layer, block]
                np.testing.assert_allclose(new, old, rtol=1e-8, atol=1e-8)
    arrays = compare(ds, args.output)
    plot(args.output, arrays)
    report = dict(checks=checks, explicit_matrix_checks=3 * ds.heads,
                  definitions='Contribution = original value * (observed-transported key dot observed r), concatenated across every block. Projections are signed and can cancel. Contrasts are full-vector L2; not causal effects.',
                  limitations='Eight crossed stories, two names, two colors, two cities. Write origin is not semantic content. Operands are frozen to each natural trajectory; no intervention or multi-hop mechanism established.')
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'DONE').touch()


if __name__ == '__main__':
    main()
