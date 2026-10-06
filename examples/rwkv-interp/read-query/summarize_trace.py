"""Describe natural read differences, including state/read-vector interactions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from trace_reads import Dataset, contributions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('trace', type=Path)
    args = parser.parse_args()
    ds = Dataset(args.dataset)
    with np.load(args.trace / 'contrasts.npz') as contrasts:
        pairs = json.loads(str(contrasts['pairs']))
    rows = []
    checks = []
    for kind, a, b in pairs:
        tokens_a = [t['token'] for t in ds.tokens(a)]
        tokens_b = [t['token'] for t in ds.tokens(b)]
        if kind != 'query' and tokens_a != tokens_b:
            raise ValueError('Fact contrast changed the literal query')
        if kind == 'query' and [i for i, (x, y) in enumerate(zip(tokens_a, tokens_b)) if x != y] != [9]:
            raise ValueError('Query contrast changed more than the city')
        with np.load(args.trace / f'{a}.npz') as x, np.load(args.trace / f'{b}.npz') as y:
            nstory = int(ds.segments[ds.segments[a]['parent']]['n_tokens'])
            delta = contributions(y, -1, None)[0] - contributions(x, -1, None)[0]
            total = y['observed'][-1] - x['observed'][-1]
            state_a = np.asarray(ds.boundary(a)['wkv'], dtype=np.float64)
            state_b = np.asarray(ds.boundary(b)['wkv'], dtype=np.float64)
            r_a, r_b = x['r'][-1], y['r'][-1]
            terms = dict(story_origins=delta[:nstory].sum(axis=0), query_origins=delta[nstory:].sum(axis=0),
                         earlier_query_origins=delta[nstory:-1].sum(axis=0), current_token=delta[-1],
                         r_only=np.einsum('lhvk,lhk->lhv', state_a, r_b - r_a),
                         state_only=np.einsum('lhvk,lhk->lhv', state_b - state_a, r_a),
                         interaction=np.einsum('lhvk,lhk->lhv', state_b - state_a, r_b - r_a))
            reconstructed = terms['r_only'] + terms['state_only'] + terms['interaction']
            checks.append(float(np.max(np.linalg.norm(reconstructed - total, axis=(-2, -1)) / np.maximum(np.linalg.norm(x['observed'][-1], axis=(-2, -1)), 1e-12))))
            denom = np.sum(total ** 2, axis=(-2, -1))
            for name, value in terms.items():
                projection = np.sum(value * total, axis=(-2, -1)) / np.maximum(denom, 1e-24)
                for layer, p in enumerate(projection):
                    rows.append(dict(kind=kind, a=a, b=b, term=name, layer=layer, signed_projection=float(p)))
    if max(checks) > 1e-4:
        raise ValueError('State/read interaction reconstruction failed')
    import csv
    with (args.trace / 'colon-decomposition.tsv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    report = dict(max_interaction_reconstruction_error=max(checks),
                  definition='Signed projection onto the full observed colon read difference, averaged over eight pairs. Story + query = 1; r_only + state_only + interaction = 1. Terms may cancel; these are not causal fractions. State decomposition is asymmetric around recipient A.',
                  bands={})
    for kind in ('query', 'colors', 'cities_fixed_query'):
        report['bands'][kind] = {}
        for lo, hi in ((0, 19), (20, 39), (40, 60)):
            report['bands'][kind][f'{lo}-{hi}'] = {term: float(np.mean([row['signed_projection'] for row in rows if row['kind'] == kind and row['term'] == term and lo <= row['layer'] <= hi])) for term in terms}
    (args.trace / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    paths = [Path(__file__), Path(__file__).with_name('trace_reads.py'), args.dataset / 'config.tsv', args.dataset / 'segments.tsv']
    provenance = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (args.trace / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
