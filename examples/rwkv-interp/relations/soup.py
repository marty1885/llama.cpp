"""A small starting point for matched-history matrix analysis, without model inference."""
import argparse
from pathlib import Path
import numpy as np
from data import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=Path)
parser.add_argument('--segment', default='h000_u0')
parser.add_argument('--layer', type=int, default=0)
args = parser.parse_args()
ds = Dataset(args.dataset)
t = ds.trajectory(args.segment)
if args.layer not in ds.selected:
    raise ValueError(f'Choose a captured layer from {ds.selected}')
slot = ds.selected.index(args.layer)
S = t['wkv'][:, slot]
vectors = t['vectors']
a, b, v, k, w = [vectors[name][:, args.layer] for name in ('a', 'b', 'v', 'k', 'w')]
# Return per-token, per-head energies of actual kernel terms, not semantic directions.
write = v[..., :, None] * k[..., None, :]
correction = np.einsum('thij,thj->thi', S[:-1], a)[..., :, None] * b[..., None, :]
decay_change = S[:-1] * (w[..., None, :] - 1)
for name, term in [('new_write', write), ('memory_correction', correction), ('decay_change', decay_change)]:
    energy = np.linalg.norm(term, axis=(-2, -1))
    print(name, 'shape=', energy.shape, 'mean=', float(energy.mean()), 'max=', float(energy.max()))
print('Final-state singular values for head 0:', np.linalg.svd(S[-1, 0], compute_uv=False))
