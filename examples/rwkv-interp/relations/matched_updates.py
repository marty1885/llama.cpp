"""Compare actual update operands for identical continuation tokens after matched histories."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from data import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=Path)
args = parser.parse_args()
ds = Dataset(args.dataset)
groups = defaultdict(list)
for sid, meta in ds.prompts.items():
    if meta['kind'] != 'update' or meta['shared'] or sid not in ds.segments:
        continue
    if not (ds.path/sid/'DONE').exists():
        continue
    key = (meta['identity'], tuple(meta['countries']), tuple(sorted(meta['original_colors'])), meta['order'], meta['update_target'])
    groups[key].append(sid)
fields = ['r', 'w', 'k', 'v', 'a', 'b']
profiles, pairs = [], []
for group in groups.values():
    if len(group) != 2:
        continue
    a, b = sorted(group)
    tokens_a, tokens_b = ds.tokens(a), ds.tokens(b)
    if [t['token'] for t in tokens_a] != [t['token'] for t in tokens_b]:
        raise ValueError(f'{a}/{b}: update token IDs differ')
    if [t['position'] for t in tokens_a] != [t['position'] for t in tokens_b]:
        raise ValueError(f'{a}/{b}: update positions differ')
    va, vb = ds.trajectory(a)['vectors'], ds.trajectory(b)['vectors']
    profile = np.empty((ds.layers, len(fields)))
    for f, name in enumerate(fields):
        x, y = np.asarray(va[name], dtype=np.float64), np.asarray(vb[name], dtype=np.float64)
        squared_difference = np.mean((x-y)**2, axis=(0, 2, 3))
        reference = np.mean((x*x+y*y)/2, axis=(0, 2, 3))
        profile[:, f] = np.sqrt(squared_difference/np.maximum(reference, 1e-24))
    profiles.append(profile)
    pairs.append((a, b))
if not profiles:
    raise ValueError('No completed matched update pairs')
profiles = np.stack(profiles)
np.savez(ds.path/'matched-update-profiles.npz', relative_rmse=profiles, pairs=np.asarray(pairs), fields=np.asarray(fields))
summary = dict(pairs=len(pairs), fields=fields, mean_relative_rmse_by_layer=profiles.mean(axis=0).tolist(),
               note='Same input token IDs and positions; observed operand dependence on prior history. This does not isolate semantic binding.')
(ds.path/'matched-update-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print('Matched update pairs:', len(pairs))
for layer in ds.selected:
    print('layer', layer, dict(zip(fields, profiles[:, layer].mean(axis=0).tolist())))
