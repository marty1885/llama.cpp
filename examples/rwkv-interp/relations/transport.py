"""Exact write bookkeeping with the update operands fixed to the observed trajectory."""
import argparse
import json
from pathlib import Path
import numpy as np
from data import Dataset

def lineage(ds, segment):
    result = []
    seen = set()
    while segment != '-':
        if segment in seen:
            raise ValueError('Cyclic segment lineage')
        seen.add(segment)
        result.append(segment)
        segment = ds.segments[segment]['parent']
    return result[::-1]

def transported_writes(ds, segment, layer, head):
    if not 0 <= layer < ds.layers or not 0 <= head < ds.heads:
        raise ValueError('Layer or head out of range')
    chain = lineage(ds, segment)
    total = sum(int(ds.segments[s]['n_tokens']) for s in chain)
    d = ds.head_size
    contributions = np.zeros((total, d, d), dtype=np.float64)
    labels = []
    count = 0
    for sid in chain:
        trace = ds.trajectory(sid)['vectors']
        for token in range(int(ds.segments[sid]['n_tokens'])):
            w, k, v, a, b = [np.asarray(trace[name][token, layer, head], dtype=np.float64) for name in ('w', 'k', 'v', 'a', 'b')]
            old = contributions[:count]
            projected = np.einsum('tij,j->ti', old, a)
            old *= w[None, None, :]
            old += projected[:, :, None]*b[None, None, :]
            contributions[count] = v[:, None]*k[None, :]
            labels.append((sid, token))
            count += 1
    actual = np.asarray(ds.boundary(segment)['wkv'][layer, head], dtype=np.float64)
    summed = contributions.sum(axis=0)
    rmse = np.sqrt(np.mean((summed-actual)**2))/max(np.sqrt(np.mean(actual**2)), 1e-12)
    r = np.asarray(trace['r'][-1, layer, head], dtype=np.float64)
    reads = np.einsum('tij,j->ti', contributions, r)
    observed_read = np.asarray(trace['wkv_read'][-1, layer, head], dtype=np.float64)
    read_rmse = np.sqrt(np.mean((reads.sum(axis=0)-observed_read)**2))/max(np.sqrt(np.mean(observed_read**2)), 1e-12)
    return dict(contributions=contributions, read_contributions=reads, labels=labels,
                relative_rmse=float(rmse), read_relative_rmse=float(read_rmse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--segment', default='h000_u0_q0')
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--head', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    ds = Dataset(args.dataset)
    result = transported_writes(ds, args.segment, args.layer, args.head)
    if result['relative_rmse'] > 1e-4 or result['read_relative_rmse'] > 1e-4:
        raise ValueError('Transport reconstruction failed')
    if args.output:
        np.savez(args.output, contributions=result['contributions'], read_contributions=result['read_contributions'], labels=np.asarray(result['labels']))
    print(json.dumps({k: v for k, v in result.items() if k not in ('contributions', 'read_contributions')}, indent=2))
