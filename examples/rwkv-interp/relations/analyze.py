"""Validate dump semantics and report retrieval behavior without fitting a probe."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from data import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=Path)
args = parser.parse_args()
ds = Dataset(args.dataset)
metrics = defaultdict(float)
counts = defaultdict(int)
for segment in ds.segments:
    t = ds.trajectory(segment)
    token_rows = ds.tokens(segment)
    n = int(ds.segments[segment]['n_tokens'])
    start = int(ds.segments[segment]['start_pos'])
    if len(token_rows) != n or [r['position'] for r in token_rows] != list(range(start, start+n)):
        raise ValueError(f'{segment}: invalid token positions')
    meta = ds.prompts.get(segment)
    if meta is not None and b''.join(r['piece'] for r in token_rows) != meta['text'].encode():
        raise ValueError(f'{segment}: token pieces differ from prompt text')
    states = t['wkv']
    v = t['vectors']
    for i, layer in enumerate(ds.selected):
        old = np.asarray(states[:-1, i], dtype=np.float64)
        new = np.asarray(states[1:, i], dtype=np.float64)
        r, w, k, value, a, b = [np.asarray(v[f][:, layer], dtype=np.float64) for f in ('r', 'w', 'k', 'v', 'a', 'b')]
        # Native matrix axes are [head, value, key]. a/b are fused-kernel operands, not the pre-fusion sigmoid.
        sa = np.einsum('thij,thj->thi', old, a)
        predicted = old*w[..., None, :] + value[..., :, None]*k[..., None, :] + sa[..., :, None]*b[..., None, :]
        read = np.einsum('thij,thj->thi', new, r)
        for name, predicted_x, actual in [('recurrence', predicted, new), ('read', read, v['wkv_read'][:, layer])]:
            delta = predicted_x - actual
            rmse = np.sqrt(np.mean(delta*delta))
            relative = rmse / max(np.sqrt(np.mean(np.asarray(actual, dtype=np.float64)**2)), 1e-12)
            metrics[name + '_max_relative_rmse'] = max(metrics[name + '_max_relative_rmse'], float(relative))
            metrics[name + '_max_abs'] = max(metrics[name + '_max_abs'], float(np.max(np.abs(delta))))
            if relative > 1e-4:
                raise ValueError(f'{segment} layer {layer}: {name} relative RMSE {relative}')
        y = np.asarray(v['wkv_read'][:, layer], dtype=np.float64)
        norm = (y - y.mean(axis=-1, keepdims=True))/np.sqrt(y.var(axis=-1, keepdims=True) + 64e-5)
        err = float(np.max(np.abs(norm - v['group_norm'][:, layer])))
        metrics['group_norm_max_abs'] = max(metrics['group_norm_max_abs'], err)
        if err > 1e-3:
            raise ValueError(f'{segment} layer {layer}: normalization error {err}')
    boundary = ds.boundary(segment)
    if not np.array_equal(boundary['wkv'][ds.selected], states[-1]):
        raise ValueError(f'{segment}: boundary WKV differs from final token')
    if not np.array_equal(boundary['shifts'][ds.selected], t['shifts'][-1]):
        raise ValueError(f'{segment}: boundary shifts differ from final token')
    parent = ds.segments[segment]['parent']
    if parent != '-':
        initial = ds.boundary(parent)
        if not np.array_equal(initial['wkv'][ds.selected], states[0]):
            raise ValueError(f'{segment}: initial WKV differs from parent')
        if not np.array_equal(initial['shifts'][ds.selected], t['shifts'][0]):
            raise ValueError(f'{segment}: initial shifts differ from parent')
    counts['segments_validated'] += 1

scores = defaultdict(dict)
with (args.dataset/'scores.tsv').open() as f:
    for row in csv.DictReader(f, delimiter='\t'):
        scores[row['segment']][bytes.fromhex(row['candidate_hex']).decode()] = float(row['log_probability'])
results = []
for segment, candidates in scores.items():
    meta = ds.prompts.get(segment)
    if meta is None:
        continue
    predicted = max(candidates, key=candidates.get)
    group = f'{meta["identity"]}/{meta["kind"]}/' + ('updated' if meta.get('update_target') == meta['target'] else 'preserved')
    correct = predicted == meta['gold']
    counts[group + '_total'] += 1
    counts[group + '_correct'] += int(correct)
    results.append(dict(segment=segment, gold=meta['gold'], predicted=predicted, correct=correct, scores=candidates))
summary = dict(validation=dict(metrics), counts=dict(counts), query_accuracy=sum(r['correct'] for r in results)/len(results) if results else None,
               total_queries=len(results), note='Forced completion candidate likelihoods; not free-generation accuracy. No mechanism inferred.')
(args.dataset/'validation.json').write_text(json.dumps(summary, indent=2) + '\n')
(args.dataset/'query-results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results))
print(json.dumps(summary, indent=2))
