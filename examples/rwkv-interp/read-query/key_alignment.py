"""Compare observed query reads with story write keys, without fitting a probe."""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'relations'))
from data import Dataset


def span_tokens(ds, sid, word):
    pieces = [t['piece'] for t in ds.tokens(sid)]
    text = b''.join(pieces)
    start = text.index(word.encode())
    end = start + len(word.encode())
    offsets = np.cumsum([0] + [len(p) for p in pieces])
    return [i for i in range(len(pieces)) if offsets[i] < end and offsets[i + 1] > start]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('prompts', type=Path)
    parser.add_argument('--full-layer', action='store_true')
    args = parser.parse_args()
    ds = Dataset(args.dataset)
    if not (args.dataset / 'DONE').exists():
        raise ValueError('Collection is incomplete')
    meta = {m['id']: m for m in map(json.loads, args.prompts.read_text().splitlines())}
    differences = []
    accuracy = []
    scores = {}
    import csv
    with (args.dataset / 'scores.tsv').open() as source:
        for row in csv.DictReader(source, delimiter='\t'):
            scores.setdefault(row['segment'], {})[bytes.fromhex(row['candidate_hex']).decode()] = float(row['log_probability'])
    for sid, story in meta.items():
        if 'names' not in story:
            continue
        keys = np.asarray(ds.trajectory(sid)['vectors']['k'], dtype=np.float64)
        metrics = []
        ranks = []
        for target in range(2):
            qid = f'{sid}_q{target}'
            r = np.asarray(ds.trajectory(qid)['vectors']['r'], dtype=np.float64)
            if args.full_layer:
                r = r.reshape(r.shape[0], ds.layers, ds.embd)
                full_keys = keys.reshape(keys.shape[0], ds.layers, ds.embd)
                dot = np.einsum('qld,sld->qls', r, full_keys)
                denom = np.linalg.norm(r, axis=-1)[..., None] * np.linalg.norm(full_keys, axis=-1).T[None]
            else:
                dot = np.einsum('qlhd,slhd->qlhs', r, keys)
                denom = np.linalg.norm(r, axis=-1)[..., None] * np.linalg.norm(keys, axis=-1).transpose(1, 2, 0)[None]
            cosine = np.divide(dot, denom, out=np.zeros_like(dot), where=denom != 0)
            values = np.stack([dot, np.abs(dot), cosine])
            suffix = 'layer-key-dots' if args.full_layer else 'key-dots'
            np.savez_compressed(args.dataset / f'{qid}-{suffix}.npz', dot=dot, cosine=cosine)
            categories = []
            category_ranks = []
            for field in ('names', 'colors'):
                groups = [span_tokens(ds, sid, word) for word in story[field]]
                matched = np.stack([values[..., group].mean(axis=-1) for group in groups])
                categories.append(matched[target] - matched[1 - target])
                selected = matched[target]
                category_ranks.append((values < selected[..., None]).mean(axis=-1))
            metrics.append(np.stack(categories))
            ranks.append(np.stack(category_ranks))
            accuracy.append(dict(query=qid, gold=meta[qid]['gold'], predicted=max(scores[qid], key=scores[qid].get), scores=scores[qid]))
        differences.append((metrics[0] + metrics[1]) / 2)
        if sid == 's0':
            first_ranks = np.stack(ranks)
    differences = np.stack(differences)
    if not np.isfinite(differences).all():
        raise ValueError('Nonfinite alignment')
    mean = differences.mean(axis=0)
    scale = np.sqrt(np.mean(differences ** 2, axis=0))
    standardized = np.divide(mean, scale, out=np.zeros_like(mean), where=scale != 0)
    signs = np.array(list(itertools.product((-1, 1), repeat=len(differences))))
    flat = np.divide(differences, scale, out=np.zeros_like(differences), where=scale != 0).reshape(len(differences), -1)
    null_max = np.max(np.abs(signs @ flat / len(differences)), axis=1)
    if args.full_layer:
        rows = []
        token_rows = ds.tokens('s0_q0')
        for category, field in enumerate(('name', 'color')):
            for metric, label in enumerate(('dot', 'absolute_dot', 'cosine')):
                for token in range(mean.shape[2]):
                    for layer in range(ds.layers):
                        stat = float(standardized[category, metric, token, layer])
                        effect = differences[:, category, metric, token, layer]
                        rows.append(dict(field=field, metric=label, query_token_index=token,
                                         query_piece=token_rows[token]['piece'].decode(), layer=layer,
                                         mean_target_minus_other=float(effect.mean()), minimum=float(effect.min()), maximum=float(effect.max()),
                                         positive_histories=int(np.sum(effect > 0)), negative_histories=int(np.sum(effect < 0)),
                                         standardized_effect=stat,
                                         global_shuffle_tail_fraction=float(np.mean(null_max >= abs(stat) - 1e-12))))
        with (args.dataset / 'layer-alignment.tsv').open('w') as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(args.dataset / 'layer-alignment.npz', differences=differences, mean=mean, standardized=standardized, null_max=null_max)
        result = dict(histories=len(differences), query_results=accuracy, layers=ds.layers, dimensions=ds.embd,
                      rows=rows, arrays='Per-query layer-key-dots NPZ axes: [query token, layer, story token].',
                      null='All 256 story-level paired target-label sign flips; maximum absolute statistic over all layers, question positions, name/color categories and three metrics.',
                      limitation='Descriptive shuffle control on a fixed crossed set. Full-layer dot sums all coordinates; cosine uses full-vector norms. Absolute dot is taken after summation. Original keys may be transformed by later recurrent updates; no causal claim.')
        (args.dataset / 'layer-alignment.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
        return
    findings = []
    for category, field in enumerate(('name', 'color')):
        for metric, label in enumerate(('dot', 'absolute_dot', 'cosine')):
            block = standardized[category, metric]
            token, layer, head = np.unravel_index(np.argmax(np.abs(block)), block.shape)
            stat = float(block[token, layer, head])
            findings.append(dict(field=field, metric=label, query_token_index=int(token),
                                 query_piece=ds.tokens('s0_q0')[token]['piece'].decode(),
                                 layer=int(layer), head=int(head), mean_target_minus_other=float(mean[category, metric, token, layer, head]),
                                 standardized_effect=stat,
                                 global_shuffle_tail_fraction=float(np.mean(null_max >= abs(stat) - 1e-12)),
                                 s0_target_rank_fraction=first_ranks[:, category, metric, token, layer, head].tolist()))
    result = dict(histories=len(differences), query_results=accuracy, findings=findings,
                  arrays='Each query NPZ has [query token, layer, head, story token] dot and cosine arrays.',
                  null='All 256 paired target-label sign flips; maximum absolute statistic across both fields, all metrics, query tokens, layers and heads. This is a descriptive shuffle control on eight crossed histories, not independent population samples.',
                  limitation='Original write keys can be transformed by later recurrent updates. Alignment is observational and does not establish causal retrieval.')
    (args.dataset / 'key-alignment.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
