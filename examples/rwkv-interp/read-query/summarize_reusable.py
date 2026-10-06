"""Test transfer of fixed natural read-query trajectories between stories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.output
    if not (root / 'DONE').exists():
        raise ValueError('Experiment incomplete')
    df = pd.read_csv(root / 'scores.tsv', sep='\t')
    if len(df) != 2272 or df.duplicated(['case', 'kind', 'start_token', 'layer', 'mode']).any():
        raise ValueError('Missing or duplicated cases')
    if not np.isfinite(df.select_dtypes('number').to_numpy()).all():
        raise ValueError('Nonfinite result')
    controls = df[df.kind == 'self']
    if len(controls) != 1040 or not (controls.margin == controls.baseline_margin).all():
        raise ValueError('Self controls failed')
    df['correct'] = np.sign(df.margin) == np.sign(df.baseline_margin)
    df['correct_margin'] = np.sign(df.baseline_margin)*df.margin
    df['baseline_correct_margin'] = df.baseline_margin.abs()
    df['margin_change'] = df.correct_margin - df.baseline_correct_margin
    groups = []
    for (start, mode, kind), sub in df[(df.layer == 61) & (df.kind != 'self')].groupby(['start_token', 'mode', 'kind']):
        groups.append(dict(start_token=int(start), mode='full_r' if mode == 1 else 'old_state_query', donor_change=kind,
                           n=len(sub), correct=int(sub.correct.sum()), mean_correct_margin=float(sub.correct_margin.mean()),
                           mean_margin_change=float(sub.margin_change.mean())))
    single = df[(df.layer < 61) & (df.kind == 'cities')].groupby('layer').agg(correct=('correct', 'sum'), mean_margin_change=('margin_change', 'mean'))
    single.to_csv(root / 'individual-layers.tsv', sep='\t')
    report = dict(runs=len(df), exact_self_controls=len(controls), all_layer_results=groups,
                  individual_layer_total_correct=int(single.correct.sum()), individual_layer_total=976,
                  baseline_correct=16, baseline_mean_correct_margin=float(df.groupby('case').baseline_correct_margin.first().mean()),
                  definition='Accuracy is forced choice between yellow and blue. Full r changes every use of r. Old-state query replacement changes only the historical contribution, leaving immediate/shortcut uses intact. No donor state is copied; recipient state evolves normally. Layer 61 is the all-layer sentinel.',
                  limitations='Fixed two-person, two-city, two-color template. Failure rejects transfer of these particular natural query vectors, not arbitrary context-dependent joint queries. Success does not prove that queries or representations are literally city plus attribute, nor exclude additional routes.')
    (root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    kinds = ['cities', 'colors', 'cities_colors', 'order']
    for ax, start in zip(axes[:2], [0, 9]):
        for mode, offset, label in [(1, -0.18, 'Full r'), (2, 0.18, 'Historical read only')]:
            sub = df[(df.layer == 61) & (df.start_token == start) & (df['mode'] == mode) & (df.kind != 'self')]
            values = sub.groupby('kind').correct.sum().reindex(kinds)
            ax.bar(np.arange(4) + offset, values, width=0.36, label=label)
        ax.set(xticks=np.arange(4), xticklabels=['Cities', 'Colors', 'Both', 'Order'], ylim=(0, 17), ylabel='Recipient-correct answers / 16', title=f'All layers: from query token {start}')
        ax.legend(fontsize=8)
    axes[2].plot(single.index, single.correct)
    axes[2].set(xlabel='Layer', ylabel='Recipient-correct answers / 16', ylim=(0, 17), title='Single-layer full r swap, from city token')
    fig.savefig(root / 'reusable-query.png', dpi=150, bbox_inches='tight')
    p = root / 'provenance.json'
    provenance = json.loads(p.read_text()) if p.exists() else {}
    provenance[__file__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    p.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
