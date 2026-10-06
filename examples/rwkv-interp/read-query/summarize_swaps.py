"""Summarize full-layer swaps and downstream full-vector read changes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    if not (args.root / 'DONE').exists():
        raise ValueError('Experiment is incomplete')
    scores = pd.read_csv(args.root / 'scores.tsv', sep='\t')
    reads = pd.read_csv(args.root / 'later-reads.tsv', sep='\t')
    if len(scores) != 3904 or scores.duplicated(['story', 'recipient', 'mode', 'patch_token', 'patch_layer']).any():
        raise ValueError('Expected 3904 distinct interventions')
    if not np.isfinite(scores.select_dtypes('number').to_numpy()).all():
        raise ValueError('Nonfinite scores')
    controls = scores[scores['mode'] == 'self']
    if not (controls['patched_margin'] == controls['baseline_margin']).all():
        raise ValueError('Self control changes scores')
    if not (reads[reads['mode'] == 'self']['relative_change'] == 0).all():
        raise ValueError('Self control changes reads')
    if not (reads[reads['layer'] <= reads['patch_layer']]['relative_change'] == 0).all():
        raise ValueError('Swap changes an upstream read')
    cross = scores[scores['mode'] == 'cross'].copy()
    terminal = cross[(cross['patch_token'] == 11) & (cross['patch_layer'] == 60)]
    if not (terminal['patched_margin'] == terminal['donor_margin']).all():
        raise ValueError('Terminal-output swap does not reproduce donor scores')
    cross['donor_answer'] = np.sign(cross['patched_margin']) == np.sign(cross['donor_margin'])
    cross['baseline_answer'] = np.sign(cross['patched_margin']) == np.sign(cross['baseline_margin'])
    summary = cross.groupby(['patch_token', 'patch_layer']).agg(
        mean_recovery=('recovery', 'mean'), minimum_recovery=('recovery', 'min'), maximum_recovery=('recovery', 'max'),
        donor_answers=('donor_answer', 'sum'), cases=('donor_answer', 'size'))
    summary.to_csv(args.root / 'layer-summary.tsv', sep='\t')
    later = reads[(reads['mode'] == 'cross') & (reads['observed_token'] == 11)]
    propagation = later.groupby(['patch_token', 'patch_layer', 'layer', 'field']).agg(
        mean_relative_change=('relative_change', 'mean'),
        mean_donor_direction_projection=('donor_direction_projection', 'mean'))
    propagation.to_csv(args.root / 'propagation-summary.tsv', sep='\t')
    report = dict(cross_cases=len(cross), self_cases=len(controls), controls='Exact self activation/logit replay and unchanged upstream reads are checked by the runner.',
                  baseline_margin='log P(blue) - log P(yellow)',
                  recovery='(patched margin - recipient margin) / (opposite-query margin - recipient margin). 0 is recipient; 1 is donor. Values can be outside [0,1].',
                  projection='Projection of read-vector change onto donor-minus-recipient vector, divided by its squared norm; this is descriptive and does not assume a semantic direction.',
                  scope='Full 4096-coordinate block output replaced at one token. All later computation reruns. With fixed input tokens, reads at or below the replaced layer remain unchanged; later layers can change their recurrent state.',
                  limitation='Transplanting a full residual output can carry general query identity. This does not establish a relation-specific algorithm or prove higher-order retrieval.',
                  layers=summary.reset_index().to_dict(orient='records'))
    (args.root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    for token, label in [(9, 'city'), (11, 'colon')]:
        block = summary.loc[token]
        axes[0].plot(block.index, block['mean_recovery'], label=label)
    axes[0].axhline(0, color='gray', linewidth=0.8)
    axes[0].axhline(1, color='gray', linewidth=0.8)
    axes[0].set(xlabel='Swapped layer', ylabel='Mean donor-margin recovery', title='Effect on answer scores')
    axes[0].legend()
    for ax, token, label in zip(axes[1:], [9, 11], ['city', 'colon']):
        matrix = propagation.loc[(token, slice(None), slice(None), 'r'), 'mean_relative_change'].reset_index().pivot(index='layer', columns='patch_layer', values='mean_relative_change')
        im = ax.imshow(matrix.to_numpy(), origin='lower', aspect='auto', cmap='magma')
        ax.set(xlabel='Swapped layer', ylabel='Measured layer', title=f'Final read-vector change after {label} swap')
        fig.colorbar(im, ax=ax, label='Relative L2 change, full layer')
    fig.savefig(args.root / 'layer-swaps.png', dpi=160)
    print(json.dumps({k: v for k, v in report.items() if k != 'layers'}, indent=2))


if __name__ == '__main__':
    main()
