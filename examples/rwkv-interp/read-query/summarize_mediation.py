"""Summarize source-read and later-query interventions with a fixed color split."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def describe(frame):
    return dict(n=len(frame), mean_recovery=float(frame.recovery.mean()),
                min_recovery=float(frame.recovery.min()), max_recovery=float(frame.recovery.max()),
                donor_answer_count=int((np.sign(frame.margin) == np.sign(frame.donor_margin)).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.output
    if not (root / 'DONE').exists():
        raise ValueError('Experiment is incomplete')
    df = pd.read_csv(root / 'scores.tsv', sep='\t')
    chosen = dict(line.split('\t') for line in (root / 'selected.tsv').read_text().splitlines())
    source = int(chosen['source_layer'])
    if len(df) != 16*61*2 + 32 + 16*4*(60 - source):
        raise ValueError('Missing intervention cases')
    if df.duplicated(['case', 'stage', 'source_layer', 'mediator_layer']).any():
        raise ValueError('Duplicate intervention cases')
    if not np.isfinite(df.select_dtypes('number').to_numpy()).all():
        raise ValueError('Nonfinite result')
    for stage in ('self', 'r_self'):
        control = df[df.stage == stage]
        if not (control.margin == control.baseline_margin).all():
            raise ValueError(f'{stage}: nonzero control effect')
    old = df[(df.stage == 'old_read') & (df.source_layer == source)].set_index('case')
    repeat = df[df.stage == 'source_repeat'].set_index('case')
    np.testing.assert_array_equal(old.margin, repeat.margin)
    replay = df[df.stage == 'r_replay']
    np.testing.assert_array_equal(replay.margin, replay['case'].map(old.margin))
    candidates = df[df.stage == 'restore_old_query'].copy()
    candidates['blocked_recovery'] = candidates['case'].map(old.recovery) - candidates.recovery
    training = candidates[candidates['case'] < 8].groupby('mediator_layer').blocked_recovery.mean()
    mediator = int(training.idxmax())
    if (root / 'mediator-selection.json').exists():
        if mediator != json.loads((root / 'mediator-selection.json').read_text())['layer']:
            raise ValueError('Mediator selection changed')
    report = dict(source_layer=source, mediator_layer=mediator, total_runs=len(df),
                  selection='Source maximizes mean old-read recovery over layers 0-59 on cases 0-7. Mediator maximizes reduction of recovery under old-state-query restoration on cases 0-7. Cases 8-15 reverse the colors and are excluded from both choices.',
                  controls=dict(self_runs=int(df.stage.isin(['self', 'r_self']).sum()), replay_runs=len(replay), passed=True), splits={})
    sweep = df[df.stage == 'old_read']
    report['source_sweep'] = describe(sweep)
    report['source_sweep']['largest_absolute_individual_recovery'] = float(sweep.recovery.abs().max())
    changes = pd.read_csv(root / 'read-changes.tsv', sep='\t')
    downstream = changes[(changes.layer > source) & (changes.field == 0)]
    report['later_r'] = {}
    for name, mask in [('selection', downstream['case'] < 8), ('color_reversal', downstream['case'] >= 8)]:
        means = downstream[mask].groupby(['token', 'layer']).relative_change.mean()
        token, layer = means.idxmax()
        report['later_r'][name] = dict(peak_token=int(token), peak_layer=int(layer), peak_mean_relative_change=float(means.max()))
    for split, condition in [('selection', df['case'] < 8), ('color_reversal', df['case'] >= 8)]:
        sub = df[condition]
        result = {}
        for stage in ('old_read', 'current_read', 'restore_r', 'restore_old_query'):
            rows = sub[sub.stage == stage]
            rows = rows[rows.source_layer == source] if stage == 'old_read' else rows
            if stage.startswith('restore'):
                rows = rows[rows.mediator_layer == mediator].copy()
                rows['blocked'] = rows['case'].map(old.recovery) - rows.recovery
            result[stage] = describe(rows)
            if stage.startswith('restore'):
                result[stage]['mean_blocked_recovery'] = float(rows.blocked.mean())
                result[stage]['cases_recovery_reduced'] = int((rows.blocked > 0).sum())
        report['splits'][split] = result
    report['limitations'] = 'Eight fixed crossed stories. Color reversal is a transfer check, not an independent population sample. Read replacement changes a full-layer old-state read; it does not isolate person identity. Restoring r can create a mixed-context computation. Even selective blocking does not establish an exclusive mechanism or prove semantic multi-hop retrieval.'
    (root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name, mask, style in [('selection', df['case'] < 8, '-'), ('color reversal', df['case'] >= 8, '--')]:
        subset = df[mask]
        curve = subset[subset.stage == 'old_read'].groupby('source_layer').recovery.mean()
        axes[0].plot(curve.index, curve, style, label=name)
        for stage, color in [('restore_r', 'tab:blue'), ('restore_old_query', 'tab:orange')]:
            curve = subset[subset.stage == stage].groupby('mediator_layer').recovery.mean()
            axes[1].plot(curve.index, curve, style, color=color, label=f'{name}: {stage}')
        reference = subset[subset.stage == 'source_repeat'].recovery.mean()
        axes[1].axhline(reference, color='gray', linestyle=style)
    axes[0].axvline(source, color='black', alpha=0.3)
    axes[1].axvline(mediator, color='black', alpha=0.3)
    for ax in axes:
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set(ylabel='Mean recovery toward donor answer', xlabel='Layer')
        ax.legend(fontsize=8)
    axes[0].set_title('Old-state read replacement at query city token')
    axes[1].set_title(f'Later query restoration after source layer {source}')
    fig.savefig(root / 'mediation.png', dpi=160, bbox_inches='tight')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
