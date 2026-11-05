"""Reusable plot builders for PREE-style experiments.

Each function takes a ``results`` dict (from run_example) and writes one
figure per (case-pair) to ``out_dir``.
"""

from pathlib import Path

import numpy as np

from .style import (
    COLOR_PARTIAL, COLOR_CONTINUOUS, LINE_W, FONT_SIZE_LABEL,
    pretty_axes, styled_figure, save,
)

def plot_associative_strength(results, model_name, cases_pair, pair_idx,
                              short_name, out_dir, xticks):
    fig, ax = styled_figure()

    all_values = []
    for case, color in zip(cases_pair, [COLOR_PARTIAL, COLOR_CONTINUOUS]):
        mu = results['MU'][(model_name, case)]['mean']    # (T, D*A)
        ax.plot(mu[:, 0], color=color, linewidth=LINE_W)
        all_values.append(mu[:, 0])

    all_values = np.concatenate(all_values)
    ymin = float(np.min(all_values))
    ymax = float(np.max(all_values))
    margin = 0.05 * max(ymax - ymin, 1e-6)

    ntrials = results['MU'][(model_name, cases_pair[0])]['mean'].shape[0]
    pretty_axes(ax,
                xlim=(0.5, ntrials + 0.5),
                ylim=(ymin - margin, ymax + margin),
                xticks=xticks)
    ax.set_ylabel('Associative strength',
                  fontweight='bold', fontsize=FONT_SIZE_LABEL)

    save(fig, out_dir / f'AssociativeStrength_{short_name}{pair_idx}.png')

def plot_beliefs(results, model_name, cases_pair, pair_idx,
                 short_name, out_dir, xticks, ylim=None):
    """Belief time series for the most-active latent state, per case.

    y-limits:
      - If ``ylim`` is passed, use it.
      - Otherwise, derive from data but never narrower than [0, 1],
        since beliefs are probabilities. The union means the plot always
        shows the full probability range even if data never hits both
        endpoints, which keeps figures comparable across experiments.

    Note: MATLAB sorts LSB *per case* by its own mean, which can pick
    different latent-state indices for each condition. That's why we
    compute ``dominant`` inside the loop.
    """
    fig, ax = styled_figure()

    plotted = []
    for case, color, ls in zip(
        cases_pair,
        [COLOR_PARTIAL, COLOR_CONTINUOUS],
        ['-', '--'],
    ):
        lsb = results['LSB'][(model_name, case)]['mean']    # (T+1, L*ncop)
        dominant = int(np.argmax(lsb.mean(axis=0)))
        ax.plot(lsb[:, dominant], linestyle=ls, color=color,
                linewidth=LINE_W)
        plotted.append(lsb[:, dominant])

    plotted = np.concatenate(plotted)
    data_ymin = float(np.min(plotted))
    data_ymax = float(np.max(plotted))

    if ylim is None:
        # Union of [0, 1] with the data range. In practice data stays
        # within [0, 1] because beliefs are normalized, but the union
        # guarantees no clipping even if a model returns slightly out
        # of range (e.g. due to numerical drift).
        ymin = min(0.0, data_ymin)
        ymax = max(1.0, data_ymax)
        ylim = (ymin, ymax)

    ntrials = results['MU'][(model_name, cases_pair[0])]['mean'].shape[0]
    pretty_axes(ax, xlim=(0.5, ntrials + 0.5), ylim=ylim, xticks=xticks)
    ax.set_ylabel('Beliefs', fontweight='bold', fontsize=FONT_SIZE_LABEL)

    save(fig, out_dir / f'Beliefs_{short_name}{pair_idx}.png')


def plot_sampled_belief(results, model_name, cases_pair, pair_idx,
                        short_name, out_dir, xticks, ylim=None):
    """Sampled state x_t over trials, per case.

    y-limits:
      - If ``ylim`` is passed, use it.
      - Otherwise, derive from data as ``(min_state - 0.5, max_state + 0.5)``.
        The half-integer padding centers each integer state on a gridline
        (matplotlib's default behavior for categorical-looking integer
        series) and prevents the top/bottom lines from being clipped.
        The lower bound is floored at 0.5 so a never-sampled-state model
        (all NaN) still produces a readable axis.

    Note: the MATLAB original plotted ``XT{k}`` twice (same data, two
    colors) and labelled the two lines 'Partial' and 'Continuous'. That
    was a bug. This version plots ``results['XT'][(model_name, case)]``
    for each case, so the two lines genuinely come from two conditions.
    """
    if not all((model_name, c) in results['XT'] for c in cases_pair):
        return   # this model didn't sample

    fig, ax = styled_figure()

    plotted = []
    for case, color, ls in zip(
        cases_pair,
        [COLOR_PARTIAL, COLOR_CONTINUOUS],
        ['-', '--'],
    ):
        x_t = results['XT'][(model_name, case)]    # (T, nrep)
        ax.plot(x_t.mean(axis=1), linestyle=ls, color=color,
                linewidth=LINE_W)
        plotted.append(x_t)

    plotted = np.concatenate(plotted)
    if ylim is None:
        if np.all(np.isnan(plotted)):
            ylim = (0.5, 1.5)
        else:
            data_min = float(np.nanmin(plotted))
            data_max = float(np.nanmax(plotted))
            ylim = (max(0.5, data_min - 0.5), data_max + 0.5)

    ntrials = results['MU'][(model_name, cases_pair[0])]['mean'].shape[0]
    pretty_axes(ax, xlim=(0.5, ntrials + 0.5), ylim=ylim, xticks=xticks)
    ax.set_ylabel('Sampled belief',
                  fontweight='bold', fontsize=FONT_SIZE_LABEL)

    save(fig, out_dir / f'Sampled_belief_{short_name}{pair_idx}.png')

