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

def plot_expectations(results, model_name, cases_pair, short_name,
                      out_dir, xticks, ylim=None, ylabel='Expected reward',
                      case_labels=('No context change', 'Context change')):
    """Expected reward (MU) over trials, one line per case.

    Analogous to plot_associative_strength but with configurable ylabel
    and legend labels — Renewal uses 'Expected reward' and condition
    names that reflect the task design.
    """
    from matplotlib import pyplot as plt
    from .style import (COLOR_PARTIAL, COLOR_CONTINUOUS, LINE_W,
                        FONT_SIZE_LABEL, pretty_axes, styled_figure, save)

    fig, ax = styled_figure()

    plotted = []
    for case, color in zip(cases_pair, [COLOR_PARTIAL, COLOR_CONTINUOUS]):
        mu = results['MU'][(model_name, case)]['mean']   # (T, D*A)
        ax.plot(mu[:, 0], color=color, linewidth=LINE_W, label=case_labels[cases_pair.index(case)])
        plotted.append(mu[:, 0])

    plotted = np.concatenate(plotted)
    if ylim is None:
        data_min = float(plotted.min())
        data_max = float(plotted.max())
        margin = 0.05 * max(data_max - data_min, 1e-6)
        ylim = (data_min - margin, data_max + margin)

    ntrials = results['MU'][(model_name, cases_pair[0])]['mean'].shape[0]
    pretty_axes(ax, xlim=(0.5, ntrials + 0.5), ylim=ylim, xticks=xticks)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=FONT_SIZE_LABEL)
    ax.legend(loc='best', fontsize=12)

    save(fig, out_dir / f'Expectations_Renewal_{short_name}.png')


def plot_mmkf_tracking(results, model_name, case, short_name,
                       out_dir, xticks, par_kf_L=None):
    """3-panel Prior / Likelihood / Posterior from MMKF, per state.

    One figure per (model, case). Each of the three panels plots one
    line per latent state, matching MATLAB's
    figure(8 + l*100 + j) block in RunRenewal.m.
    """
    from matplotlib import pyplot as plt

    key = (model_name, case)
    if key not in results['MMKF_PRIOR']:
        return

    # (T, L_kf, nrep) → (T, L_kf) by averaging over replicates
    prior = results['MMKF_PRIOR'][key].mean(axis=-1)
    like  = results['MMKF_LIKE'][key].mean(axis=-1)
    post  = results['MMKF_POST'][key].mean(axis=-1)

    ntrials, L_kf = prior.shape
    if par_kf_L is not None:
        L_kf = min(L_kf, par_kf_L)

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    panels = [(prior, 'Prior'), (like, 'Likelihood'), (post, 'Posterior')]
    for ax, (data, name) in zip(axes, panels):
        for l in range(L_kf):
            ax.plot(data[:, l], linewidth=2, label=f'State {l+1}')
        ax.set_ylabel(name, fontweight='bold', fontsize=14)
        ax.set_xlim(1, ntrials)

    axes[-1].set_xlabel('Trial', fontweight='bold', fontsize=14)
    axes[-1].set_xticks(xticks)
    axes[0].set_title(f'{short_name} — Case {case} (MMKF Tracking)',
                      fontsize=14)
    axes[0].legend(loc='center left', bbox_to_anchor=(1.0, 0.5),
                   frameon=False, fontsize=11)

    fig.tight_layout()
    fig.savefig(out_dir / f'MMKF_Tracking_Renewal_{short_name}_Case{case}.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)
