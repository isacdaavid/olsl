"""Reusable plot builders.

Each function takes a ``results`` dict (from run_example) and writes one
figure per (case-pair) to ``out_dir``.

Belief-producing functions (beliefs, sampled states, MMKF panels) accept
``heatmap=True`` to additionally produce a heatmap variant, where y-axis
is latent state, x-axis is trial, and color encodes belief probability.
Line-plot behavior is unchanged.
"""

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from .style import (
    COLOR_PARTIAL, COLOR_CONTINUOUS, COLOR_NEUTRAL, COLOR_SHADE, CUE_COLORS,
    DEFAULT_ACCENT, LINE_W, LINE_ALPHA, _ls, FONT_SIZE_LABEL,
    SCHEDULE_CUE_COLORS, MM_CASE_COLORS,
    pretty_axes, styled_figure, save,
)


# =============================================================================
# Heatmap helpers
# =============================================================================

def _sequential_cmap(color, name='case_seq'):
    """White → color linear colormap matching the case's line color.

    Used for probability heatmaps so the intensity reads as "amount of
    belief", and the hue matches the corresponding line in the paired
    time-series plot.
    """
    return LinearSegmentedColormap.from_list(name, ['white', color])


def _used_states(lsb_list, threshold=1e-3):
    """Number of latent states actually engaged across a list of belief arrays.

    ``lsb_list``: sequence of arrays of shape (T+1, L).
    Returns the maximum ``k`` such that some state ``k`` had belief
    ``> threshold`` on at least one trial, across any case. Guarantees
    ``>= 1`` so a perfectly uniform run still shows one row.
    """
    max_used = 1
    for lsb in lsb_list:
        used = int(np.sum(np.max(lsb, axis=0) > threshold))
        max_used = max(max_used, used)
    return max_used


def _draw_heatmap(ax, data_2d, y_labels, x_range, cmap,
                  vmin=0.0, vmax=1.0, xticks=None, xlim=None):
    """Draw a heatmap where y = category, x = time.

    Parameters
    ----------
    data_2d : ndarray, shape (n_rows, n_cols)
        Rows map to ``y_labels``; columns map to time (trials).
    x_range : (x_min, x_max)
        Range for the imshow extent (data coordinates).
    cmap : str or Colormap
        Colormap for the heatmap.
    """
    n_rows, n_cols = data_2d.shape
    extent = (x_range[0], x_range[1], 0.5, n_rows + 0.5)
    im = ax.imshow(data_2d, aspect='auto', origin='lower', extent=extent,
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_yticks(np.arange(1, n_rows + 1))
    ax.set_yticklabels(y_labels)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel('Trial', fontweight='bold', fontsize=12)
    ax.set_ylabel('Latent state', fontweight='bold', fontsize=12)
    return im


# =============================================================================
# PREE-family belief plots
# =============================================================================

def plot_associative_strength(results, model_name, cases_pair, pair_idx,
                              short_name, out_dir, xticks):
    fig, ax = styled_figure()

    all_values = []
    for i, (case, color) in enumerate(
        zip(cases_pair, [COLOR_PARTIAL, COLOR_CONTINUOUS])
    ):
        mu = results['MU'][(model_name, case)]['mean']    # (T, D*A)
        ax.plot(mu[:, 0], color=color, linewidth=LINE_W, alpha=LINE_ALPHA,
                linestyle=_ls(i), label=f'Case {case}')
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
    ax.legend(loc='best', fontsize=11)

    save(fig, out_dir / f'AssociativeStrength_{short_name}{pair_idx}.png')


def plot_beliefs(results, model_name, cases_pair, pair_idx,
                 short_name, out_dir, xticks, ylim=(-0.1, 1.1),
                 heatmap=False, cmap=None):
    """Belief time series for the most-active latent state, per case.

    y-limits default to (-0.1, 1.1): a small margin below 0 and above 1
    so beliefs that saturate at the probability boundaries remain
    visually distinguishable from the axis edges. Pass ``ylim`` to
    override.

    Note: MATLAB sorts LSB *per case* by its own mean, which can pick
    different latent-state indices for each condition. That's why we
    compute ``dominant`` inside the loop.

    When ``heatmap=True``, produces an additional vertically stacked
    figure showing *all* used latent states per case (y=state, x=trial,
    color=belief). Each subplot's colormap is a white→case-color gradient
    matching the paired line plot. Pass ``cmap`` to override with a single
    fixed colormap for all subplots. Saved as
    ``Beliefs_{short_name}{pair_idx}_heatmap.png``.
    """
    if heatmap:
        lsb_list = [results['LSB'][(model_name, c)]['mean']
                    for c in cases_pair]
        L_used = _used_states(lsb_list)
        n_cases = len(cases_pair)
        fig, axes = plt.subplots(n_cases, 1, figsize=(8, 1.6 * L_used * n_cases + 1),
                                  squeeze=False)
        case_colors = [COLOR_PARTIAL, COLOR_CONTINUOUS]
        for i, (case, lsb) in enumerate(zip(cases_pair, lsb_list)):
            ax = axes[i, 0]
            T = lsb.shape[0] - 1
            data = lsb[1:, :L_used].T          # (L_used, T)
            y_labels = [str(l) for l in range(1, L_used + 1)]
            sub_cmap = cmap if cmap is not None else _sequential_cmap(case_colors[i])
            im = _draw_heatmap(ax, data, y_labels, (0.5, T + 0.5),
                               cmap=sub_cmap, xticks=xticks)
            ax.set_title(f'Case {case}', fontsize=11)
            fig.colorbar(im, ax=ax, label='Belief')
        fig.tight_layout()
        save(fig, out_dir / f'Beliefs_{short_name}{pair_idx}_heatmap.png')
        return

    fig, ax = styled_figure()

    for i, (case, color) in enumerate(
        zip(cases_pair, [COLOR_PARTIAL, COLOR_CONTINUOUS])
    ):
        lsb = results['LSB'][(model_name, case)]['mean']    # (T+1, L*ncop)
        dominant = int(np.argmax(lsb.mean(axis=0)))
        ax.plot(lsb[:, dominant], linestyle=_ls(i), color=color,
                linewidth=LINE_W, alpha=LINE_ALPHA,
                label=f'Case {case} (state {dominant + 1})')

    ntrials = results['MU'][(model_name, cases_pair[0])]['mean'].shape[0]
    pretty_axes(ax, xlim=(0.5, ntrials + 0.5), ylim=ylim, xticks=xticks)
    ax.set_ylabel('Beliefs', fontweight='bold', fontsize=FONT_SIZE_LABEL)
    ax.legend(loc='best', fontsize=11)

    save(fig, out_dir / f'Beliefs_{short_name}{pair_idx}.png')


def plot_sampled_belief(results, model_name, cases_pair, pair_idx,
                        short_name, out_dir, xticks, ylim=None,
                        heatmap=False, cmap=None):
    """Sampled state x_t over trials, per case.

    y-limits:
      - If ``ylim`` is passed, use it.
      - Otherwise, derive from data as ``(min_state - 0.5, max_state + 0.5)``.
        The half-integer padding centers each integer state on a gridline
        (matplotlib's default behavior for categorical-looking integer
        series) and prevents the top/bottom lines from being clipped.
        The lower bound is floored at 0.5 so a never-sampled-state model
        (all NaN) still produces a readable axis.

    NOTE: unlike ``plot_beliefs``, this plots sampled integer state
    indices (1, 2, 3, …), not probabilities. The probability-oriented
    default (-0.1, 1.1) does not apply here; the data-derived bounds
    are the correct choice.

    Note: the MATLAB original plotted ``XT{k}`` twice (same data, two
    colors) and labelled the two lines 'Partial' and 'Continuous'. That
    was a bug. This version plots ``results['XT'][(model_name, case)]``
    for each case, so the two lines genuinely come from two conditions.

    When ``heatmap=True``, produces an additional figure showing the
    sampled state as a (case × trial) strip heatmap, saved as
    ``Sampled_belief_{short_name}{pair_idx}_heatmap.png``. Sampled-state
    strips are categorical (state id, not probability), so a continuous
    probability colormap does not apply; ``viridis`` (or ``cmap``) is used.
    """
    if not all((model_name, c) in results['XT'] for c in cases_pair):
        return   # this model didn't sample

    if heatmap:
        matrices = []
        for case in cases_pair:
            x_t = results['XT'][(model_name, case)]    # (T, nrep)
            matrices.append(x_t.mean(axis=1))          # (T,)
        data = np.vstack(matrices)                     # (n_cases, T)
        n_cases, T = data.shape
        L_max = int(np.nanmax(data)) + 1 if not np.all(np.isnan(data)) else 2

        fig, ax = plt.subplots(figsize=(10, 1.5 + 0.5 * n_cases))
        im = ax.imshow(data, aspect='auto', origin='lower',
                       extent=(0.5, T + 0.5, 0.5, n_cases + 0.5),
                       cmap=cmap or 'viridis',
                       vmin=-0.5, vmax=L_max - 0.5,
                       interpolation='nearest')
        ax.set_yticks(np.arange(1, n_cases + 1))
        ax.set_yticklabels([f'Case {c}' for c in cases_pair])
        ax.set_ylabel('Case', fontweight='bold', fontsize=12)
        ax.set_xlabel('Trial', fontweight='bold', fontsize=12)
        ax.set_xticks(xticks)
        cbar = fig.colorbar(im, ax=ax, label='Sampled state',
                            ticks=np.arange(0, L_max))
        cbar.ax.set_yticklabels([str(i + 1) for i in range(L_max)])
        fig.tight_layout()
        save(fig, out_dir
             / f'Sampled_belief_{short_name}{pair_idx}_heatmap.png')
        return

    fig, ax = styled_figure()

    plotted = []
    for i, (case, color) in enumerate(
        zip(cases_pair, [COLOR_PARTIAL, COLOR_CONTINUOUS])
    ):
        x_t = results['XT'][(model_name, case)]    # (T, nrep)
        ax.plot(x_t.mean(axis=1), linestyle=_ls(i), color=color,
                linewidth=LINE_W, alpha=LINE_ALPHA,
                label=f'Case {case}')
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
    ax.legend(loc='best', fontsize=11)

    save(fig, out_dir / f'Sampled_belief_{short_name}{pair_idx}.png')


def plot_expectations(results, model_name, cases_pair, short_name,
                      out_dir, xticks, ylim=None, ylabel='Expected reward',
                      case_labels=('No context change', 'Context change')):
    """Expected reward (MU) over trials, one line per case.

    Analogous to plot_associative_strength but with configurable ylabel
    and legend labels — Renewal uses 'Expected reward' and condition
    names that reflect the task design.
    """
    fig, ax = styled_figure()

    plotted = []
    for i, (case, color) in enumerate(
        zip(cases_pair, [COLOR_PARTIAL, COLOR_CONTINUOUS])
    ):
        mu = results['MU'][(model_name, case)]['mean']   # (T, D*A)
        ax.plot(mu[:, 0], color=color, linewidth=LINE_W, alpha=LINE_ALPHA,
                linestyle=_ls(i),
                label=case_labels[cases_pair.index(case)])
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
                       out_dir, xticks, par_kf_L=None,
                       heatmap=False, cmap=None):
    """3-panel Prior / Likelihood / Posterior from MMKF, per state.

    One figure per (model, case). Each of the three panels plots one
    line per latent state, matching MATLAB's
    figure(8 + l*100 + j) block in RunRenewal.m.

    State colors cycle through the package's CUE_COLORS palette and
    state linestyles cycle through the LINESTYLES progression, so state
    identity is visually consistent across plots.

    NOTE: y-axis is left to autoscale because the three panels have
    different scales (prior/posterior are probabilities in [0, 1];
    likelihood is a density that can exceed 1). A shared (-0.1, 1.1)
    limit would clip the likelihood panel.

    When ``heatmap=True``, produces a heatmap variant where each panel
    is (state × trial). The colormap is a white→DEFAULT_ACCENT gradient
    (MMKF has one case per figure, so there is no per-case color from the
    line plot to inherit; DEFAULT_ACCENT matches the package-wide accent
    used for single-series plots). Saved as
    ``MMKF_Tracking_Renewal_{short_name}_Case{case}_heatmap.png``.
    """
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

    if heatmap:
        sub_cmap = cmap if cmap is not None else _sequential_cmap(DEFAULT_ACCENT)
        fig, axes = plt.subplots(3, 1, figsize=(10, 9))
        panels = [
            (prior[:ntrials, :L_kf],  'Prior',      1.0),
            (like[:ntrials, :L_kf],   'Likelihood', float(np.nanmax(like[:ntrials, :L_kf])) or 1.0),
            (post[:ntrials, :L_kf],   'Posterior',  1.0),
        ]
        for ax, (data_panel, name, vmax_panel) in zip(axes, panels):
            data_2d = data_panel.T       # (L, T)
            im = ax.imshow(data_2d, aspect='auto', origin='lower',
                           extent=(0.5, ntrials + 0.5, 0.5, L_kf + 0.5),
                           cmap=sub_cmap, vmin=0.0, vmax=vmax_panel,
                           interpolation='nearest')
            ax.set_ylabel(name, fontweight='bold', fontsize=12)
            ax.set_yticks(np.arange(1, L_kf + 1))
            ax.set_yticklabels([str(l) for l in range(1, L_kf + 1)])
            ax.set_xticks(xticks)
            fig.colorbar(im, ax=ax)
        axes[-1].set_xlabel('Trial', fontweight='bold', fontsize=12)
        axes[0].set_title(
            f'{short_name} — Case {case} (MMKF heatmaps)', fontsize=13
        )
        fig.tight_layout()
        save(fig, out_dir
             / f'MMKF_Tracking_Renewal_{short_name}_Case{case}_heatmap.png')
        return

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    panels = [(prior, 'Prior'), (like, 'Likelihood'), (post, 'Posterior')]
    for ax, (data, name) in zip(axes, panels):
        for l in range(L_kf):
            ax.plot(data[:, l], linewidth=2, alpha=LINE_ALPHA,
                    color=CUE_COLORS[l % len(CUE_COLORS)],
                    linestyle=_ls(l),
                    label=f'State {l+1}')
        ax.set_ylabel(name, fontweight='bold', fontsize=14)
        ax.set_xlim(1, ntrials)
        ax.legend(loc='best', fontsize=10)

    axes[-1].set_xlabel('Trial', fontweight='bold', fontsize=14)
    axes[-1].set_xticks(xticks)
    axes[0].set_title(f'{short_name} — Case {case} (MMKF Tracking)',
                      fontsize=14)

    fig.tight_layout()
    fig.savefig(out_dir / f'MMKF_Tracking_Renewal_{short_name}_Case{case}.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_associative_strength_per_cue(results, task_name, model_name,
                                      cue_indices, cue_labels, xticks,
                                      short_name, out_dir,
                                      xlim=None, ylim=None,
                                      cue_colors=None):
    """Plot V[:, cue_indices] over trials, one line per cue.

    The caller is expected to pass a per-task ``out_dir`` — the task name
    is only used for the results-dict key lookup, not for the filename.
    """
    key = (model_name, task_name)
    if key not in results['V']:
        return

    V = results['V'][key]['mean']    # (T+1, D*C*L)
    T = V.shape[0] - 1

    if cue_colors is None:
        cue_colors = CUE_COLORS[:len(cue_indices)]

    fig, ax = styled_figure()
    for i, (idx, label, color) in enumerate(
        zip(cue_indices, cue_labels, cue_colors)
    ):
        ax.plot(V[:, idx], color=color, linewidth=LINE_W, alpha=LINE_ALPHA,
                linestyle=_ls(i), label=label)

    if xlim is None:
        xlim = (0.5, T + 0.5)
    if ylim is None:
        sub = V[:, list(cue_indices)]
        data_min, data_max = float(sub.min()), float(sub.max())
        margin = 0.05 * max(data_max - data_min, 1e-6)
        ylim = (data_min - margin, data_max + margin)

    pretty_axes(ax, xlim=xlim, ylim=ylim, xticks=xticks)
    ax.set_ylabel('Associative strength',
                  fontweight='bold', fontsize=FONT_SIZE_LABEL)
    ax.legend(loc='best', fontsize=11)

    # out_dir is already per-task; filename omits the task name.
    save(fig, out_dir / f'AssociativeStrength_{short_name}.png')


# =============================================================================
# Spontaneous-recovery family (broken axis)
# =============================================================================

def _setup_broken_axis(ax, ntrials, gap_visual=20):
    """X-axis layout used by spontaneous-recovery plots.

    Left half: trials 1..40 at x=1..40.
    Right half: trials 41..ntrials at x=(40+gap_visual+1)..(40+gap_visual+ntrials-40).
    The gap is purely visual — it represents the silent delay, not
    actual trials — and is separated by a white-out on the y=0 line.
    """
    x_break = 40
    x_offset = gap_visual
    x_left  = np.arange(1, x_break + 1)
    x_right = x_offset + np.arange(x_break + 1, ntrials + 1)
    return x_left, x_right, x_break, x_offset


def plot_spontaneous_recovery_expectations(results, model_name, cases,
                                            case_labels, short_name, out_dir,
                                            xticks=None, ylim=None,
                                            case_colors=None):
    """Expected reward (MU) over trials, broken-axis, one line per case."""
    if case_colors is None:
        case_colors = [COLOR_PARTIAL, COLOR_CONTINUOUS]

    fig, ax = styled_figure()

    for i, (case, label, color) in enumerate(
        zip(cases, case_labels, case_colors)
    ):
        key = (model_name, case)
        if key not in results['MU']:
            continue
        mu = results['MU'][key]['mean']    # (T, D*A)
        ntrials = mu.shape[0]
        ls = _ls(i)

        x_left, x_right, x_break, x_offset = _setup_broken_axis(ax, ntrials)

        # Left half: acquisition + extinction
        ax.plot(x_left, mu[:x_break, 0], color=color, linewidth=LINE_W,
                alpha=LINE_ALPHA, linestyle=ls, label=label)
        # Right half: renewal (no label — same series continues)
        ax.plot(x_right, mu[x_break:, 0], color=color, linewidth=LINE_W,
                alpha=LINE_ALPHA, linestyle=ls)
        # End-of-extinction and start-of-renewal markers
        ax.plot(x_break, mu[x_break - 1, 0], 'x', color=color,
                markersize=12, markeredgewidth=3, alpha=LINE_ALPHA)
        ax.plot(x_break + x_offset + 1, mu[x_break, 0], 'x', color=color,
                markersize=12, markeredgewidth=3, alpha=LINE_ALPHA)

    # White rectangle hides the axis spine in the visual gap
    ax.add_patch(Rectangle((40.2, -0.1), 19.6, 0.2,
                           facecolor='white', edgecolor='white', zorder=3))

    if xticks is None:
        xticks = [0, 20, 40, 60, 70]
    ax.set_xticks(xticks)
    ax.set_xticklabels(['0', '20', '40', '0', '10'])
    ax.set_xlim(0, 71)
    if ylim is None:
        ylim = (-0.55, 0.55)
    ax.set_ylim(ylim)
    ax.set_ylabel('Associative strength', fontweight='bold',
                  fontsize=FONT_SIZE_LABEL)
    ax.legend(loc='best', fontsize=12)

    save(fig, out_dir / f'AS_SponRec_{short_name}.png')


def plot_spontaneous_recovery_beliefs(results, model_name, cases,
                                       case_labels, short_name, out_dir,
                                       xticks=None, case_colors=None,
                                       heatmap=False, cmap=None):
    """Dominant-state belief over trials, broken-axis, one line per case.

    When ``heatmap=True``, produces a vertically stacked heatmap variant
    showing *all* used states per case, with the same visual gap between
    extinction (trials 1–40) and renewal (trials 41–T) phases. Each
    subplot's colormap is a white→case-color gradient matching the line
    plot. Saved as ``Beliefs_SponRec_{short_name}_heatmap.png``.
    """
    if case_colors is None:
        case_colors = [COLOR_PARTIAL, COLOR_CONTINUOUS]

    if heatmap:
        lsb_list = [results['LSB'][(model_name, c)]['mean']
                    for c in cases if (model_name, c) in results['LSB']]
        L_used = _used_states(lsb_list)
        n_cases = len(cases)
        fig, axes = plt.subplots(n_cases, 1,
                                  figsize=(10, 1.6 * L_used * n_cases + 1),
                                  squeeze=False)
        for i, case in enumerate(cases):
            ax = axes[i, 0]
            key = (model_name, case)
            if key not in results['LSB']:
                ax.set_visible(False)
                continue
            lsb = results['LSB'][key]['mean']    # (T+1, L)
            T = lsb.shape[0] - 1
            data_full = lsb[1:, :L_used].T       # (L_used, T)
            data_left  = data_full[:, :40]
            data_right = data_full[:, 40:]

            sub_cmap = cmap if cmap is not None else _sequential_cmap(case_colors[i])
            im = ax.imshow(data_left, aspect='auto', origin='lower',
                           extent=(0.5, 40.5, 0.5, L_used + 0.5),
                           cmap=sub_cmap, vmin=0.0, vmax=1.0,
                           interpolation='nearest')
            ax.imshow(data_right, aspect='auto', origin='lower',
                      extent=(60.5, 60.5 + data_right.shape[1],
                              0.5, L_used + 0.5),
                      cmap=sub_cmap, vmin=0.0, vmax=1.0,
                      interpolation='nearest')
            ax.set_xticks([0, 20, 40, 60, 70])
            ax.set_xticklabels(['0', '20', '40', '0', '10'])
            ax.set_xlim(0, 71)
            ax.set_yticks(np.arange(1, L_used + 1))
            ax.set_yticklabels([str(l) for l in range(1, L_used + 1)])
            ax.set_ylabel('Latent state', fontweight='bold', fontsize=11)
            ax.set_title(f'{case_labels[i]}', fontsize=11)
            fig.colorbar(im, ax=ax, label='Belief')
        axes[-1, 0].set_xlabel('Trial', fontweight='bold', fontsize=12)
        fig.tight_layout()
        save(fig, out_dir / f'Beliefs_SponRec_{short_name}_heatmap.png')
        return

    fig, ax = styled_figure()

    for i, (case, label, color) in enumerate(
        zip(cases, case_labels, case_colors)
    ):
        key = (model_name, case)
        if key not in results['LSB']:
            continue
        lsb = results['LSB'][key]['mean']    # (T+1, L*ncop)
        ntrials = lsb.shape[0] - 1

        # Dominant state per case, chosen by maximum mean belief
        dominant = int(np.argmax(lsb.mean(axis=0)))
        ls = _ls(i)

        x_left, x_right, x_break, x_offset = _setup_broken_axis(ax, ntrials)

        # lsb[0] is pre-trial init; lsb[t] is belief after trial t
        ax.plot(x_left, lsb[1:x_break + 1, dominant],
                color=color, linewidth=LINE_W, alpha=LINE_ALPHA,
                linestyle=ls, label=f'{label} (state {dominant + 1})')
        ax.plot(x_right, lsb[x_break + 1:, dominant],
                color=color, linewidth=LINE_W, alpha=LINE_ALPHA,
                linestyle=ls)
        ax.plot(x_break, lsb[x_break, dominant], 'x',
                color=color, markersize=12, markeredgewidth=3,
                alpha=LINE_ALPHA)
        ax.plot(x_break + x_offset + 1, lsb[x_break + 1, dominant], 'x',
                color=color, markersize=12, markeredgewidth=3,
                alpha=LINE_ALPHA)

    ax.add_patch(Rectangle((40.2, -0.1), 19.6, 0.2,
                           facecolor='white', edgecolor='white', zorder=3))

    if xticks is None:
        xticks = [0, 20, 40, 60, 70]
    ax.set_xticks(xticks)
    ax.set_xticklabels(['0', '20', '40', '0', '10'])
    ax.set_xlim(0, 82)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel('Beliefs', fontweight='bold', fontsize=FONT_SIZE_LABEL)
    ax.legend(loc='best', fontsize=12)

    save(fig, out_dir / f'Beliefs_SponRec_{short_name}.png')


def _context_shift_belief(lsb_end, gamma, n_silent, n_active):
    """Analytically apply ContextShift over n_silent silent trials.

    Mirrors LatentState.ContextShift::

        prob_stay = (1 - gamma) ** (n_silent - 1)
        lsb = lsb * prob_stay + (1 - prob_stay) / n_active

    Wait — ContextShift uses (gap - 1), not gap. With gap = case and
    n_silent = case - 1 silent trials, the exponent is n_silent - 1.
    (When case=1 → n_silent=0, ContextShift at trial 41 is skipped
    because Δtime=1. When case=2 → n_silent=1, exponent 0, no decay.
    Consistent.)
    """
    if n_silent <= 0:
        return lsb_end.copy()
    prob_stay = (1 - gamma) ** (n_silent - 1)
    target = 1.0 / n_active
    return lsb_end * prob_stay + (1 - prob_stay) * target


def plot_spontaneous_recovery_dilution(results, model_name, cases,
                                        case_labels, gamma, short_name,
                                        out_dir, state_idx=0,
                                        xticks=None, case_colors=None,
                                        case_linestyles=None,
                                        heatmap=False, cmap=None):
    """Belief in State 1 through the delay, with an analytic decay curve
    bridging the silent gap.

    Each case gets a distinct linestyle (solid for the first, dashed for
    the second, etc.) applied end-to-end, including the gap segment, so
    each case remains traceable across the whole plot.

    Parameters
    ----------
    gamma : float
        LatentState transition probability (par['ls']['gamma']).
    state_idx : int
        Which state to plot. Default 0 = State 1 = acquisition state.
    case_linestyles : list of str, optional
        One linestyle per case. Defaults to the LINESTYLES progression
        (solid, dashed, dash-dot, ...) sliced to len(cases).

    When ``heatmap=True``, produces a vertically stacked heatmap variant
    showing *all* used states per case with the same broken axis. Each
    subplot uses a white→case-color gradient. Saved as
    ``BeliefDilution_SponRec_{short_name}_heatmap.png``.
    """
    if case_colors is None:
        case_colors = [COLOR_PARTIAL, COLOR_CONTINUOUS]
    if case_linestyles is None:
        case_linestyles = [_ls(i) for i in range(len(cases))]

    if heatmap:
        lsb_list = [results['LSB'][(model_name, c)]['mean']
                    for c in cases if (model_name, c) in results['LSB']]
        L_used = _used_states(lsb_list)
        n_cases = len(cases)
        fig, axes = plt.subplots(n_cases, 1,
                                  figsize=(10, 1.6 * L_used * n_cases + 1),
                                  squeeze=False)
        for i, case in enumerate(cases):
            ax = axes[i, 0]
            key = (model_name, case)
            if key not in results['LSB']:
                ax.set_visible(False)
                continue
            lsb = results['LSB'][key]['mean']    # (T+1, L)
            T = lsb.shape[0] - 1
            data_full = lsb[1:, :L_used].T       # (L_used, T)
            data_left  = data_full[:, :40]
            data_right = data_full[:, 40:]

            sub_cmap = cmap if cmap is not None else _sequential_cmap(case_colors[i])
            im = ax.imshow(data_left, aspect='auto', origin='lower',
                           extent=(0.5, 40.5, 0.5, L_used + 0.5),
                           cmap=sub_cmap, vmin=0.0, vmax=1.0,
                           interpolation='nearest')
            ax.imshow(data_right, aspect='auto', origin='lower',
                      extent=(60.5, 60.5 + data_right.shape[1],
                              0.5, L_used + 0.5),
                      cmap=sub_cmap, vmin=0.0, vmax=1.0,
                      interpolation='nearest')
            ax.set_xticks([0, 20, 40, 60, 70])
            ax.set_xticklabels(['0', '20', '40', '0', '10'])
            ax.set_xlim(0, 71)
            ax.set_yticks(np.arange(1, L_used + 1))
            ax.set_yticklabels([str(l) for l in range(1, L_used + 1)])
            ax.set_ylabel('Latent state', fontweight='bold', fontsize=11)
            ax.set_title(f'{case_labels[i]}', fontsize=11)
            fig.colorbar(im, ax=ax, label='Belief')
        axes[-1, 0].set_xlabel('Trial', fontweight='bold', fontsize=12)
        fig.tight_layout()
        save(fig, out_dir
             / f'BeliefDilution_SponRec_{short_name}_heatmap.png')
        return

    fig, ax = styled_figure()

    for case, label, color, ls in zip(cases, case_labels,
                                      case_colors, case_linestyles):
        key = (model_name, case)
        if key not in results['LSB']:
            continue
        lsb = results['LSB'][key]['mean']
        ntrials = lsb.shape[0] - 1
        n_silent = case - 1

        lsb_end_ext = lsb[40, :]
        n_active_case = max(1, int(np.sum(lsb_end_ext > 1e-10)))

        # --- Left half: acquisition + extinction, posterior per trial ---
        x_left = np.arange(1, 41)
        ax.plot(x_left, lsb[1:41, state_idx], color=color,
                linestyle=ls, linewidth=LINE_W, alpha=LINE_ALPHA,
                label=label)

        # --- Right half: renewal, posterior per trial ---
        x_offset = 20
        x_right = x_offset + np.arange(41, ntrials + 1)
        ax.plot(x_right, lsb[41:, state_idx], color=color,
                linestyle=ls, linewidth=LINE_W, alpha=LINE_ALPHA)

        # --- Gap: analytic exponential decay from posterior-40 to prior-41 ---
        # Same linestyle as the surrounding curve, so the case remains
        # traceable end-to-end.
        if n_silent > 0:
            tau = np.linspace(0, n_silent, 200)
            prob_stay_tau = (1 - gamma) ** tau
            target = 1.0 / n_active_case
            belief_tau = (lsb_end_ext[state_idx] * prob_stay_tau
                          + (1 - prob_stay_tau) * target)
            x_gap = 40 + x_offset * tau / n_silent
            ax.plot(x_gap, belief_tau, color=color,
                    linestyle=ls, linewidth=LINE_W, alpha=LINE_ALPHA)

        # --- Endpoint markers ---
        ax.plot(40, lsb[40, state_idx], 'x', color=color,
                markersize=12, markeredgewidth=3, alpha=LINE_ALPHA)
        if n_silent > 0:
            prior_41 = _context_shift_belief(
                lsb_end_ext, gamma, n_silent, n_active_case,
            )[state_idx]
        else:
            prior_41 = lsb[40, state_idx]
        ax.plot(40 + x_offset + 1, prior_41, 'x', color=color,
                markersize=12, markeredgewidth=3, alpha=LINE_ALPHA)

    # Visual gap
    ax.add_patch(Rectangle((40.2, -0.1), 19.6, 0.2,
                           facecolor='white', edgecolor='white', zorder=3))

    if xticks is None:
        xticks = [0, 20, 40, 60, 70]
    ax.set_xticks(xticks)
    ax.set_xticklabels(['0', '20', '40', '0', '10'])
    ax.set_xlim(0, 71)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel(f'Belief in State {state_idx + 1}',
                  fontweight='bold', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Belief dilution during silent delay', fontsize=13)
    ax.legend(loc='best', fontsize=12)

    save(fig, out_dir / f'BeliefDilution_SponRec_{short_name}.png')


# =============================================================================
# Wilson-Boumphrey-Pearce schedule + zoomed plots
# =============================================================================

def plot_cue_schedule(par, cue_names, title, out_dir, filename):
    """Render a cue-present schedule as a colored patch grid.

    Rows are cues (``cue_names``); columns are trials. A cell is filled
    with the cue's color when that cue is present on that trial, and
    white otherwise.

    Uses the first ``len(cue_names)`` columns of ``par['outcomes']`` —
    for LatentState these are [Food, Light, Tone] (interaction is
    column 3, ignored here); for non-LatentState they are all columns.
    """
    trialorder = par['trialorder']
    outcomes   = par['outcomes'][:, :len(cue_names)]
    ntrials    = len(trialorder)
    n_cues     = len(cue_names)

    fig, ax = plt.subplots(figsize=(12, 3))
    for t in range(ntrials):
        row = int(trialorder[t]) - 1
        for j in range(n_cues):
            present = outcomes[row, j] > 0
            color   = SCHEDULE_CUE_COLORS[j] if present else 'white'
            ax.add_patch(Rectangle(
                (t + 0.5, j + 0.5), 1, 1,
                facecolor=color, edgecolor='none', alpha=0.75,
            ))

    ax.set_xlim(0, ntrials)
    ax.set_ylim(0.5, n_cues + 0.5)
    ax.set_yticks(range(1, n_cues + 1))
    ax.set_yticklabels(cue_names)
    ax.set_xticks([20, 40, 60])
    ax.set_xlabel('Trial')
    ax.set_title(title, fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend via Line2D proxies (patches aren't picked up automatically)
    handles = [
        Line2D([0], [0], color=SCHEDULE_CUE_COLORS[j], lw=10,
               alpha=0.75, label=name)
        for j, name in enumerate(cue_names)
    ]
    ax.legend(handles=handles, loc='upper right',
              bbox_to_anchor=(1.0, 1.35), ncol=n_cues, fontsize=10,
              frameon=False)

    save(fig, out_dir / filename)


def plot_cue_strength_zoomed(results, model_name, cases, case_labels,
                              cue_idx, short_name, out_dir,
                              xlim, ylim, xticks, filename_stem):
    """Plot V[:, cue_idx] over trials, one line per case, zoomed to xlim.

    One line per case, so both groups' trajectories for a single cue can
    be compared directly. Distinct from plot_associative_strength_per_cue:
    that one plots several cues from a *single* condition; this one plots
    one cue across *several* conditions.

    Uses V[1:], matching the intent of MATLAB's plot(V(1:end-1, ...)) —
    showing post-trial values (1..ntrials) rather than the pre-trial init.
    """
    fig, ax = styled_figure()

    ntrials = None
    for i, (case, label, color) in enumerate(
        zip(cases, case_labels, [COLOR_CONTINUOUS, COLOR_PARTIAL])
    ):
        key = (model_name, case)
        if key not in results['V']:
            continue
        V = results['V'][key]['mean']
        ntrials = V.shape[0] - 1
        x = np.arange(1, ntrials + 1)
        ax.plot(x, V[1:, cue_idx], color=color,
                linewidth=LINE_W, alpha=LINE_ALPHA, linestyle=_ls(i),
                label=label)

    pretty_axes(ax, xlim=xlim, ylim=ylim, xticks=xticks)
    ax.set_ylabel('Associative strength',
                  fontweight='bold', fontsize=FONT_SIZE_LABEL)
    ax.legend(loc='best', fontsize=11)

    save(fig, out_dir / f'{filename_stem}_{short_name}.png')


def plot_beliefs_top2(results, model_name, cases, case_labels,
                      short_name, out_dir, xlim, xticks, filename_stem,
                      ylim=(-0.1, 1.1), heatmap=False, cmap=None):
    """Plot the top-2 dominant states per case, zoomed to xlim.

    For each case, plots the two states with the highest mean belief.
    Each of the (case × rank) curves gets a distinct linestyle so all
    four (or fewer) trajectories are distinguishable without relying on
    color alone. Matches MATLAB's pattern in RunWilsonBoumphreyPearce.m,
    where dominant-state indices often differ between groups and are
    picked independently per case.

    y-limits default to (-0.1, 1.1) — a small margin around the
    probability range so beliefs that saturate at 0 or 1 remain visually
    distinguishable from the axis edges. Pass ``ylim`` to override.

    When ``heatmap=True``, produces a vertically stacked heatmap variant
    showing *all* used states per case (zoomed to ``xlim``). Each
    subplot's colormap is a white→case-color gradient matching the
    paired line plot (order: continuous, partial — same as the line plot).
    Saved as ``{filename_stem}_{short_name}_heatmap.png``.
    """
    if heatmap:
        lsb_list = [results['LSB'][(model_name, c)]['mean']
                    for c in cases if (model_name, c) in results['LSB']]
        L_used = _used_states(lsb_list)
        n_cases = len(cases)
        case_colors = [COLOR_CONTINUOUS, COLOR_PARTIAL]
        fig, axes = plt.subplots(n_cases, 1,
                                  figsize=(8, 1.6 * L_used * n_cases + 1),
                                  squeeze=False)
        for i, case in enumerate(cases):
            ax = axes[i, 0]
            key = (model_name, case)
            if key not in results['LSB']:
                ax.set_visible(False)
                continue
            lsb = results['LSB'][key]['mean']    # (T+1, L)
            T = lsb.shape[0] - 1
            data = lsb[1:, :L_used].T            # (L_used, T)
            y_labels = [str(l) for l in range(1, L_used + 1)]
            sub_cmap = cmap if cmap is not None else _sequential_cmap(
                case_colors[i % len(case_colors)]
            )
            im = _draw_heatmap(ax, data, y_labels, (0.5, T + 0.5),
                               cmap=sub_cmap, xticks=xticks, xlim=xlim)
            ax.set_title(f'{case_labels[i]}', fontsize=11)
            fig.colorbar(im, ax=ax, label='Belief')
        fig.tight_layout()
        save(fig, out_dir / f'{filename_stem}_{short_name}_heatmap.png')
        return

    fig, ax = styled_figure()

    ls_counter = 0
    for case, label, color in zip(
        cases, case_labels, [COLOR_CONTINUOUS, COLOR_PARTIAL],
    ):
        key = (model_name, case)
        if key not in results['LSB']:
            continue
        lsb = results['LSB'][key]['mean']          # (T+1, L*ncop)
        ntrials = lsb.shape[0] - 1
        x = np.arange(1, ntrials + 2)              # 1..T+1 (incl. init)
        means = lsb.mean(axis=0)
        order = np.argsort(means)[::-1]

        for rank in range(2):
            if rank >= len(order):
                break
            state_idx = int(order[rank])
            ax.plot(x, lsb[:, state_idx], color=color,
                    linestyle=_ls(ls_counter), linewidth=LINE_W,
                    alpha=LINE_ALPHA,
                    label=f'{label}, state {rank + 1}')
            ls_counter += 1

    pretty_axes(ax, xlim=xlim, ylim=ylim, xticks=xticks)
    ax.set_ylabel('Beliefs', fontweight='bold', fontsize=FONT_SIZE_LABEL)
    ax.legend(loc='best', fontsize=10)

    save(fig, out_dir / f'{filename_stem}_{short_name}.png')


# =============================================================================
# Rescorla 2000 plots
# =============================================================================

def plot_delta_associative_strength(results, model_name, case,
                                     model_label, out_dir,
                                     cue_indices=(0, 1),
                                     cue_labels=('A', 'B'),
                                     ylim=(-0.5, 0.75), tol=0.02,
                                     case_labels=None):
    """Bar plot of Δ associative strength from trial 501 to trial 509.

    Translation of MATLAB RunRescorla2000.m's first plot loop.

    ⚠ DEVIATION FROM MATLAB: The source reads
        h1 = sum(MU{l,k}([501,509], 1), 2); h1 = h1(2) - h1(1);
        h2 = sum(MU{l,k}([501,509], 2), 2); h2 = h2(2) - h2(1);
    which accesses columns 1 and 2 of MU. For Rescorla2000 (D=1, A=1),
    MU has shape (512, 1) — accessing column 2 raises IndexError. Since
    the title says "Δ Associative strength" and the legend is {A, B},
    we interpret the intent as "change in associative strength (V) for
    cues A and B". This version uses V[:, cue_idx], which for LatentState
    (C=10) maps to columns 0=A and 1=B, and for non-LatentState (C=5)
    the same two indices.

    Parameters
    ----------
    cue_indices : tuple of int
        V-column indices for the two bars. Default (0, 1) = A, B.
    cue_labels : tuple of str
        Legend labels.
    ylim : tuple of float
        Y-axis limits. MATLAB used (-0.5, 0.75).
    tol : float
        Padding added to bar tops so bars are visible even when the
        Δ-value is near zero. Mirrors MATLAB's ``tol * sign(h)``.

    NOTE: bar patches keep their own alpha (0.75), not LINE_ALPHA — this
    is a summary bar chart, not a time series, so the bars should read
    as at-a-glance values. Legend entries are patch proxies.
    """
    key = (model_name, case)
    if key not in results['V']:
        return

    V = results['V'][key]['mean']
    idx_before = 500
    idx_after = 508

    deltas = []
    for ci in cue_indices:
        delta = float(V[idx_after, ci] - V[idx_before, ci])
        deltas.append(delta)

    fig, ax = styled_figure()
    bar_colors = [COLOR_PARTIAL, COLOR_CONTINUOUS]
    bar_centers = [1, 2]

    for center, delta, color, label in zip(
        bar_centers, deltas, bar_colors, cue_labels,
    ):
        height = delta + tol * np.sign(delta) if delta != 0 else tol
        # Draw from y=0 to height (positive or negative)
        bottom = min(0, height)
        top    = max(0, height)
        ax.add_patch(Rectangle(
            (center - 0.5, bottom), 1.0, top - bottom,
            facecolor=color, edgecolor='none', alpha=0.75,
        ))

    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_ylabel('Δ Associative strength',
                  fontweight='bold', fontsize=FONT_SIZE_LABEL)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend via patch proxies (patches aren't picked up automatically)
    handles = [
        Line2D([0], [0], color=color, lw=10, alpha=0.75, label=label)
        for color, label in zip(bar_colors, cue_labels)
    ]
    ax.legend(handles=handles, loc='best', frameon=False)

    case_str = f'_Case{case}' if case_labels is None else f'_{case_labels}'
    safe_case = case_str.replace(' ', '').replace('(', '').replace(')', '')
    save(fig, out_dir / f'AssociativeStrength_{model_label}{safe_case}.png')


def plot_beliefs_compound_window(results, model_name, case,
                                  model_label, out_dir,
                                  state_indices=(0, 1, 2),
                                  xlim=(490, 510),
                                  xticks=(495, 500, 505),
                                  shade_window=(501.5, 508.5),
                                  heatmap=False, cmap=None):
    """Plot 3 states' belief curves across the compound training window.

    Translation of MATLAB RunRescorla2000.m's second plot loop, with
    two corrections to the MATLAB source:

    1. MATLAB's ``patch([501.5, 501.5, 0, 0]', ...)`` has degenerate
       coordinates that don't form a sensible shaded region. We use
       ``shade_window=(501.5, 508.5)``, which matches the compound
       phase (trials 501–508) — the presumably-intended behavior.
    2. MATLAB's inner ``j``-loop re-plots each state's curve twice,
       which appears to be a copy-paste bug. This version plots each
       state once.

    Uses ``state_indices`` directly rather than "dominant" sorting,
    since the Rescorla plot is specifically showing the trajectory of
    the first three latent states.

    y-limits are fixed to (-0.1, 1.1) — a small margin around the
    probability range so beliefs that saturate at 0 or 1 remain
    visually distinguishable from the axis edges.

    When ``heatmap=True``, produces a heatmap showing *all* used states
    zoomed to ``xlim``. The colormap is a white→COLOR_NEUTRAL gradient,
    matching the line plot's single dark-neutral color for all states.
    Saved as ``Beliefs_{model_label}_Case{case}_heatmap.png``.
    """
    key = (model_name, case)
    if key not in results['LSB']:
        return

    lsb = results['LSB'][key]['mean']   # (T+1, L*ncop)
    n_states = lsb.shape[1]

    if heatmap:
        L_used = _used_states([lsb])
        T = lsb.shape[0] - 1
        data = lsb[1:, :L_used].T       # (L_used, T)
        y_labels = [str(l) for l in range(1, L_used + 1)]
        sub_cmap = cmap if cmap is not None else _sequential_cmap(COLOR_NEUTRAL)
        fig, ax = styled_figure()
        im = _draw_heatmap(ax, data, y_labels, (0.5, T + 0.5),
                           cmap=sub_cmap, xticks=xticks, xlim=xlim)
        # Shaded compound window
        ax.axvspan(shade_window[0], shade_window[1],
                   facecolor=COLOR_SHADE, alpha=0.15,
                   edgecolor='none', zorder=2)
        ax.set_title(f'{model_label} — Case {case} (all states)',
                     fontsize=13)
        fig.colorbar(im, ax=ax, label='Belief')
        save(fig, out_dir
             / f'Beliefs_{model_label}_Case{case}_heatmap.png')
        return

    available = [i for i in state_indices if i < n_states]
    if not available:
        return

    fig, ax = styled_figure()

    # Shaded compound window (trials 501-508)
    ax.axvspan(shade_window[0], shade_window[1],
               facecolor=COLOR_SHADE, alpha=0.15, edgecolor='none', zorder=0)

    # Belief curves — each state gets a distinct linestyle, all dark neutral
    for rank, state_idx in enumerate(available):
        x = np.arange(1, lsb.shape[0] + 1)   # 1..T+1 (incl. init)
        ax.plot(x, lsb[:, state_idx], color=COLOR_NEUTRAL,
                linestyle=_ls(rank), linewidth=LINE_W, alpha=LINE_ALPHA,
                label=f'State {state_idx + 1}')

    pretty_axes(ax, xlim=xlim, ylim=(-0.1, 1.1), xticks=xticks)
    ax.set_ylabel('Beliefs', fontweight='bold', fontsize=FONT_SIZE_LABEL)
    ax.legend(loc='best', fontsize=11)

    save(fig, out_dir / f'Beliefs_{model_label}_Case{case}.png')


# =============================================================================
# Memory-modification (Monfils-Schiller) plots
# =============================================================================

def plot_retrieval_time(results, model_name, cases, short_name, out_dir,
                        xticks=None, color=None):
    """Final test-trial associative strength vs. retrieval→extinction gap.

    For each case (which sets the delay between retrieval and extinction),
    plot the MU value on the final test trial. As the delay grows past the
    reconsolidation window, this value rises — that's the paper's core
    result.

    Parameters
    ----------
    cases : sequence of int
        Case values (the retrieval→extinction gap parameter), typically
        1..150 as in the MATLAB RunMonfilsSchiller.m loop.
    """
    if color is None:
        color = DEFAULT_ACCENT

    y_vals = []
    for case in cases:
        key = (model_name, case)
        if key not in results['MU']:
            y_vals.append(np.nan)
            continue
        mu = results['MU'][key]['mean']           # (T, D*A)
        y_vals.append(float(mu[-1, 0]))           # final trial

    x_vals = np.arange(len(cases))                # 0..len-1, matches MATLAB sp*(0:end-1)

    fig, ax = styled_figure()
    ax.plot(x_vals, y_vals, '-', color=color, linewidth=LINE_W,
            alpha=LINE_ALPHA, marker='.', markersize=8,
            label='Test-trial associative strength')
    ax.set_xlim(-0.5, len(cases) - 0.5)
    if xticks is None:
        xticks = [0, 50, 100, 150]
    ax.set_xticks(xticks)
    ax.set_ylabel('Associative strength, testing',
                  fontweight='bold', fontsize=FONT_SIZE_LABEL)
    ax.set_xlabel('Time delay after retrieval',
                  fontweight='bold', fontsize=FONT_SIZE_LABEL)
    ax.legend(loc='best', fontsize=11)
    save(fig, out_dir / f'RetrievalTime_{short_name}.png')


def plot_expectations_by_case(results, model_name, selected_cases,
                              case_colors, short_name, out_dir,
                              ntrials=24, acq_end=3, retr=4, test=24,
                              ylim=None, inset=False):
    """MU trajectory over trials for selected cases, one line per case.

    Reproduces MATLAB's per-trial segmentation:

        x=1..3   acquisition (line)
        x=4      retrieval (marker, no line)
        x=5..T-1 extinction (line)
        x=T      test (marker, no line)

    The gap between segments is intentional (visual emphasis on retrieval
    and test as "moments" rather than a continuous trajectory). Each case
    gets a distinct linestyle applied to both the line segments and the
    marker strokes, so cases are distinguishable without relying on color.

    Parameters
    ----------
    selected_cases : sequence of int
        Case values to plot (e.g. [100, 5, 1]).
    case_colors : dict mapping case → color.
    inset : bool
        If True, also produce a zoomed view of the test trial
        (xlim=[T-0.5, T+0.5]) saved as a separate PNG.
    """
    fig, ax = styled_figure()

    for i, case in enumerate(selected_cases):
        key = (model_name, case)
        if key not in results['MU']:
            continue
        mu = results['MU'][key]['mean'][:, 0]
        color = case_colors[case]
        ls = _ls(i)

        ax.plot(range(1, acq_end + 1), mu[:acq_end],
                '-', color=color, linewidth=LINE_W, alpha=LINE_ALPHA,
                label=f'Case {case}')
        ax.plot(retr, mu[retr - 1], 'x', color=color,
                markersize=12, markeredgewidth=3, alpha=LINE_ALPHA)
        ax.plot(range(retr + 1, ntrials), mu[retr:ntrials - 1],
                ls, color=color, linewidth=LINE_W, alpha=LINE_ALPHA)
        ax.plot(test, mu[ntrials - 1], 'x', color=color,
                markersize=12, markeredgewidth=3, alpha=LINE_ALPHA)

    ax.set_xlim(0.5, ntrials + 0.5)
    ax.set_xticks([retr, test])
    ax.set_xticklabels(['Retrieval', 'Test'])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel('Associative strength',
                  fontweight='bold', fontsize=FONT_SIZE_LABEL)
    ax.legend(loc='best', fontsize=11)
    save(fig, out_dir / f'Expectations_{short_name}.png')

    if inset:
        # Zoom to the test trial to show small differences.
        fig2, ax2 = styled_figure()
        for case in selected_cases:
            key = (model_name, case)
            if key not in results['MU']:
                continue
            mu_val = float(results['MU'][key]['mean'][-1, 0])
            ax2.plot(test, mu_val, 'x', color=case_colors[case],
                     markersize=14, markeredgewidth=3, alpha=LINE_ALPHA,
                     label=f'Case {case}')
        ax2.set_xlim(test - 0.5, test + 0.5)
        ax2.legend(loc='best', fontsize=11)
        save(fig2, out_dir / f'Inset_{short_name}.png')


def plot_beliefs_by_case(results, model_name, selected_cases,
                         case_colors, short_name, out_dir,
                         state_idx=0, ntrials=24, acq_end=3, retr=4,
                         test=24, ylim=(-0.1, 1.1),
                         heatmap=False, cmap=None):
    """Belief in a single state over trials, one line per case.

    Same segment structure and per-case linestyle progression as
    plot_expectations_by_case. State 1 (state_idx=0) is the acquisition
    state — the state whose belief dilution and reformation carries the
    memory-modification signal.

    y-limits default to (-0.1, 1.1) — a small margin around the
    probability range so beliefs that saturate at 0 or 1 remain visually
    distinguishable from the axis edges. Pass ``ylim`` to override.

    When ``heatmap=True``, produces a vertically stacked heatmap variant
    showing *all* used states per case. Each subplot's colormap is a
    white→case-color gradient matching the case's line color. Saved as
    ``Beliefs_{short_name}_heatmap.png``.
    """
    if heatmap:
        lsb_list = [results['LSB'][(model_name, c)]['mean']
                    for c in selected_cases if (model_name, c) in results['LSB']]
        L_used = _used_states(lsb_list)
        n_cases = len(selected_cases)
        fig, axes = plt.subplots(n_cases, 1,
                                  figsize=(8, 1.6 * L_used * n_cases + 1),
                                  squeeze=False)
        for i, case in enumerate(selected_cases):
            ax = axes[i, 0]
            key = (model_name, case)
            if key not in results['LSB']:
                ax.set_visible(False)
                continue
            lsb = results['LSB'][key]['mean']    # (T+1, L)
            T = lsb.shape[0] - 1
            data = lsb[1:, :L_used].T            # (L_used, T)
            y_labels = [str(l) for l in range(1, L_used + 1)]
            sub_cmap = cmap if cmap is not None else _sequential_cmap(
                case_colors[case]
            )
            im = _draw_heatmap(ax, data, y_labels, (0.5, T + 0.5),
                               cmap=sub_cmap, xticks=[retr, test])
            ax.set_xticklabels(['Retrieval', 'Test'])
            ax.set_title(f'Case {case}', fontsize=11)
            fig.colorbar(im, ax=ax, label='Belief')
        fig.tight_layout()
        save(fig, out_dir / f'Beliefs_{short_name}_heatmap.png')
        return

    fig, ax = styled_figure()

    for i, case in enumerate(selected_cases):
        key = (model_name, case)
        if key not in results['LSB']:
            continue
        lsb = results['LSB'][key]['mean']          # (T+1, L*ncop)
        color = case_colors[case]
        ls = _ls(i)

        # lsb[0] = pre-trial init; lsb[t] = post-trial-t.
        # Segments aligned to trials 1..T, so use lsb[1:] throughout.
        ax.plot(range(1, acq_end + 1), lsb[1:acq_end + 1, state_idx],
                '-', color=color, linewidth=LINE_W, alpha=LINE_ALPHA,
                label=f'Case {case}')
        ax.plot(retr, lsb[retr, state_idx], 'x', color=color,
                markersize=12, markeredgewidth=3, alpha=LINE_ALPHA)
        ax.plot(range(retr + 1, ntrials), lsb[retr + 1:ntrials, state_idx],
                ls, color=color, linewidth=LINE_W, alpha=LINE_ALPHA)
        ax.plot(test, lsb[ntrials, state_idx], 'x', color=color,
                markersize=12, markeredgewidth=3, alpha=LINE_ALPHA)

    ax.set_xlim(0.5, ntrials + 0.5)
    ax.set_xticks([retr, test])
    ax.set_xticklabels(['Retrieval', 'Test'])
    ax.set_ylim(*ylim)
    ax.set_ylabel(f'Belief in State {state_idx + 1}',
                  fontweight='bold', fontsize=FONT_SIZE_LABEL)
    ax.legend(loc='best', fontsize=11)
    save(fig, out_dir / f'Beliefs_{short_name}.png')
