"""Reusable plot builders.

Each function takes a ``results`` dict (from run_example) and writes one
figure per (case-pair) to ``out_dir``.

Belief-producing functions accept ``heatmap=True`` to render a heatmap
variant instead of their corresponding line plot. Posterior heatmaps use
latent state on the y-axis, trial on the x-axis, and color for belief.
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
    """Count the latent-state columns engaged across several belief arrays.

    Each array has shape ``(T+1, L)``. A state is considered used when its
    maximum belief exceeds ``threshold`` on at least one trial. The result
    is at least one so a uniform run still produces a visible heatmap row.
    """
    max_used = 1
    for lsb in lsb_list:
        if lsb is None:
            continue
        used = int(np.sum(np.max(lsb, axis=0) > threshold))
        max_used = max(max_used, used)
    return max_used


def _style_colorbar(cbar, label=None):
    """Apply shared label/tick styling and widen vertical colorbars."""
    if label is not None:
        cbar.set_label(label, fontweight='bold', fontsize=FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=14)

    # A vertical colorbar's box aspect is height / width. Reducing it by
    # four makes the bar approximately four times wider without changing
    # its allocated height.
    if cbar.orientation == 'vertical':
        aspect = cbar.ax.get_box_aspect()
        if aspect is not None:
            cbar.ax.set_box_aspect(aspect / 4)
    return cbar


def _add_case_legend(ax, label, color):
    """Label a case heatmap with a compact color swatch instead of a title."""
    handle = Line2D([], [], marker='s', linestyle='None',
                    markerfacecolor=color, markeredgecolor=color,
                    markersize=FONT_SIZE_LABEL * .5, label=label)
    ax.legend(handles=[handle], loc='upper right', fontsize=FONT_SIZE_LABEL * .7,
              frameon=False, handletextpad=0.4, borderaxespad=0.4)


def _format_line_plot(ax, ylabel, *, xlim=None, ylim=None, xticks=None,
                      xticklabels=None, legend_fontsize=None,
                      legend_loc='best'):
    """Apply the common time-series axis, label, and optional legend style."""
    pretty_axes(ax, xlim=xlim, ylim=ylim, xticks=xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=FONT_SIZE_LABEL)
    if legend_fontsize is not None:
        ax.legend(loc=legend_loc, fontsize=legend_fontsize)


def _draw_heatmap(ax, data_2d, y_labels, x_range, cmap,
                  vmin=0.0, vmax=1.0, xticks=None, xlim=None,
                  xlabel='Trial', ylabel='State',
                  show_xlabel=True, show_xticklabels=True):
    """Draw a heatmap where rows are categories and columns are trials."""
    n_rows = data_2d.shape[0]
    extent = (x_range[0], x_range[1], 0.5, n_rows + 0.5)
    im = ax.imshow(data_2d, aspect='auto', origin='lower', extent=extent,
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_yticks(np.arange(1, n_rows + 1))
    ax.set_yticklabels(y_labels)
    pretty_axes(ax, xlim=xlim, xticks=xticks)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=FONT_SIZE_LABEL)
    if show_xlabel:
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=FONT_SIZE_LABEL)
    if not show_xticklabels:
        ax.tick_params(labelbottom=False)
    return im


def _draw_state_heatmap(ax, lsb, n_states, cmap, *, xticks=None, xlim=None,
                        split_at=None, gap_width=20, xticklabels=None,
                        xlabel='Trial', ylabel='State',
                        show_xlabel=True, show_xticklabels=True):
    """Draw posterior beliefs for the first ``n_states`` latent states.

    ``lsb`` includes the pre-trial row at index zero; that row is omitted.
    When ``split_at`` is supplied, the trial axis is broken after that trial
    and the remaining trials are shifted right by ``gap_width``.
    """
    data = lsb[1:, :n_states].T
    y_labels = [str(i) for i in range(1, n_states + 1)]

    if split_at is None:
        im = _draw_heatmap(
            ax, data, y_labels, (0.5, data.shape[1] + 0.5), cmap,
            xticks=xticks, xlim=xlim, xlabel=xlabel, ylabel=ylabel,
            show_xlabel=show_xlabel,
            show_xticklabels=show_xticklabels,
        )
    else:
        left = data[:, :split_at]
        right = data[:, split_at:]
        y_extent = (0.5, n_states + 0.5)
        im = ax.imshow(
            left, aspect='auto', origin='lower',
            extent=(0.5, split_at + 0.5, *y_extent),
            cmap=cmap, vmin=0.0, vmax=1.0, interpolation='nearest',
        )
        if right.shape[1]:
            right_start = split_at + gap_width + 0.5
            ax.imshow(
                right, aspect='auto', origin='lower',
                extent=(right_start, right_start + right.shape[1], *y_extent),
                cmap=cmap, vmin=0.0, vmax=1.0, interpolation='nearest',
            )
        ax.set_yticks(np.arange(1, n_states + 1))
        ax.set_yticklabels(y_labels)
        pretty_axes(ax, xlim=xlim, xticks=xticks)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=FONT_SIZE_LABEL)
        if show_xlabel:
            ax.set_xlabel(xlabel, fontweight='bold', fontsize=FONT_SIZE_LABEL)
        if not show_xticklabels:
            ax.tick_params(labelbottom=False)
        if xticklabels is not None and show_xticklabels:
            ax.set_xticklabels(xticklabels)
    return im


def _plot_state_heatmap_stack(lsb_list, labels, colors, output_path, *,
                              cmap=None, xticks=None, xlim=None,
                              split_at=None, gap_width=20,
                              xticklabels=None, ylabel='State',
                              fig_width=8, row_height=0.8):
    """Render a stack of comparable latent-state posterior heatmaps.

    Missing cases can be represented by ``None`` in ``lsb_list``; their
    subplot is hidden while the remaining cases retain their original order.
    """
    valid_arrays = [lsb for lsb in lsb_list if lsb is not None]
    if not valid_arrays:
        return None

    n_states = _used_states(valid_arrays)
    n_panels = len(lsb_list)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(fig_width, row_height * n_states * n_panels + 1),
        squeeze=False,
    )
    visible_indices = [i for i, lsb in enumerate(lsb_list) if lsb is not None]
    bottom_index = visible_indices[-1]

    for i, (ax, lsb, label, color) in enumerate(
        zip(axes[:, 0], lsb_list, labels, colors)
    ):
        if lsb is None:
            ax.set_visible(False)
            continue
        panel_cmap = cmap if cmap is not None else _sequential_cmap(color)
        im = _draw_state_heatmap(
            ax, lsb, n_states, panel_cmap,
            xticks=xticks, xlim=xlim,
            split_at=split_at, gap_width=gap_width,
            xticklabels=xticklabels,
            ylabel=ylabel,
            show_xlabel=(i == bottom_index),
            show_xticklabels=(i == bottom_index),
        )
        _add_case_legend(ax, label, color)
        _style_colorbar(fig.colorbar(im, ax=ax), label='Belief')

    fig.tight_layout()
    save(fig, output_path)
    return fig


# Spontaneous-recovery plots use the same visual trial-axis break.
RECOVERY_BREAK_TRIAL = 40
RECOVERY_GAP_WIDTH = 20
RECOVERY_XTICKS = [0, 20, 40, 60, 70]
RECOVERY_XTICK_LABELS = ['0', '20', '40', '0', '10']


def _broken_trial_coordinates(ntrials, break_trial=RECOVERY_BREAK_TRIAL,
                              gap_width=RECOVERY_GAP_WIDTH):
    """Return x-coordinates for the two visible portions of a broken axis."""
    x_left = np.arange(1, break_trial + 1)
    x_right = gap_width + np.arange(break_trial + 1, ntrials + 1)
    return x_left, x_right


def _plot_broken_series(ax, values, *, color, linestyle='-', label=None,
                        break_trial=RECOVERY_BREAK_TRIAL,
                        gap_width=RECOVERY_GAP_WIDTH, mark_endpoints=True):
    """Plot trial-indexed values on the two sides of a visual time gap."""
    values = np.asarray(values)
    x_left, x_right = _broken_trial_coordinates(
        len(values), break_trial=break_trial, gap_width=gap_width,
    )
    ax.plot(x_left, values[:break_trial], color=color, linewidth=LINE_W,
            alpha=LINE_ALPHA, linestyle=linestyle, label=label)
    ax.plot(x_right, values[break_trial:], color=color, linewidth=LINE_W,
            alpha=LINE_ALPHA, linestyle=linestyle)
    if mark_endpoints and len(values) > break_trial:
        marker_style = dict(color=color, markersize=12, markeredgewidth=3,
                            alpha=LINE_ALPHA)
        ax.plot(break_trial, values[break_trial - 1], 'x', **marker_style)
        ax.plot(break_trial + gap_width + 1, values[break_trial], 'x',
                **marker_style)
    return x_left, x_right


def _format_broken_axis(ax, ylabel, *, xticks=None, xticklabels=None,
                        xlim=(0, 71), ylim=None, legend_fontsize=12):
    """Apply consistent labels and trial ticks to a broken-axis plot."""
    ax.set_xticks(RECOVERY_XTICKS if xticks is None else xticks)
    ax.set_xticklabels(RECOVERY_XTICK_LABELS if xticklabels is None
                       else xticklabels)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=FONT_SIZE_LABEL)
    if legend_fontsize is not None:
        ax.legend(loc='best', fontsize=legend_fontsize)


def _plot_retrieval_test_segments(ax, values, *, color, linestyle, label,
                                  ntrials, acq_end, retr, test):
    """Plot acquisition/extinction trajectories and retrieval/test markers."""
    values = np.asarray(values)
    ax.plot(range(1, acq_end + 1), values[:acq_end],
            linestyle, color=color, linewidth=LINE_W, alpha=LINE_ALPHA,
            label=label)
    ax.plot(retr, values[retr - 1], 'x', color=color,
            markersize=12, markeredgewidth=3, alpha=LINE_ALPHA)
    ax.plot(range(retr + 1, ntrials), values[retr:ntrials - 1],
            linestyle, color=color, linewidth=LINE_W, alpha=LINE_ALPHA)
    ax.plot(test, values[ntrials - 1], 'x', color=color,
            markersize=12, markeredgewidth=3, alpha=LINE_ALPHA)


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
    _format_line_plot(ax, 'Associative strength',
                      xlim=(0.5, ntrials + 0.5),
                      ylim=(ymin - margin, ymax + margin),
                      xticks=xticks,
                      legend_fontsize=FONT_SIZE_LABEL * .7)

    save(fig, out_dir / f'AssociativeStrength_{short_name}{pair_idx}.png')


def plot_beliefs(results, model_name, cases_pair, pair_idx,
                 short_name, out_dir, xticks, ylim=(-0.1, 1.1),
                 heatmap=False, cmap=None):
    """Plot the dominant-state belief trajectory for each case.

    The dominant state is selected independently for each case by its
    highest mean belief. This mirrors the MATLAB implementation, which
    sorts the LSB columns separately per condition; the selected latent
    state can therefore differ across cases.

    With ``heatmap=True``, render all used states as stacked posterior
    heatmaps instead of the dominant-state line plot.
    """
    if heatmap:
        lsb_list = [results['LSB'][(model_name, case)]['mean']
                    for case in cases_pair]
        _plot_state_heatmap_stack(
            lsb_list, [f'Case {case}' for case in cases_pair],
            [COLOR_PARTIAL, COLOR_CONTINUOUS],
            out_dir / f'Beliefs_{short_name}{pair_idx}_heatmap.png',
            cmap=cmap, xticks=xticks, ylabel='State', fig_width=8,
        )
        return

    fig, ax = styled_figure()
    for i, (case, color) in enumerate(
        zip(cases_pair, [COLOR_PARTIAL, COLOR_CONTINUOUS])
    ):
        lsb = results['LSB'][(model_name, case)]['mean']
        dominant = int(np.argmax(lsb.mean(axis=0)))
        ax.plot(lsb[:, dominant], linestyle=_ls(i), color=color,
                linewidth=LINE_W, alpha=LINE_ALPHA,
                label=f'Case {case} (state {dominant + 1})')

    ntrials = results['MU'][(model_name, cases_pair[0])]['mean'].shape[0]
    _format_line_plot(ax, 'Beliefs', xlim=(0.5, ntrials + 0.5),
                      ylim=ylim, xticks=xticks,
                      legend_fontsize=FONT_SIZE_LABEL * .7)
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

    When ``heatmap=True``, renders a (case × trial) strip heatmap instead of
    the line plot, saved as
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
        pretty_axes(ax, xlim=(0.5, T + 0.5), ylim=(0.5, n_cases + 0.5),
                    xticks=xticks)
        ax.set_ylabel('Case', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_xlabel('Trial', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        cbar = fig.colorbar(im, ax=ax, ticks=np.arange(0, L_max))
        cbar.ax.set_yticklabels([str(i + 1) for i in range(L_max)])
        _style_colorbar(cbar, label='Sampled state')
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
    _format_line_plot(ax, 'Sampled belief',
                      xlim=(0.5, ntrials + 0.5), ylim=ylim, xticks=xticks,
                      legend_fontsize=11)

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
    _format_line_plot(ax, ylabel, xlim=(0.5, ntrials + 0.5),
                      ylim=ylim, xticks=xticks, legend_fontsize=12)

    save(fig, out_dir / f'Expectations_Renewal_{short_name}.png')


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

    _format_line_plot(ax, 'Associative strength', xlim=xlim, ylim=ylim,
                      xticks=xticks,
                      legend_fontsize=FONT_SIZE_LABEL * .7)

    # out_dir is already per-task; filename omits the task name.
    save(fig, out_dir / f'AssociativeStrength_{short_name}.png')


# =============================================================================
# Spontaneous-recovery family (broken axis)
# =============================================================================

def plot_spontaneous_recovery_expectations(results, model_name, cases,
                                            case_labels, short_name, out_dir,
                                            xticks=None, ylim=None,
                                            case_colors=None):
    """Expected reward (MU) over trials, with a visual break for the delay."""
    if case_colors is None:
        case_colors = [COLOR_PARTIAL, COLOR_CONTINUOUS]

    fig, ax = styled_figure()
    for i, (case, label, color) in enumerate(
        zip(cases, case_labels, case_colors)
    ):
        key = (model_name, case)
        if key not in results['MU']:
            continue
        mu = results['MU'][key]['mean']
        _plot_broken_series(ax, mu[:, 0], color=color, linestyle=_ls(i),
                            label=label)

    _format_broken_axis(ax, 'Associative strength', xticks=xticks,
                        xlim=(0, 71),
                        ylim=(-0.55, 0.55) if ylim is None else ylim,
                        legend_fontsize=12)
    save(fig, out_dir / f'AS_SponRec_{short_name}.png')


def plot_spontaneous_recovery_beliefs(results, model_name, cases,
                                       case_labels, short_name, out_dir,
                                       xticks=None, case_colors=None,
                                       heatmap=False, cmap=None):
    """Dominant-state belief over trials, with a visual break for the delay.

    With ``heatmap=True``, render all used states in stacked heatmaps using
    the same broken trial axis.
    """
    if case_colors is None:
        case_colors = [COLOR_PARTIAL, COLOR_CONTINUOUS]

    if heatmap:
        lsb_list = [results['LSB'].get((model_name, case), {}).get('mean')
                    for case in cases]
        _plot_state_heatmap_stack(
            lsb_list, case_labels, case_colors,
            out_dir / f'Beliefs_SponRec_{short_name}_heatmap.png',
            cmap=cmap, xticks=RECOVERY_XTICKS, xlim=(0, 71),
            split_at=RECOVERY_BREAK_TRIAL,
            gap_width=RECOVERY_GAP_WIDTH,
            xticklabels=RECOVERY_XTICK_LABELS,
            ylabel='State', fig_width=10,
        )
        return

    fig, ax = styled_figure()
    for i, (case, label, color) in enumerate(
        zip(cases, case_labels, case_colors)
    ):
        key = (model_name, case)
        if key not in results['LSB']:
            continue
        lsb = results['LSB'][key]['mean']
        dominant = int(np.argmax(lsb.mean(axis=0)))
        _plot_broken_series(
            ax, lsb[1:, dominant], color=color, linestyle=_ls(i),
            label=f'{label} (state {dominant + 1})',
        )

    _format_broken_axis(ax, 'Beliefs', xticks=xticks, xlim=(0, 82),
                        ylim=(-0.1, 1.1), legend_fontsize=12)
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
    """Belief in one state through the delay, including analytic decay.

    ``gamma`` is the LatentState transition probability. The silent interval
    contains ``case - 1`` transitions; its posterior trajectory is computed
    analytically and drawn across the visual gap. ``state_idx`` selects the
    state to display, and ``case_linestyles`` can override the linestyle
    progression. With ``heatmap=True``, render all used states in stacked
    broken-axis heatmaps instead of the line plot.
    """
    if case_colors is None:
        case_colors = [COLOR_PARTIAL, COLOR_CONTINUOUS]
    if case_linestyles is None:
        case_linestyles = [_ls(i) for i in range(len(cases))]

    if heatmap:
        lsb_list = [results['LSB'].get((model_name, case), {}).get('mean')
                    for case in cases]
        _plot_state_heatmap_stack(
            lsb_list, case_labels, case_colors,
            out_dir / f'BeliefDilution_SponRec_{short_name}_heatmap.png',
            cmap=cmap, xticks=RECOVERY_XTICKS, xlim=(0, 71),
            split_at=RECOVERY_BREAK_TRIAL,
            gap_width=RECOVERY_GAP_WIDTH,
            xticklabels=RECOVERY_XTICK_LABELS,
            ylabel='State', fig_width=10,
        )
        return

    fig, ax = styled_figure()
    break_trial = RECOVERY_BREAK_TRIAL
    gap_width = RECOVERY_GAP_WIDTH

    for case, label, color, linestyle in zip(
        cases, case_labels, case_colors, case_linestyles
    ):
        key = (model_name, case)
        if key not in results['LSB']:
            continue
        lsb = results['LSB'][key]['mean']
        n_silent = case - 1
        lsb_end_ext = lsb[break_trial, :]
        n_active_case = max(1, int(np.sum(lsb_end_ext > 1e-10)))

        values = lsb[1:, state_idx]
        _plot_broken_series(
            ax, values, color=color, linestyle=linestyle, label=label,
            mark_endpoints=False,
        )

        # Analytic posterior evolution during the silent interval.
        if n_silent > 0:
            tau = np.linspace(0, n_silent, 200)
            prob_stay_tau = (1 - gamma) ** tau
            target = 1.0 / n_active_case
            belief_tau = (lsb_end_ext[state_idx] * prob_stay_tau
                          + (1 - prob_stay_tau) * target)
            x_gap = break_trial + gap_width * tau / n_silent
            ax.plot(x_gap, belief_tau, color=color, linestyle=linestyle,
                    linewidth=LINE_W, alpha=LINE_ALPHA)
            prior_41 = _context_shift_belief(
                lsb_end_ext, gamma, n_silent, n_active_case,
            )[state_idx]
        else:
            prior_41 = lsb_end_ext[state_idx]

        marker_style = dict(color=color, markersize=12,
                            markeredgewidth=3, alpha=LINE_ALPHA)
        ax.plot(break_trial, lsb_end_ext[state_idx], 'x', **marker_style)
        ax.plot(break_trial + gap_width + 1, prior_41, 'x', **marker_style)

    _format_broken_axis(
        ax, f'Belief in State {state_idx + 1}', xticks=xticks,
        xlim=(0, 71), ylim=(-0.1, 1.1), legend_fontsize=12,
    )
    ax.set_title('Belief dilution during silent delay',
                 fontsize=FONT_SIZE_LABEL)
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
    ax.set_title(title, fontsize=FONT_SIZE_LABEL)
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

    _format_line_plot(ax, 'Associative strength', xlim=xlim, ylim=ylim,
                      xticks=xticks,
                      legend_fontsize=FONT_SIZE_LABEL * .7)

    save(fig, out_dir / f'{filename_stem}_{short_name}.png')


def plot_beliefs_top2(results, model_name, cases, case_labels,
                      short_name, out_dir, xlim, xticks, filename_stem,
                      ylim=(-0.1, 1.1), heatmap=False, cmap=None):
    """Plot the two most active latent-state beliefs for each case.

    State rankings are computed independently per case from mean belief.
    Each case has its own color, while a global linestyle progression
    distinguishes the state-rank trajectories. With ``heatmap=True``, show
    all used states in stacked posterior heatmaps instead.
    """
    case_colors = [COLOR_CONTINUOUS, COLOR_PARTIAL]
    if heatmap:
        lsb_list = [results['LSB'].get((model_name, case), {}).get('mean')
                    for case in cases]
        _plot_state_heatmap_stack(
            lsb_list, case_labels, case_colors,
            out_dir / f'{filename_stem}_{short_name}_heatmap.png',
            cmap=cmap, xticks=xticks, xlim=xlim,
            ylabel='State', fig_width=8,
        )
        return

    fig, ax = styled_figure()
    ls_counter = 0
    for case, label, color in zip(cases, case_labels, case_colors):
        key = (model_name, case)
        if key not in results['LSB']:
            continue
        lsb = results['LSB'][key]['mean']
        order = np.argsort(lsb.mean(axis=0))[::-1]
        x = np.arange(1, lsb.shape[0] + 1)
        for rank, state_idx in enumerate(order[:2]):
            ax.plot(x, lsb[:, state_idx], color=color,
                    linestyle=_ls(ls_counter), linewidth=LINE_W,
                    alpha=LINE_ALPHA,
                    label=f'{label}, state {rank + 1}')
            ls_counter += 1

    _format_line_plot(ax, 'Beliefs', xlim=xlim, ylim=ylim, xticks=xticks,
                      legend_fontsize=10)
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
    """Plot selected latent-state beliefs across the compound training window.

    The shaded region marks compound training. The heatmap variant shows all
    used states; the line variant shows the requested state indices.

    This translates the second plot loop in ``RunRescorla2000.m`` with two
    corrections: the original MATLAB patch has degenerate coordinates, so
    ``shade_window`` marks trials 501–508; and a duplicated inner loop in the
    MATLAB source is removed so each requested state is plotted once. The
    explicitly requested state indices are used rather than sorting by
    dominance because this plot tracks particular states.
    """
    key = (model_name, case)
    if key not in results['LSB']:
        return

    lsb = results['LSB'][key]['mean']
    n_states = lsb.shape[1]
    output_stem = f'Beliefs_{model_label}_Case{case}'

    if heatmap:
        n_used = _used_states([lsb])
        fig, ax = plt.subplots(figsize=(6, 2.5))
        panel_cmap = cmap if cmap is not None else _sequential_cmap(COLOR_NEUTRAL)
        im = _draw_state_heatmap(
            ax, lsb, n_used, panel_cmap, xticks=xticks, xlim=xlim,
        )
        ax.axvspan(shade_window[0], shade_window[1],
                   facecolor=COLOR_SHADE, alpha=0.15,
                   edgecolor='none', zorder=2)
        _add_case_legend(ax, f'{model_label} — Case {case}', COLOR_NEUTRAL)
        _style_colorbar(fig.colorbar(im, ax=ax), label='Belief')
        save(fig, out_dir / f'{output_stem}_heatmap.png')
        return

    available = [i for i in state_indices if i < n_states]
    if not available:
        return

    fig, ax = styled_figure()
    ax.axvspan(shade_window[0], shade_window[1],
               facecolor=COLOR_SHADE, alpha=0.15,
               edgecolor='none', zorder=0)
    x = np.arange(1, lsb.shape[0] + 1)
    for rank, state_idx in enumerate(available):
        ax.plot(x, lsb[:, state_idx], color=COLOR_NEUTRAL,
                linestyle=_ls(rank), linewidth=LINE_W, alpha=LINE_ALPHA,
                label=f'State {state_idx + 1}')

    _format_line_plot(ax, 'Beliefs', xlim=xlim, ylim=(-0.1, 1.1),
                      xticks=xticks,
                      legend_fontsize=FONT_SIZE_LABEL * .7)
    save(fig, out_dir / f'{output_stem}.png')


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
    ax.legend(loc='best', fontsize=FONT_SIZE_LABEL * .7)
    save(fig, out_dir / f'RetrievalTime_{short_name}.png')


def plot_expectations_by_case(results, model_name, selected_cases,
                              case_colors, short_name, out_dir,
                              ntrials=24, acq_end=3, retr=4, test=24,
                              ylim=None, inset=False):
    """MU trajectory with retrieval and test shown as separate markers.

    Acquisition and extinction are drawn as separate line segments;
    retrieval and test are isolated ``x`` markers. Each case receives a
    distinct linestyle throughout both line segments. ``inset=True`` also
    saves a separate zoom of the test-trial values.
    """
    fig, ax = styled_figure()
    for i, case in enumerate(selected_cases):
        key = (model_name, case)
        if key not in results['MU']:
            continue
        values = results['MU'][key]['mean'][:, 0]
        _plot_retrieval_test_segments(
            ax, values, color=case_colors[case], linestyle=_ls(i),
            label=f'Case {case}', ntrials=ntrials,
            acq_end=acq_end, retr=retr, test=test,
        )

    ax.set_xlim(0.5, ntrials + 0.5)
    ax.set_xticks([retr, test])
    ax.set_xticklabels(['Retrieval', 'Test'])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel('Associative strength',
                  fontweight='bold', fontsize=FONT_SIZE_LABEL)
    ax.legend(loc='best', fontsize=FONT_SIZE_LABEL * .7)
    save(fig, out_dir / f'Expectations_{short_name}.png')

    if inset:
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
    """Plot one state's belief through acquisition, retrieval, and test.

    Belief arrays contain a pre-trial initial row, so ``lsb[1:]`` is used
    to align the values with trial numbers. Acquisition and extinction are
    separate line segments, with retrieval and test marked independently.
    State 1 is the default because it represents the acquisition state in
    the memory-modification task. With ``heatmap=True``, show all used
    latent states for each case instead of the selected-state trajectory.
    """
    if heatmap:
        lsb_list = [results['LSB'].get((model_name, case), {}).get('mean')
                    for case in selected_cases]
        _plot_state_heatmap_stack(
            lsb_list, [f'Case {case}' for case in selected_cases],
            [case_colors[case] for case in selected_cases],
            out_dir / f'Beliefs_{short_name}_heatmap.png',
            cmap=cmap, xticks=[retr, test],
            xticklabels=['Retrieval', 'Test'],
            ylabel='State', fig_width=8,
        )
        return

    fig, ax = styled_figure()
    for i, case in enumerate(selected_cases):
        key = (model_name, case)
        if key not in results['LSB']:
            continue
        lsb = results['LSB'][key]['mean']
        # Convert pre-trial + post-trial beliefs to trial-indexed values.
        _plot_retrieval_test_segments(
            ax, lsb[1:, state_idx], color=case_colors[case],
            linestyle=_ls(i), label=f'Case {case}', ntrials=ntrials,
            acq_end=acq_end, retr=retr, test=test,
        )

    ax.set_xlim(0.5, ntrials + 0.5)
    ax.set_xticks([retr, test])
    ax.set_xticklabels(['Retrieval', 'Test'])
    ax.set_ylim(*ylim)
    ax.set_ylabel(f'Belief in State {state_idx + 1}',
                  fontweight='bold', fontsize=FONT_SIZE_LABEL)
    ax.legend(loc='best', fontsize=FONT_SIZE_LABEL * .7)
    save(fig, out_dir / f'Beliefs_{short_name}.png')
