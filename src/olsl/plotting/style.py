"""Shared plotting style. Stand-in for MATLAB's ``Colors`` and ``PrettyFig``."""

import matplotlib.pyplot as plt


# --- Palette (matches the MATLAB Colors script) ---
COLOR_PARTIAL    = (0.15, 0.15, 0.15)    # dark grey
COLOR_CONTINUOUS = (0.45, 0.60, 0.80)    # light blue

LINE_W = 4
FIG_SIZE = (6, 5)
FONT_SIZE_LABEL = 18


def pretty_axes(ax, xlim, ylim, xticks):
    """Minimal stand-in for MATLAB's PrettyFig.

    NB: the MATLAB version's arg order is (title, ylim, xlim, xticks) —
    the parameter names ``xLim``/``ylim`` in RunPREE.m are misleading.
    Here we use the conventional (xlim, ylim, xticks) order.
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(xticks)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=14)


def styled_figure():
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    return fig, ax


def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
