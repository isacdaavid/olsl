"""Shared plotting style. Stand-in for MATLAB's ``Colors`` and ``PrettyFig``.

Uses the Selenized Light palette (https://github.com/jan-warchol/selenized)
for all colors, so every figure across the package shares the same visual
identity. Light palette chosen because figures are typically rendered on
white backgrounds.
"""

import matplotlib.pyplot as plt


# --- Selenized Light palette (hex from the canonical palette table) ---
SELENIZED = {
    # Backgrounds and neutrals
    'bg_0':      '#fbf3db',
    'bg_1':      '#ece3cc',
    'bg_2':      '#d5cdb6',
    'dim_0':     '#909995',
    'fg_0':      '#53676d',
    'fg_1':      '#3a4d53',
    # Regular accents
    'red':       '#d2212d',
    'green':     '#489100',
    'yellow':    '#ad8900',
    'blue':      '#0072d4',
    'magenta':   '#ca4898',
    'cyan':      '#009c8f',
    'orange':    '#c25d1e',
    'violet':    '#8762c6',
    # Bright accents
    'br_red':    '#cc1729',
    'br_green':  '#428b00',
    'br_yellow': '#a78300',
    'br_blue':   '#006dce',
    'br_magenta':'#c44392',
    'br_cyan':   '#00978a',
    'br_orange': '#bc5819',
    'br_violet': '#825dc0',
}


# --- Semantic aliases used across plot helpers -------------------------------

# Two-condition comparisons (PREE, Renewal, Spontaneous Recovery):
# COLOR_PARTIAL    = control / baseline condition      (dark neutral)
# COLOR_CONTINUOUS = manipulated / experimental        (accent blue)
COLOR_PARTIAL    = SELENIZED['fg_1']       # #3a4d53  dark slate
COLOR_CONTINUOUS = SELENIZED['blue']       # #0072d4  saturated blue

# Ordered palette for multi-cue and multi-state plots. Chosen for maximum
# mutual contrast on the Selenized light background.
CUE_COLORS = [
    SELENIZED['fg_1'],     # dark slate
    SELENIZED['blue'],     # #0072d4
    SELENIZED['orange'],   # #c25d1e
    SELENIZED['green'],    # #489100
    SELENIZED['violet'],   # #8762c6
    SELENIZED['magenta'],  # #ca4898
]

# Cue row colors for schedule grids (food, light, tone).
SCHEDULE_CUE_COLORS = [
    SELENIZED['red'],      # #d2212d  food
    SELENIZED['yellow'],   # #ad8900  light
    SELENIZED['green'],    # #489100  tone
]

# Case colors for memory-modification dose-response plots (delay = 100, 5, 1).
# Blue = long delay (outside reconsolidation window) → fear returns
# Orange = medium delay
# Green = short delay (inside window) → extinction sticks
MM_CASE_COLORS = {
    100: SELENIZED['blue'],
    5:   SELENIZED['orange'],
    1:   SELENIZED['green'],
}

# Default accent for single-series plots (retrieval-time curve, etc.).
DEFAULT_ACCENT = SELENIZED['blue']

# Dark neutral for single-series curves that should read as "plain".
COLOR_NEUTRAL = SELENIZED['fg_1']

# Light shading for annotated regions (e.g. compound-phase window).
COLOR_SHADE = SELENIZED['dim_0']


LINE_W = 5
FIG_SIZE = (6, 5)
LINE_ALPHA = 0.5
FONT_SIZE_LABEL = 18
LINESTYLES = ['-', '--', '-.', ':']


def _ls(i):
    """Return the i-th linestyle in the cycle."""
    return LINESTYLES[i % len(LINESTYLES)]


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
