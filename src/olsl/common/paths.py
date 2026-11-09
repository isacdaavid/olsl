"""common/paths.py — resolve project-relative paths independent of cwd.

Finding the repo root by walking up from __file__ avoids the classic
"figures/ is wherever I happened to run from" problem: run
``python examples/run_pree.py`` from the repo root, from a subdirectory,
or from ``~/Downloads`` — figures always land in the same place.
"""

from pathlib import Path


_MARKERS = ('pyproject.toml', 'setup.py', 'setup.cfg', '.git')


def repo_root() -> Path:
    """Return the repository root (the directory holding pyproject.toml).

    Walks up from this file; the first directory containing a marker
    wins. Falls back to ``<repo>/src/olsl/common/paths.py`` minus 3
    parents if no marker is found — that fallback matches the src
    layout but is fragile, so add a marker (pyproject.toml is standard)
    to be safe.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if any((parent / marker).exists() for marker in _MARKERS):
            return parent
    return here.parents[3]


def figures_dir(script_name: str) -> Path:
    """Return ``<repo>/figures/<script_name>``, creating it if needed.

    Per-script subdirectory prevents run_pree.py and run_renewal.py
    from clobbering each other's outputs, and gives you a place to
    drop a README or a per-script .gitignore if you want.
    """
    d = repo_root() / 'figures' / script_name
    d.mkdir(parents=True, exist_ok=True)
    return d
