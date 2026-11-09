"""simulation/blocking_family.py — shared run+plot for blocking-family tasks.

The four blocking-family experiments (Blocking, Overexpectation,
Conditioned Inhibition, Backwards Blocking) share the same simulation
loop and plotting logic; they differ only in which task module to run
and which V columns to plot. ``TASK_CONFIGS`` holds the per-task
config and ``run_and_plot`` does the work.

The point of a shared helper — rather than duplicating ~40 lines across
four driver files — is that adding a fifth blocking-family task is one
entry in ``TASK_CONFIGS`` plus a 15-line driver. It also means the four
drivers stay trivially readable.
"""

from pathlib import Path

from ..common.paths import figures_dir   # re-exported for drivers
from ..models.latent_state import LatentState
from ..plotting import timeseries
from ..tasks import (
    blocking, backwards_blocking, overexpectation, conditioned_inhibition,
)
from .run_example import run_example, save_results


MODELS = [('LatentState', LatentState)]
MODEL_SHORT = {'LatentState': 'LS'}


# Per-task configuration. ``cue_indices`` are V-column indices; for
# LatentState (C=6) they are A=0, B=1, C=2, D=3, AB_int=4, CD_int=5.
# For non-LatentState (C=4) the first four columns are A/B/C/D — so
# indices 0-2 work for either model.
TASK_CONFIGS = {
    'Blocking': dict(
        module=blocking,
        cue_indices=[0, 1, 2],
        cue_labels=['A', 'B', 'C/D'],
        xticks=[10, 20, 30, 40],
        xlim=(-0.55, 1.15),
    ),
    'Overexpectation': dict(
        module=overexpectation,
        cue_indices=[0, 1],
        cue_labels=['A', 'B'],
        xticks=[30, 60, 90, 120],
        xlim=(-0.55, 1.5),
    ),
    'ConditionedInhibition': dict(
        module=conditioned_inhibition,
        cue_indices=[0, 1],
        cue_labels=['A', 'B'],
        xticks=[50, 100, 150, 200],
        xlim=(-0.55, 1.15),
    ),
    'BackwardsBlocking': dict(
        module=backwards_blocking,
        cue_indices=[0, 1],
        cue_labels=['A', 'B'],
        xticks=[10, 20, 30, 40],
        xlim=(-0.55, 1.15),
    ),
}


def run_and_plot(task_name, par, out_dir):
    """Run one blocking-family task and write its figure.

    Parameters
    ----------
    task_name : str
        One of the keys in ``TASK_CONFIGS``.
    par : dict
        Patient parameters (from ``patient_parameters()``).
    out_dir : Path
        Per-task figure directory. The caller is responsible for making
        the subdirectory distinct per task (e.g. ``figures/run_blocking``).
    """
    if task_name not in TASK_CONFIGS:
        raise ValueError(
            f"Unknown blocking-family task {task_name!r}. "
            f"Choose from: {', '.join(TASK_CONFIGS)}"
        )
    cfg = TASK_CONFIGS[task_name]

    # --- Run ---
    print(f"Running ... {task_name}")
    results = run_example(
        par=par,
        example=cfg['module'],
        models=MODELS,
        nrep=1,
        cases=[1],
        observer=None,
        progress=False,
    )

    # run_example keys by (model_name, case); each task is a single
    # experiment, so relabel (model_name, 1) → (model_name, task_name)
    # so the figure helper's results[key] lookup works naturally.
    relabeled = {}
    for top_key in ('MU', 'LSB', 'V', 'XT',
                    'MMKF_PRIOR', 'MMKF_LIKE', 'MMKF_POST', 'MMKF_XT',
                    'par_by_key'):
        if top_key not in results:
            continue
        relabeled[top_key] = {}
        for (model_name, _), val in results[top_key].items():
            relabeled[top_key][(model_name, task_name)] = val

    save_results(relabeled, task_name, out_dir=str(out_dir))

    # --- Plot ---
    for model_name, _ in MODELS:
        short = MODEL_SHORT[model_name]
        timeseries.plot_associative_strength_per_cue(
            results=relabeled,
            task_name=task_name,
            model_name=model_name,
            cue_indices=cfg['cue_indices'],
            cue_labels=cfg['cue_labels'],
            xticks=cfg['xticks'],
            short_name=short,
            out_dir=out_dir,
            xlim=cfg['xlim'],
        )

    print(f'Figure written to {out_dir.resolve()}')
