"""examples/run_wilson_boumphrey_pearce.py — Wilson, Boumphrey & Pearce (1992).

Translation of MATLAB RunWilsonBoumphreyPearce.m. Produces:

  - Two schedule figures, one per group (C and E), showing which cues
    (food, light, tone) are present on each trial.
  - One associative-strength figure per model, plotting the light's V
    for both groups zoomed to the last 15 trials (the light-alone
    reinforced phase).
  - One beliefs figure per model, plotting the top-2 dominant states per
    group, also zoomed to the last 15 trials.

The zoom is the point: it lets you see whether Group E's light (which
was an unreliable predictor during Phase 2) recovers higher associative
strength faster than Group C's light when it becomes a reliable
predictor in Phase 3 — the Pearce-Hall prediction.
"""

import numpy as np

from olsl.common.paths import figures_dir
from olsl.common.patient_parameters import patient_parameters
from olsl.models.latent_state import LatentState
from olsl.simulation.run_example import run_example, save_results
from olsl.tasks import wilson_boumphrey_pearce as wbp
from olsl.plotting import timeseries


MODELS = [('LatentState', LatentState)]
MODEL_SHORT = {'LatentState': 'LS'}

CASES = ['C', 'E']
CASE_LABELS = ['Group C', 'Group E']


def main():
    out_dir = figures_dir('run_wilson_boumphrey_pearce')
    par = patient_parameters()

    # --- Schedule figures (don't need simulation results) ---
    for case, label in zip(CASES, CASE_LABELS):
        schedule_par = wbp.initialize({
            'case': case,
            'ls': {'ncop': 1},
            'modelname': 'RescorlaWagner',   # non-LatentState: C=2, outcomes are [US, light, tone]
        })
        timeseries.plot_cue_schedule(
            schedule_par,
            cue_names=['Food', 'Light', 'Tone'],
            title=f'{label} Schedule',
            out_dir=out_dir,
            filename=f'Schedule_{case}.png',
        )

    # --- Run simulation ---
    results = run_example(
        par=par,
        example=wbp,
        models=MODELS,
        nrep=1,
        cases=CASES,
        observer=None,
        progress=True,
    )
    save_results(results, 'WilsonBoumphreyPearce', out_dir=str(out_dir))

    # --- Zoomed plots: last 15 trials, light cue only ---
    xlim   = (0, 61)
    xticks = np.arange(0, 61, 20)

    for model_name, _ in MODELS:
        short = MODEL_SHORT[model_name]

        # Associative strength of the light (cue 0) for both groups
        timeseries.plot_cue_strength_zoomed(
            results, model_name, CASES, CASE_LABELS,
            cue_idx=0, short_name=short, out_dir=out_dir,
            xlim=xlim, ylim=(-0.55, 0.55), xticks=xticks,
            filename_stem='Prediction_Error',
        )

        # Top-2 beliefs per group
        timeseries.plot_beliefs_top2(
            results, model_name, CASES, CASE_LABELS,
            short_name=short, out_dir=out_dir,
            xlim=xlim, ylim=(0, 1), xticks=xticks,
            filename_stem='Beliefs',
        )
        timeseries.plot_beliefs_top2(
            results, model_name, CASES, CASE_LABELS,
            short_name=short, out_dir=out_dir,
            xlim=xlim, ylim=(0, 1), xticks=xticks,
            filename_stem='Beliefs', heatmap=True,
        )

    print(f'Figures written to {out_dir.resolve()}')


if __name__ == '__main__':
    main()
