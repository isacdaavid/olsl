"""examples/run_renewal.py — run and visualize Renewal.

Translation of MATLAB RunRenewal.m. Like run_pree.py, this is a
*driver*: it configures, runs, and plots. Reusable plotting lives in
olsl.plotting.timeseries.

The MATLAB original targeted ``LatentStateHard`` (a hard-sampling variant
we haven't ported yet). This script uses ``LatentState`` (soft) instead.
The belief / MMKF plots work identically; the "Sampled belief" figure is
skipped because LatentState doesn't produce an x_t sequence. To exercise
that code path, swap in a hard-sampling model.
"""

from pathlib import Path

from olsl.common.paths import figures_dir
from olsl.common.patient_parameters import patient_parameters
from olsl.models.latent_state import LatentState
from olsl.simulation.run_example import run_example, save_results
from olsl.tasks import renewal
from olsl.plotting import timeseries


MODELS = [('LatentState', LatentState)]
MODEL_SHORT = {'LatentState': 'LS'}

# Models whose latent-state beliefs are worth plotting.
# Mirrors MATLAB's ``intersect([1,5,6,7,8], ...)`` gate.
LATENT_MODELS = {'LatentState', 'InfiniteMixture',
                 'Gershman2017', 'Gershman2017_alpha1', 'Redish2007'}

# Models that produce a sampled x_t sequence (hard-sampling variants).
HARD_MODELS = {'LatentStateHard'}

CASES = [1, 2]
OUT_DIR = figures_dir('run_renewal')


def main():
    OUT_DIR.mkdir(exist_ok=True)

    par = patient_parameters()

    results = run_example(
        par=par,
        example=renewal,
        models=MODELS,
        nrep=1,
        cases=CASES,
        observer=None,
        progress=True,
    )
    save_results(results, 'Renewal', out_dir=str(OUT_DIR))

    xticks = [0, 10, 20, 30, 40, 50]

    for model_name, _ in MODELS:
        short = MODEL_SHORT[model_name]

        # --- Expectations (MU) ---
        # Y-limits depend on whether the model reduces context to cues:
        #   - LatentState: rewards ∈ {0, 1}, centered → μ ∈ [-0.5, 0.5]
        #   - others:      additional context cues push μ up to ~1.0
        ylim = (-0.55, 0.55) if model_name in LATENT_MODELS else (-0.55, 1.05)
        timeseries.plot_expectations(
            results, model_name, CASES, short, OUT_DIR,
            xticks=xticks, ylim=ylim,
            ylabel='Expected reward',
            case_labels=('No context change', 'Context change'),
        )

        if model_name in LATENT_MODELS:
            # --- Beliefs (dominant state per case) ---
            timeseries.plot_beliefs(
                results, model_name, CASES, pair_idx=1,
                short_name=short, out_dir=OUT_DIR, xticks=xticks,
            )
            timeseries.plot_beliefs(
                results, model_name, CASES, pair_idx=1,
                short_name=short, out_dir=OUT_DIR, xticks=xticks,
                heatmap=True,
            )

            # --- Sampled x_t (only for hard-sampling models) ---
            if model_name in HARD_MODELS:
                timeseries.plot_sampled_belief(
                    results, model_name, CASES, pair_idx=1,
                    short_name=short, out_dir=OUT_DIR, xticks=xticks,
                )

            # --- MMKF tracking (one 3-panel figure per case) ---
            for case in CASES:
                timeseries.plot_mmkf_tracking(
                    results, model_name, case, short, OUT_DIR,
                    xticks=xticks,
                    par_kf_L=par['kf']['L'],
                )

    print(f'Figures written to {OUT_DIR.resolve()}')


if __name__ == '__main__':
    main()
