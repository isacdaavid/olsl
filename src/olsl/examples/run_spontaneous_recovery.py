"""examples/run_spontaneous_recovery.py — Spontaneous Recovery.

Translation of MATLAB RunSpontaneousRecovery.m.

Runs two cases:
    case=1   : no silent gap (baseline; extinction does not decay)
    case=200 : 199-trial silent gap (beliefs decay → recovery)

Both are plotted on the same broken x-axis: acquisition/extinction on
the left, renewal phase on the right (shifted right by a visual gap to
represent the silent delay). The only difference between cases is how
much the dominant-state belief drops during the gap — which is what
drives the size of the recovery effect.
"""

from olsl.common.patient_parameters import patient_parameters
from olsl.common.paths import figures_dir
from olsl.models.latent_state import LatentState
from olsl.simulation.run_example import run_example, save_results
from olsl.tasks import spontaneous_recovery
from olsl.plotting import timeseries


MODELS = [('LatentState', LatentState)]
MODEL_SHORT = {'LatentState': 'LS'}

CASES = [1, 200]
CASE_LABELS = ('No context change', 'Context change')


def main():
    out_dir = figures_dir('run_spontaneous_recovery')
    par = patient_parameters()

    results = run_example(
        par=par,
        example=spontaneous_recovery,
        models=MODELS,
        nrep=1,
        cases=CASES,
        observer=None,
        progress=True,
    )
    save_results(results, 'SpontaneousRecovery', out_dir=str(out_dir))

    for model_name, _ in MODELS:
        short = MODEL_SHORT[model_name]
        timeseries.plot_spontaneous_recovery_expectations(
            results, model_name, CASES, CASE_LABELS, short, out_dir,
            ylim=(-0.55, 0.55),
        )
        timeseries.plot_spontaneous_recovery_beliefs(
            results, model_name, CASES, CASE_LABELS, short, out_dir,
        )
        timeseries.plot_spontaneous_recovery_beliefs(
            results, model_name, CASES, CASE_LABELS, short, out_dir,
            heatmap=True,
        )
        # NEW: belief in the extinction state through the delay
        timeseries.plot_spontaneous_recovery_dilution(
            results, model_name, CASES, CASE_LABELS,
            gamma=par['ls']['gamma'], short_name=short, out_dir=out_dir,
        )
        timeseries.plot_spontaneous_recovery_dilution(
            results, model_name, CASES, CASE_LABELS,
            gamma=par['ls']['gamma'], short_name=short, out_dir=out_dir,
            heatmap=True,
        )

    print(f'Figures written to {out_dir.resolve()}')


if __name__ == '__main__':
    main()

