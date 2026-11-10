"""examples/run_rescorla_2000.py — Rescorla (2000) Experiments 1A and 1B.

Translation of MATLAB RunRescorla2000.m.

Compares the effect of reinforced vs. unreinforced compound training:

    case=1 : AB+ (Experiment 1A)  — B becomes excitatory
    case=2 : AB- (Experiment 1B)  — B becomes inhibitory

Two figure types:

  1. Δ Associative strength bar plot, one per case, showing the change
     in a cue's associative strength between the end of single-cue
     training (trial 500) and the end of compound training (trial 508).
     Two bars per figure: one for cue A, one for cue B.
     ⚠ See the docstring of plot_delta_associative_strength for the
     deviation from MATLAB's MU-based version.

  2. Belief trajectories over the compound window (trials 490-510),
     one figure per case. Shows how latent-state beliefs evolve as
     the compound rule is learned.

Only LatentState is ported so far; add other models to MODELS as they
land.
"""

from olsl.common.paths import figures_dir
from olsl.common.patient_parameters import patient_parameters
from olsl.models.latent_state import LatentState
from olsl.simulation.run_example import run_example, save_results
from olsl.tasks import rescorla_2000 as r2000
from olsl.plotting import timeseries


MODELS = [('LatentState', LatentState)]
MODEL_SHORT = {'LatentState': 'LS'}

CASES = [1, 2]


def main():
    out_dir = figures_dir('run_rescorla_2000')
    par = patient_parameters()

    results = run_example(
        par=par,
        example=r2000,
        models=MODELS,
        nrep=1,
        cases=CASES,
        observer=None,
        progress=True,
    )
    save_results(results, 'Rescorla2000', out_dir=str(out_dir))

    for model_name, _ in MODELS:
        short = MODEL_SHORT[model_name]

        # Δ Associative strength — one figure per case
        for case in CASES:
            timeseries.plot_delta_associative_strength(
                results, model_name, case,
                model_label=short, out_dir=out_dir,
            )

        # Beliefs over the compound window — one figure per case
        for case in CASES:
            timeseries.plot_beliefs_compound_window(
                results, model_name, case,
                model_label=short, out_dir=out_dir,
            )
            timeseries.plot_beliefs_compound_window(
                results, model_name, case,
                model_label=short, out_dir=out_dir,
                heatmap=True,
            )

    print(f'Figures written to {out_dir.resolve()}')


if __name__ == '__main__':
    main()
