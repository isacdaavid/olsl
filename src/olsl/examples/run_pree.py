"""examples/run_pree.py — run and visualize PREE.

Translation of MATLAB RunPREE.m. This is a *driver*, not library code:
it configures, runs, and plots. If you later want a single script that
handles PREE, Renewal, and blocking with the same plotting code, the
reusable pieces already live in ``olsl.plotting``.
"""

from pathlib import Path

from olsl.models.latent_state import LatentState
from olsl.simulation.run_example import run_example, save_results
from olsl.tasks import pree
from olsl.plotting import timeseries


# --- Configuration --------------------------------------------------------

MODELS = [('LatentState', LatentState)]
MODEL_SHORT = {'LatentState': 'LS'}

# Pairs of cases to compare (partial vs. continuous)
CASES_PAIRS = [
    ([1, 2], [10, 20, 30]),              # 30-trial versions
    ([3, 4], [10, 20, 30, 40, 50]),      # 50-trial versions
]

# Models whose LSB is worth plotting (subset, matches MATLAB's
# intersect([1,5,6,7,8], ...))
LATENT_MODELS = {'LatentState', 'InfiniteMixture',
                 'Gershman2017', 'Gershman2017_alpha1', 'Redish2007'}

# Models that produce a sampled x_t sequence
HARD_MODELS = {'LatentStateHard'}

OUT_DIR = Path('figures')


# --- Agent parameters (translation of PatientParameters) ------------------

def patient_parameters():
    return {
        'tau': 10.0,
        'L': 1, 'D': 1, 'C': 1, 'A': 1, 'center': 0.5,
        'ls': {
            'ncop':   10,
            'sigma0': 0.5,
            'gamma':  0.05,
            'alpha0': 0.05,
            'alpha1': 0.05,
            'alpha2': 0.05,
            'chi':    5,
            'delta':  0.6,
            'eta':    0.2,
        },
        'kf': {
            'L': 2,
            'Q': 0.01,
            'R': 0.25,
            'initP': 1.0,
            'gamma': 0.05,
        },
    }


# --- Main -----------------------------------------------------------------

def main():
    OUT_DIR.mkdir(exist_ok=True)

    par = patient_parameters()
    all_cases = [c for pair, _ in CASES_PAIRS for c in pair]

    results = run_example(
        par=par,
        example=pree,
        models=MODELS,
        nrep=1,
        cases=all_cases,
        observer=None,
        progress=True,
    )
    save_results(results, 'PREE', out_dir=str(OUT_DIR))

    for model_name, _ in MODELS:
        short = MODEL_SHORT[model_name]

        for pair_idx, (cases_pair, xticks) in enumerate(CASES_PAIRS, start=1):
            timeseries.plot_associative_strength(
                results, model_name, cases_pair, pair_idx,
                short, OUT_DIR, xticks,
            )
            if model_name in LATENT_MODELS:
                timeseries.plot_beliefs(
                    results, model_name, cases_pair, pair_idx,
                    short, OUT_DIR, xticks,
                )
            if model_name in HARD_MODELS:
                timeseries.plot_sampled_belief(
                    results, model_name, cases_pair, pair_idx,
                    short, OUT_DIR, xticks,
                )

    print(f'Figures written to {OUT_DIR.resolve()}')


if __name__ == '__main__':
    main()
