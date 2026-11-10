"""examples/run_memory_modification.py — Monfils/Schiller memory reconsolidation.

Translation of MATLAB RunMonfilsSchiller.m.

Runs 150 cases of the memory-modification task. Each case differs only
in the retrieval→extinction delay (par['case'] = 1, 2, ..., 150). The
question: does the size of that delay affect whether extinction
reconsolidates the acquisition memory (short delay = inside window =
extinction sticks) or forms a new extinction memory (long delay =
outside window = fear returns at test)?

Three figures per model:
  1. RetrievalTime_LS.png — final test-trial associative strength vs.
                             delay; the dose-response curve.
  2. Expectations_LS.png  — MU trajectory for three illustrative cases
                             (delay=100, delay=5, delay=1) overlaid.
  3. Beliefs_LS.png       — belief in state 1 (acquisition state) for
                             the same three cases.
"""

from olsl.common.paths import figures_dir
from olsl.common.patient_parameters import patient_parameters
from olsl.models.latent_state import LatentState
from olsl.simulation.run_example import run_example, save_results
from olsl.tasks import memory_modification as mm
from olsl.plotting import timeseries


MODELS = [('LatentState', LatentState)]
MODEL_SHORT = {'LatentState': 'LS'}

# 150 delay values, matching MATLAB's tmp{i} = i*sp for i=1..150
CASES = list(range(1, 151))

# Cases highlighted in the trajectory plots (MATLAB: ix = [100, 5, 1])
SELECTED = [100, 5, 1]
CASE_COLORS = timeseries.MM_CASE_COLORS


def main():
    out_dir = figures_dir('run_memory_modification')
    par = patient_parameters()

    results = run_example(
        par=par,
        example=mm,
        models=MODELS,
        nrep=1,
        cases=CASES,
        observer=None,
        progress=True,
    )
    save_results(results, 'MemoryModification', out_dir=str(out_dir))

    for model_name, _ in MODELS:
        short = MODEL_SHORT[model_name]

        timeseries.plot_retrieval_time(
            results, model_name, CASES, short, out_dir,
        )
        timeseries.plot_expectations_by_case(
            results, model_name, SELECTED, CASE_COLORS, short, out_dir,
            inset=True,
        )
        timeseries.plot_beliefs_by_case(
            results, model_name, SELECTED, CASE_COLORS, short, out_dir,
            state_idx=0,
        )

    print(f'Figures written to {out_dir.resolve()}')


if __name__ == '__main__':
    main()
