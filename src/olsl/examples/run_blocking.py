"""Run and visualize the Blocking task."""

from olsl.common.paths import figures_dir
from olsl.common.patient_parameters import patient_parameters
from olsl.simulation._blocking_common import run_and_plot


def main():
    run_and_plot(
        'Blocking',
        par=patient_parameters(),
        out_dir=figures_dir('run_blocking'),
    )


if __name__ == '__main__':
    main()
