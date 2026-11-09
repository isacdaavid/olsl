"""tasks/conditioned_inhibition.py — Conditioned inhibition.

200 trials:
    100 blocks of [A+, ABweak+]

Expected behavior: A is always reinforced whether alone (A+, reward=1) or
with B (ABweak+, reward=0.5). Over many trials A grows to predict
reward reliably while B acquires negative associative strength — its
presence signals a lower reward than A alone. B becomes a conditioned
inhibitor.
"""

import numpy as np
from ._blocking_common import finalize_outcomes, randomize as _shared_randomize


def initialize(par):
    par = dict(par)
    par['A']       = 1
    par['D']       = 1
    par['L']       = 1
    par['center']  = 0.5
    par['nphases'] = 100
    par['ntrials'] = 200

    trialorder = []
    for p in range(1, par['nphases'] + 1):
        trialorder.extend([1, 5])        # A+, ABweak+
    par['trialorder'] = np.array(trialorder, dtype=int)

    return finalize_outcomes(par)


def randomize(par):
    return _shared_randomize(par)
