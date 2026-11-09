"""tasks/backwards_blocking.py — Backward blocking.

28 trials:
    1–20:  AB+  (compound, reinforced)
    21–28: A+   (single cue, reinforced)

Expected behavior: A and B share associative strength during compound
conditioning. When A is then reinforced alone, the prediction error
requires A to grow while B must shrink (to keep the compound prediction
correct). B's weight drops even though B is never on-screen alone —
which requires the model to represent the cue-cue relationship, not
just cue-reward.
"""

import numpy as np
from ._blocking_common import finalize_outcomes, randomize as _shared_randomize


def initialize(par):
    par = dict(par)
    par['A']       = 1
    par['D']       = 1
    par['L']       = 1
    par['center']  = 0.5
    par['nphases'] = 28
    par['ntrials'] = 28

    trialorder = []
    for p in range(1, par['nphases'] + 1):
        if p <= 20:
            trialorder.append(3)         # AB+
        else:
            trialorder.append(1)         # A+
    par['trialorder'] = np.array(trialorder, dtype=int)

    return finalize_outcomes(par)


def randomize(par):
    return _shared_randomize(par)
