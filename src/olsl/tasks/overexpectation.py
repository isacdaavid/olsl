"""tasks/overexpectation.py — Overexpectation.

110 trials:
    1–100: A+, B+  (50 blocks of two separate single-cue trials)
    101–110: AB+   (10 compound trials)

Expected behavior: during single-cue conditioning A and B each acquire
strong associative strength. When compounded, their predicted rewards
sum to roughly twice the actual reward, producing a large negative
prediction error. Both A and B lose associative strength — the model
"learns" that it overpredicted.
"""

import numpy as np
from ._blocking_common import finalize_outcomes, randomize as _shared_randomize


def initialize(par):
    par = dict(par)
    par['A']       = 1
    par['D']       = 1
    par['L']       = 1
    par['center']  = 0.5
    par['nphases'] = 60
    par['ntrials'] = 110

    trialorder = []
    for p in range(1, par['nphases'] + 1):
        if p <= 50:
            trialorder.extend([1, 2])    # A+, B+
        else:
            trialorder.append(3)         # AB+
    par['trialorder'] = np.array(trialorder, dtype=int)

    return finalize_outcomes(par)


def randomize(par):
    return _shared_randomize(par)
