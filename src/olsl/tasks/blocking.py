"""tasks/blocking.py — Kamin-style blocking.

36 trials:
    1–20:  A+  (single cue, reinforced)
    21–36: 8 blocks of [AB+, CD+] — intermixed compound conditioning

Expected behavior: cue A acquires strong associative strength during the
single-cue phase, so when AB+ appears, A's prediction error is already
small and B's associative strength stays near zero. B's conditioning is
"blocked" by A. The CD+ trials serve as filler to maintain context, not
as part of the blocking manipulation itself.
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
    par['ntrials'] = 36

    trialorder = []
    for p in range(1, par['nphases'] + 1):
        if p <= 20:
            trialorder.append(1)         # A+
        else:
            trialorder.extend([3, 4])    # AB+, CD+
    par['trialorder'] = np.array(trialorder, dtype=int)

    return finalize_outcomes(par)


def randomize(par):
    return _shared_randomize(par)
