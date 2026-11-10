"""tasks/memory_modification.py — Monfils/Schiller memory reconsolidation.

Translation of MATLAB ``MonfilsSchiller.m``.

Four-phase paradigm (Monfils et al. 2009; Schiller et al. 2010):

    Phase 1 — Acquisition : 3 trials (rewarded)
    Phase 2 — Retrieval   : 1 trial  (CS alone, unrewarded)
    Phase 3 — Extinction  : 19 trials (CS alone, unrewarded)
    Phase 4 — Test        : 2 trial slots (final one is the boundary)

The ``case`` value sets the retrieval→extinction delay, and controls
whether the extinction phase is experienced "inside" or "outside" the
reconsolidation window. Typical values from the paper:

    case=1    ~10 min    (inside window)
    case=5    ~1 hour    (inside window)
    case=100  ~6+ hours  (outside window)

Mechanism inside the model: a small gap keeps the agent in the
acquisition state during extinction, so ``Update``'s rumination loop
replays acquisition updates and reduces the acquired strength. A large
gap dilutes beliefs (via ``ContextShift``), so a new extinction state
forms and the acquisition memory survives — producing a return of fear
at test.

Notes
-----
* ``par['ntrials'] = sum(N) - 1 = 24``. ``par['time']`` has 25 entries —
  the final entry serves as the "end of final trial" boundary used by
  ``Update`` for rumination on the last simulated trial. As a
  consequence, only the first of the two Phase-4 test slots is
  simulated. This matches the MATLAB source; documented here so anyone
  cross-checking figures against the paper isn't surprised.

* ``par['ruminate'] = 3`` is set here but never consumed by
  ``LatentState.Update``, which uses ``par['ls']['chi']`` instead.
  Preserved for parity with the MATLAB source.

* No model-dependent cue reduction: C=1 (single CS) for every model.
"""

import numpy as np


def _case_int(case):
    """Coerce par['case'] into a plain int, tolerating list/array wrappers."""
    if not isinstance(case, (int, np.integer)):
        case = np.ravel(case)[0]
    return int(case)


def initialize(par):
    par = dict(par)

    par['A']        = 1
    par['D']        = 1
    par['C']        = 1
    par['L']        = 1
    par['center']   = 0.5
    par['ruminate'] = 3

    case = _case_int(par['case'])
    if case < 0:
        raise ValueError(
            f"MemoryModification case (retrieval→extinction gap) "
            f"must be >= 0, got {case}"
        )

    # Phase structure mirrors MonfilsSchiller.m:
    #   N = [3, 1, 19, 2]     trials per phase
    #   I = [0, 20, case, 200]  gap before each phase (in `time` units)
    N = [3, 1, 19, 2]
    I = [0, 20, case, 200]

    t = 1
    times = []
    for i in range(len(N)):
        t += I[i]
        for _ in range(N[i]):
            times.append(t)
            t += 1
    par['time'] = np.array(times, dtype=float)

    # MATLAB: par.ntrials = sum(N) - 1
    # The final `time` entry is a boundary, not a trial, so only
    # sum(N) - 1 = 24 actual trials are simulated. This drops the
    # second Phase-4 test slot; see module docstring.
    par['ntrials'] = sum(N) - 1

    return par


def randomize(par):
    """Return the current trial's reward and cue vector.

    Only the first 3 trials (acquisition) are reinforced; every other
    trial — retrieval, extinction, test — has reward 0. The single CS
    is present on every trial.
    """
    reward = 1.0 if par['t'] <= 3 else 0.0

    L_total = par['L'] * par['ls']['ncop']
    c_vec = np.ones((par['D'], par['C'], L_total, par['A']))

    return {'win': reward, 'c_vec': c_vec}
