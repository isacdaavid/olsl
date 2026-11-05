"""tasks/pree.py — Partial Reinforcement Extinction Effect task.

Literal translation of MATLAB ``PREE.m``, exposed as two module-level
functions matching the ``run_task`` interface:

    initialize(par) -> par
    randomize(par)  -> dict with 'win' and 'c_vec'

Deviations from MATLAB (all shape/indexing, none semantic):

  - ``switch step`` dispatch → two named functions.
  - ``par['case']`` may be a scalar or length-1 sequence in the caller;
    MATLAB's ``par.case(1)`` accepts both, so we ravel-and-take-0.
  - ``c_vec`` is produced with shape (D, C, L*ncop, A), not MATLAB's
    (D, C, L*ncop). MATLAB silently accepts ``c_vec(:,:,:,1)`` on a 3-D
    array because A=1 for PREE; Python cannot, so the A axis is explicit.
  - ``par['t']`` is 1-based (matches MATLAB). Indexing ``trialorder`` uses
    ``par['t'] - 1``.
"""

import numpy as np


def initialize(par):
    """Set up PREE task parameters for the given case.

    Cases:
        1: partial reinforcement (alternating) -> extinction  (30 trials)
        2: continuous reinforcement            -> extinction  (30 trials)
        3: partial reinforcement -> continuous -> extinction  (50 trials)
        4: continuous            -> continuous -> extinction  (50 trials)
    """
    par = dict(par)   # shallow copy; do not mutate the caller's dict

    par['A']       = 1     # Number of arms
    par['D']       = 1     # Dimension of outcomes
    par['C']       = 1     # Dimension of cues
    par['L']       = 1     # Initial number of latent states
    par['center']  = 0.5   # Reward centering value
    par['ntrials'] = 30

    # Rewards + cues encoded as rows: [reward, cue]
    par['outcomes'] = np.array([
        [1.0, 1.0],   # A+  (rewarded)
        [0.0, 1.0],   # A   (not rewarded)
    ])

    # par['case'] may be a scalar or a length-1 sequence
    case = int(np.ravel(par['case'])[0])

    if case == 1:      # PREE -> no reward
        par['trialorder'] = np.concatenate([
            np.tile([1, 2], 10),           # alternating: 20 trials
            np.full(10, 2, dtype=int),     # extinction: 10 trials
        ])
    elif case == 2:    # Continuous -> no reward
        par['trialorder'] = np.concatenate([
            np.tile([1, 1], 10),           # always-rewarded: 20 trials
            np.full(10, 2, dtype=int),     # extinction:      10 trials
        ])
    elif case == 3:    # PREE -> reward -> no reward
        par['ntrials'] = 50
        par['trialorder'] = np.concatenate([
            np.tile([1, 2], 10),           # alternating: 20 trials
            np.ones(20, dtype=int),        # rewarded:    20 trials
            np.full(10, 2, dtype=int),     # extinction:  10 trials
        ])
    elif case == 4:    # Continuous -> reward -> no reward
        par['ntrials'] = 50
        par['trialorder'] = np.concatenate([
            np.tile([1, 1], 10),
            np.ones(20, dtype=int),
            np.full(10, 2, dtype=int),
        ])
    else:
        raise ValueError(f"Unknown PREE case: {case!r} (expected 1-4)")

    return par


def randomize(par):
    """Return the current trial's reward and cue vector.

    Output dict:
        'win'   : Python float (the current reward)
        'c_vec' : ndarray of shape (D, C, L*ncop, A)
    """
    # par['t'] is 1-based
    trial_type = int(par['trialorder'][par['t'] - 1])

    # par['outcomes'] rows are 1-based in the trialorder encoding
    row = trial_type - 1

    reward = float(par['outcomes'][row, 0])
    cue    = par['outcomes'][row, 1:]    # shape (C,)

    L_total = par['L'] * par['ls']['ncop']
    c_vec = np.zeros((par['D'], par['C'], L_total, par['A']))
    for l in range(L_total):
        for a in range(par['A']):
            c_vec[0, :, l, a] = cue

    return {'win': reward, 'c_vec': c_vec}
