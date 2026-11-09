"""tasks/spontaneous_recovery.py — Spontaneous recovery of extinguished responding.

Translation of MATLAB ``SpontaneousRecovery.m``.

Task structure (50 trials, 3 phases):
    Acquisition : trials  1–20   cue+ (reward=1)
    Extinction  : trials 21–40   cue  (reward=0)
    Renewal     : trials 41–50   cue+ (reward=1)

Unlike Renewal, context is **not** encoded as a cue — C=1 for all models.
The mechanism for spontaneous recovery is purely temporal: ``par['case']``
controls a silent gap inserted between extinction and renewal, and the
agent's forgetting is driven by how long that gap is. Larger ``par['case']``
means a longer gap, which drives stronger decay of the extinction belief
(via ``LatentState.ContextShift``) and stronger return of responding.

This is why ``par['time']`` is stored: both ``ContextShift`` (belief decay)
and ``Update`` (rumination count) reference the time between consecutive
trials, and the silent gap is represented as a jump in ``par['time']``.

``par['case']`` semantics:
    case=1  → gap of 0 trials (no shift; baseline)
    case=2  → gap of 1 trial
    case=k  → gap of (k - 1) trials
"""

import numpy as np


def initialize(par):
    """Set up Spontaneous Recovery task parameters."""
    par = dict(par)

    par['A']       = 1
    par['D']       = 1
    par['C']       = 1
    par['ntrials'] = 50
    par['L']       = 1
    par['center']  = 0.5
    par['r0']      = 1.0

    case = int(np.ravel(par['case'])[0])
    if case < 1:
        raise ValueError(
            f"SpontaneousRecovery case must be >= 1 (gap = case - 1), "
            f"got {case}"
        )

    # Phase boundaries (1-based trial indices): acquisition ends at 20,
    # extinction ends at 40.
    phase_boundaries = (20, 40)

    # --- Trial order: 1 = cue+, 2 = cue- ---
    trialorder = np.empty(par['ntrials'], dtype=int)
    trialorder[:phase_boundaries[0]]                     = 1   # acquisition
    trialorder[phase_boundaries[0]:phase_boundaries[1]]  = 2   # extinction
    trialorder[phase_boundaries[1]:]                     = 1   # renewal
    par['trialorder'] = trialorder

    # --- Outcomes: [reward, cue] ---
    par['outcomes'] = np.array([
        [1.0, 1.0],   # Cue+ (rewarded)
        [0.0, 1.0],   # Cue  (unrewarded; cue still on-screen)
    ])

    # --- Inter-trial "time" ---
    # Base is 1..(ntrials+1) — 51 values, matching MATLAB's
    # ``par.time = (1:par.ntrials+1)``.
    # The renewal phase (1-based trials 41+) is pushed forward by
    # (case - 1) to encode the silent gap. The gap surfaces in
    # LatentState as a larger ``nIter`` and stronger belief decay at
    # the extinction → renewal boundary.
    par['time'] = np.arange(1, par['ntrials'] + 2, dtype=float)
    par['time'][phase_boundaries[1]:] += (case - 1)

    return par


def randomize(par):
    """Return the current trial's reward and cue vector."""
    trial_type = int(par['trialorder'][par['t'] - 1])
    row = trial_type - 1

    reward = float(par['outcomes'][row, 0])
    cue    = par['outcomes'][row, 1:]

    L_total = par['L'] * par['ls']['ncop']
    c_vec = np.zeros((par['D'], par['C'], L_total, par['A']))
    for l in range(L_total):
        for a in range(par['A']):
            c_vec[0, :, l, a] = cue

    return {'win': reward, 'c_vec': c_vec}
