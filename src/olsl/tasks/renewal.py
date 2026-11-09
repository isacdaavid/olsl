"""tasks/renewal.py — Renewal of extinguished responding task.

Translation of MATLAB ``Renewal.m``, exposed as two module-level
functions matching the ``run_task`` interface:

    initialize(par) -> par
    randomize(par)  -> dict with 'win' and 'c_vec'

Task structure (50 trials, 3 phases):
    Acquisition : trials  1–20   reward=1  (context A, cue +)
    Extinction  : trials 21–40   reward=0  (context A for case 1, context B for case 2)
    Renewal     : trials 41–50   reward=1  (context A, cue +)

Two cases:
    case=1 : extinction in the same context as acquisition (no renewal signal)
    case=2 : extinction in a different context (context shift at trials 21, 41)

Two possible model interfaces:
    - LatentState models: context is *not* a cue; instead context changes
      are encoded as ``par['context']`` = trial indices at which the
      agent's beliefs should be reset to uniform. Cue vector has C=1.
    - All other models: context is an explicit cue (C=3, one per
      context plus one for the cue). ``par['context']`` is absent.

Deviations from MATLAB (all shape/indexing, none semantic):
  - ``switch step`` dispatch → two named functions.
  - ``par['case']`` may be a scalar or length-1 sequence; matches
    MATLAB's ``par.case(1)``.
  - ``par['outcomes']``'s column reduction for LatentState is
    ``[:, [0, 3]]`` instead of MATLAB's ``:,[1,4]`` (Python 0-based).
  - ``c_vec`` is produced with shape (D, C, L*ncop, A) explicitly,
    rather than (D, C, L*ncop) as MATLAB produces for A=1.
"""

import numpy as np


def initialize(par):
    """Set up Renewal task parameters for the given case."""
    par = dict(par)

    par['A']       = 1      # Number of arms
    par['D']       = 1      # Dimension of outcomes
    par['C']       = 3      # Dimension of cues (context A, context B, cue)
    par['nphases'] = 3      # Number of task phases (blocks)
    par['ntrials'] = 50
    par['L']       = 1      # Initial number of latent states
    par['center']  = 0.5    # Reward centering
    par['r0']      = 1.0    # Reward value

    case = int(np.ravel(par['case'])[0])
    if case not in (1, 2):
        raise ValueError(f"Renewal case must be 1 or 2, got {case!r}")

    # --- Trial order (1-based trial types matching outcomes rows) ---
    # Row 1: [1, 1, 0, 1]  Context A, Cue+   (acquisition + renewal)
    # Row 2: [0, 1, 0, 1]  Context A, Cue    (case 1 extinction)
    # Row 3: [0, 0, 1, 1]  Context B, Cue    (case 2 extinction)
    trialorder = np.empty(par['ntrials'], dtype=int)
    trialorder[:20]    = 1              # acquisition
    trialorder[20:40]  = case + 1       # extinction (row 2 or 3)
    trialorder[40:]    = 1              # renewal
    par['trialorder']  = trialorder

    # --- Outcomes matrix: [reward, ctx_A, ctx_B, cue] ---
    par['outcomes'] = np.array([
        [1.0, 1.0, 0.0, 1.0],   # Context A, Cue+ (rewarded)
        [0.0, 1.0, 0.0, 1.0],   # Context A, Cue  (unrewarded, same ctx)
        [0.0, 0.0, 1.0, 1.0],   # Context B, Cue  (unrewarded, diff ctx)
    ])

    # --- LatentState-specific adjustments ---
    # LatentState models treat context as hypotheses, not as cues: the
    # context is signalled by resetting beliefs at the phase boundaries.
    if 'LatentState' in par.get('modelname', ''):
        par['C'] = 1
        par['outcomes'] = par['outcomes'][:, [0, 3]]   # reward + cue only

        if case == 2:
            # Context shift at the start of extinction (trial 21) and
            # renewal (trial 41). 1-based trial indices.
            par['context'] = [21, 41]

    return par


def randomize(par):
    """Return the current trial's reward and cue vector.

    Output dict:
        'win'   : Python float (the current reward)
        'c_vec' : ndarray of shape (D, C, L*ncop, A)
    """
    trial_type = int(par['trialorder'][par['t'] - 1])
    row = trial_type - 1

    reward = float(par['outcomes'][row, 0])
    cue    = par['outcomes'][row, 1:]   # shape (C,)

    L_total = par['L'] * par['ls']['ncop']
    c_vec = np.zeros((par['D'], par['C'], L_total, par['A']))
    for l in range(L_total):
        for a in range(par['A']):
            c_vec[0, :, l, a] = cue

    return {'win': reward, 'c_vec': c_vec}
