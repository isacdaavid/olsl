"""tasks/_blocking_common.py — shared helpers for the blocking family.

The Blocking, Backwards Blocking, Overexpectation, and Conditioned
Inhibition tasks all use the same 7-column outcomes matrix and the same
cue-vector construction. Only the trial schedule differs, so that
schedule lives in the per-task modules while everything else lives here.

Column layout of ``OUTCOMES_FULL``:
    [reward, A, B, C, D, AB_int, CD_int]

For non-LatentState models the interaction columns (6-7) are dropped,
giving C=4. For LatentState models they're kept, giving C=6 — the
agent then uses the interaction terms as explicit compound cues.
"""

import numpy as np


# Row indices correspond to 1-based trial-type labels used in trialorder:
#   0: A+      (trial_type 1)
#   1: B+      (trial_type 2)
#   2: AB+     (trial_type 3)
#   3: CD+     (trial_type 4)
#   4: ABweak+ (trial_type 5)
OUTCOMES_FULL = np.array([
    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # A+
    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],   # B+
    [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],   # AB+ (A, B, AB interaction)
    [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],   # CD+ (C, D, CD interaction)
    [0.5, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],   # ABweak+ (half-reinforced AB)
])


def finalize_outcomes(par):
    """Apply the model-dependent outcome reduction.

    LatentState models: keep all 7 columns (C=6).
    All other models:   keep only [reward, A, B, C, D] (C=4).

    Mirrors MATLAB's::

        if ~contains(par.modelname, 'LatentState')
            par.outcomes(:,6:end) = [];
            par.C = 4;
        else
            par.C = 6;
        end
    """
    if 'LatentState' in par.get('modelname', ''):
        par['outcomes'] = OUTCOMES_FULL.copy()
        par['C'] = 6
    else:
        par['outcomes'] = OUTCOMES_FULL[:, :5].copy()
        par['C'] = 4
    return par


def randomize(par):
    """Return the current trial's reward and cue vector.

    Identical cue vector across all L*ncop latent states, matching the
    block of MATLAB's ``Randomize`` case that every blocking-family task
    shares.
    """
    trial_type = int(par['trialorder'][par['t'] - 1])
    row = trial_type - 1

    reward = float(par['outcomes'][row, 0])
    cue    = par['outcomes'][row, 1:]    # (C,)

    L_total = par['L'] * par['ls']['ncop']
    c_vec = np.zeros((par['D'], par['C'], L_total, par['A']))
    for l in range(L_total):
        for a in range(par['A']):
            c_vec[0, :, l, a] = cue

    return {'win': reward, 'c_vec': c_vec}
