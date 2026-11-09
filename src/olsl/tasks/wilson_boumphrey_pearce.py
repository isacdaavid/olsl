"""tasks/wilson_boumphrey_pearce.py — Wilson, Boumphrey & Pearce (1992).

Translation of MATLAB ``WilsonBoumphreyPearce.m``.

Two groups differing only in Phase 2:

    Case E ("Experimental")
        Phase 1 (trials  1–10): light+tone, alternating reward
        Phase 2 (trials 11–50): intermixed compound (light+tone) and
                                light-alone, alternating reward
        Phase 3 (trials 51–60): light alone, always rewarded

    Case C ("Control")
        Phase 1 (trials  1–50): light+tone, alternating reward
        Phase 2 (trials 51–60): light alone, always rewarded

    In Case E, the light becomes a *less accurate* predictor of reward in
    Phase 2 because it appears in both rewarded and unrewarded trials and
    sometimes with the tone, sometimes without. According to the
    Pearce-Hall account, that decline in predictive accuracy keeps the
    light's associability high. When it is then reliably reinforced in
    Phase 3, it acquires associative strength faster than in Case C,
    where the light was a consistent predictor throughout.

Outcome rows (columns: US, light, tone, light×tone):
    row 1: (1, 1, 1, 1)  US+, light, tone            (rewarded compound)
    row 2: (0, 1, 1, 1)  US-, light, tone            (unrewarded compound)
    row 3: (0, 1, 0, 0)  US-, light alone
    row 4: (1, 1, 0, 0)  US+, light alone            (rewarded light)

Model-dependent cue reduction:
    LatentState models: keep the light×tone interaction column → C=3
    All other models:   drop interaction, cues are [light, tone] → C=2

``par['nphases'] = 9`` is preserved from MATLAB but is not consumed by
the model or the plots. It has no arithmetic relation to the trial
structure (which has 60 trials) and appears to be dead metadata.
"""

import numpy as np


def _case_str(case):
    """Coerce par['case'] into a plain str, tolerating list/array wrappers."""
    if not isinstance(case, str):
        case = np.ravel(case)[0]
        case = str(case)
    return case


def initialize(par):
    par = dict(par)

    par['A']       = 1
    par['D']       = 1
    par['C']       = 2       # overridden below for LatentState
    par['nphases'] = 9
    par['ntrials'] = 60
    par['L']       = 1
    par['center']  = 0.5

    # Outcomes: [US, light, tone, light×tone]
    par['outcomes'] = np.array([
        [1.0, 1.0, 1.0, 1.0],   # 1: US+, light, tone
        [0.0, 1.0, 1.0, 1.0],   # 2: US-, light, tone
        [0.0, 1.0, 0.0, 0.0],   # 3: US-, light alone
        [1.0, 1.0, 0.0, 0.0],   # 4: US+, light alone
    ])

    case = _case_str(par['case'])

    if case == 'E':
        par['trialorder'] = np.concatenate([
            np.tile([1, 2], 5),           # Phase 1: 10 trials (compound)
            np.tile([1, 3], 20),          # Phase 2: 40 trials (mixed)
            np.full(10, 4, dtype=int),    # Phase 3: 10 trials (light+)
        ])
    elif case == 'C':
        par['trialorder'] = np.concatenate([
            np.tile([1, 2], 25),          # Phase 1: 50 trials (compound)
            np.full(10, 4, dtype=int),    # Phase 2: 10 trials (light+)
        ])
    else:
        raise ValueError(
            f"WilsonBoumphreyPearce case must be 'E' or 'C', got {case!r}"
        )

    # Model-dependent cue reduction
    if 'LatentState' in par.get('modelname', ''):
        par['C'] = 3    # light, tone, light×tone
    else:
        par['outcomes'] = par['outcomes'][:, :3].copy()   # US, light, tone

    return par


def randomize(par):
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
