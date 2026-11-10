"""tasks/rescorla_2000.py — Rescorla (2000) Experiments 1A and 1B.

Translation of MATLAB ``Rescorla2000.m``.

Task structure (512 trials):
    Phase 1 (p=1..100, 500 trials):
        5 single-cue trials per phase, in random order each phase:
            A+  (reward=1, cue A)
            C+  (reward=1, cue C)
            X+  (reward=1, cue X)
            BX- (reward=0, cues B and X)
            DX- (reward=0, cues D and X)
    Phase 2 (p=101..108, 8 trials):
        case=1: AB+ (reward=1, cues A and B) — Experiment 1A
        case=2: AB- (reward=0, cues A and B) — Experiment 1B
    Phase 3 (p=109..110, 4 trials):
        Test compounds AD- and BC- in random order.

Design rationale: Phase 1 creates an ambiguous cue structure. A and C
are perfect predictors of reward (always reinforced alone). X is always
unreinforced. B and D are ambiguous — they appear in unreinforced
compounds with X (BX- and DX-). Phase 2 adds compound information about
A and B: reinforced (AB+, case 1) makes B excitatory, unreinforced (AB-,
case 2) makes B inhibitory. Phase 3 then tests whether the agent expects
D or C to predict reward alongside A/B — the associative strength of the
test compounds reveals what the agent learned about B's sign.

Outcome rows (columns: US, A, B, C, D, X, BX_int, DX_int, AB_int,
AD_int, BC_int):
    row 1: A+
    row 2: C+
    row 3: X+
    row 4: BX-
    row 5: DX-
    row 6: AB+ (case 1 compound)
    row 7: AB- (case 2 compound)
    row 8: AD- (test compound)
    row 9: BC- (test compound)

Model-dependent cue reduction:
    LatentState models: keep all 10 cue columns → C=10
    All other models:   drop the 5 interaction columns → C=5
        (cues are A, B, C, D, X)

``rng_seed`` (default 1): controls trial-order randomization. MATLAB
hard-codes ``rng(1)`` so all replicates of the same case produce
identical schedules. Pass ``par['rng_seed']`` to override; the caller
is responsible for varying it across replicates if they want
schedule-level variation.
"""

import numpy as np


# Cue columns: A, B, C, D, X, then 5 interaction terms.
OUTCOMES_FULL = np.array([
    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # A+
    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # C+
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # X+
    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # BX-
    [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # DX-
    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # AB+
    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # AB-
    [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # AD-
    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # BC-
])


def _case_int(case):
    """Coerce par['case'] into a plain int, tolerating list/array wrappers."""
    if not isinstance(case, (int, np.integer)):
        case = np.ravel(case)[0]
    return int(case)


def initialize(par):
    par = dict(par)

    p1 = 100   # phases of single-cue training

    par['A']       = 1
    par['D']       = 1
    par['C']       = 10
    par['nphases'] = p1 + 10       # 100 training + 8 compound + 2 test
    par['ntrials'] = p1 * 5 + 8 + 4   # 500 + 8 + 4 = 512
    par['L']       = 1
    par['center']  = 0.5

    case = _case_int(par['case'])
    if case not in (1, 2):
        raise ValueError(
            f"Rescorla2000 case must be 1 or 2 (1=AB+ / 2=AB-), got {case!r}"
        )

    # --- Trial order ---
    # MATLAB hard-codes rng(1) inside initialize; use a local Generator
    # so the global numpy RNG is untouched, and read the seed from par
    # so callers can vary it if desired.
    rng = np.random.default_rng(par.get('rng_seed', 1))

    trial_chunks = []
    for p in range(1, par['nphases'] + 1):
        if p <= p1:
            # 5 single-cue trial types in random order (1..5)
            ix = rng.permutation(5) + 1
        elif p <= p1 + 8:
            # Compound trial: row 6 (AB+) or 7 (AB-)
            ix = np.array([5 + case], dtype=int)
        else:
            # Test compound trials: rows 8 (AD-) and 9 (BC-) in random order
            ix = rng.permutation(2) + 8
        trial_chunks.append(ix)
    par['trialorder'] = np.concatenate(trial_chunks).astype(int)

    # --- Outcomes (11 columns: US + 10 cues, incl. interactions) ---
    par['outcomes'] = OUTCOMES_FULL.copy()

    # --- Model-dependent cue reduction ---
    if 'LatentState' not in par.get('modelname', ''):
        par['outcomes'] = par['outcomes'][:, :6].copy()   # US + 5 single cues
        par['C'] = 5

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
