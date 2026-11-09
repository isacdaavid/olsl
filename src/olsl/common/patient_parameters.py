"""common/patient_parameters.py — default model parameters.

Translation of MATLAB ``PatientParameters.m``. Task-specific dimensions
(``D``, ``C``, ``L``, ``A``, ``center``) are *not* included here: each
task's ``initialize()`` sets them, which keeps a single source of truth
for what each task is.
"""


def patient_parameters():
    """Return a dict of default patient/model parameters.

    Keys mirror MATLAB's struct field names so a caller can cross-check
    against ``PatientParameters.m`` line-by-line.
    """
    par = {}

    # Softmax exploration parameter (all models)
    par['tau'] = 10.0

    # --- LatentState model ---
    par['ls'] = {
        'alpha0': 0.05,   # learning rate for associative strength
        'alpha1': 0.05,   # learning rate for variance
        'alpha2': 0.05,   # learning rate for covariance
        'gamma':  0.05,   # transition probability between states
        'ncop':   10,     # max replicates of latent states
        'eta':    0.2,    # threshold to activate new state
        'delta':  0.6,    # constant in change-point statistic
        'chi':    5,      # number of rumination updates
        'sigma0': 0.5,    # initial standard deviation
    }

    # --- Rescorla-Wagner ---
    par['rw'] = {'alpha': 0.15}

    # --- Hybrid model ---
    par['hybrid'] = {'kappa': 0.3, 'eta': 0.857, 'alpha': 0.926}

    # --- fRL decay ---
    par['fRLDecay'] = {'eta': 0.15, 'd': 0.45}

    # --- Infinite mixture model ---
    par['imm'] = {'alpha': 1.0, 'M': 100, 'K': 15}

    # --- Gershman 2017 (memory modification) ---
    par['gershman'] = {
        'alpha': 0.1, 'g': 1.0, 'eta': 0.3, 'w0': 0.0,
        'sr': 0.4, 'sx': 1.0, 'K': 15, 'EMSteps': 3,
    }

    # --- Multiple-Model Kalman Filter observer ---
    # R and gamma inherit from ls so that changing sigma0 or gamma in
    # one place propagates to the observer. MATLAB does the same.
    par['kf'] = {
        'Q':     0.01,
        'R':     par['ls']['sigma0'] ** 2,
        'initP': 1.0,
        'L':     2,
        'gamma': par['ls']['gamma'],
    }

    return par
