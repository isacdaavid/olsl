"""simulation/run_task.py — single-replicate simulation loop.

Literal translation of MATLAB ``RunTask.m``. The following substitutions
were made to avoid MATLAB idioms that have no clean Python counterpart:

  - ``eval([example, '(...)', par])``  →  ``example.initialize(par)`` /
                                            ``example.randomize(par)``
    where ``example`` is a module exposing those two functions.
  - ``eval([approach, '(...)', ...])`` →  ``approach(par, agent, world)``
    where ``approach`` is a callable (typically a class).
  - ``agent.V``, ``agent.win``, …      →  ``agent['V']``, ``agent['win']``, …
    (dict-like, matching the MATLAB struct).
  - Arrays that grow inside the MATLAB loop (``V(t+1,:) = …``) are
    pre-allocated to ``(ntrials+1, …)`` before the loop.
  - MATLAB is 1-based; loop index ``t`` is 0-based in Python, but
    ``par['t']`` is kept 1-based so downstream code that indexes
    ``par['time'][par['t']-1]`` still works.
"""

import numpy as np

def run_task(par, example, approach, observer=None):
    """Run one simulation replicate.

    Input:
        par      : dict of parameters. Must contain 'ntrials', 'D', 'A',
                   'center'. If ``observer`` is not None, must also contain
                   'kf' with at least 'L'.
        example  : module exposing ``initialize(par) -> par`` and
                   ``randomize(par) -> world``. Replaces MATLAB's
                   ``eval([example, '(''Initialize'', par)'])``.
        approach : callable ``(par, agent, world) -> agent``. Call with
                   ``agent=None`` for initialization. Replaces MATLAB's
                   ``eval([approach, '(par, ...)'])``.
        observer : optional callable ``(par, mmkf, agent, world) -> mmkf``.
                   Call with ``mmkf=None`` on the first trial. If None,
                   the MMKF observer is skipped and the MMKF_* outputs
                   are None.

    Output: dict with keys
        'mu', 'V', 'lsb', 'X_t_seq',
        'MMKF_lsb_prior', 'MMKF_likelihood', 'MMKF_lsb_post', 'MMKF_X_t_est',
        'mmkf', 'agent', 'par'
    """

    # --- Initialize task and agent --------------------------------------
    par = example.initialize(par)
    agent = approach(par, None, None)

    D = par['D']
    A = par['A']
    ntrials = par['ntrials']

    # --- Pre-allocate output arrays (replaces MATLAB's dynamic growth) --
    # V: flatten agent['V'] using Fortran order to match MATLAB's A(:)
    V_flat_size = agent['V'].size
    V = np.full((ntrials + 1, V_flat_size), np.nan)
    V[0, :] = agent['V'].flatten(order='F')

    # lsb: only if the model tracks it
    if 'lsb' in agent:
        lsb_size = agent['lsb'].size
        lsb = np.full((ntrials + 1, lsb_size), np.nan)
        lsb[0, :] = agent['lsb']
    else:
        lsb = None

    # mu: fixed at D*A, matching the LatentState contract for agent['mu']
    mu = np.full((ntrials, D * A), np.nan)

    # X_t_seq: 1-D array, filled if agent['x_t'] exists
    X_t_seq = np.full(ntrials, np.nan)

    # MMKF: allocate only if observer is provided
    if observer is not None:
        L_kf = par['kf']['L']
        MMKF_lsb_prior  = np.full((ntrials, L_kf), np.nan)
        MMKF_likelihood = np.full((ntrials, L_kf), np.nan)
        MMKF_lsb_post   = np.full((ntrials, L_kf), np.nan)
        MMKF_X_t_est    = np.full(ntrials, np.nan)
        mmkf = None
    else:
        MMKF_lsb_prior = MMKF_likelihood = MMKF_lsb_post = MMKF_X_t_est = None
        mmkf = None

    # --- Trial loop -----------------------------------------------------
    for t in range(ntrials):
        par['t'] = t + 1   # 1-based, to match MATLAB

        # par.lastwin = ~(t == 1 || agent.win == 0)
        # On t=0 (trial 1), MATLAB short-circuits so agent.win isn't read.
        if t == 0:
            par['lastwin'] = False
        else:
            par['lastwin'] = bool(
                not np.all(np.atleast_1d(agent['win']) == 0)
            )

        # Get rewards and cue vector
        world = example.randomize(par)

        # Update agent
        agent = approach(par, agent, world)

        # Run observer alongside the agent using only observable data
        if observer is not None:
            mmkf = observer(par, mmkf, agent, world)

        # --- Logging ---
        V[t + 1, :] = agent['V'].flatten(order='F')

        if 'lsb' in agent and lsb is not None:
            lsb[t + 1, :] = agent['lsb']

        if 'mu' in agent:
            mu_flat = np.atleast_1d(agent['mu']).flatten()
            if mu_flat.size != D * A:
                raise ValueError(
                    f"agent['mu'] has {mu_flat.size} elements; "
                    f"expected D*A = {D * A}"
                )
            mu[t, :] = mu_flat

        if 'x_t' in agent:
            X_t_seq[t] = agent['x_t']

        if observer is not None:
            MMKF_lsb_prior[t, :]  = mmkf['lsb_prior']
            MMKF_likelihood[t, :] = mmkf['likelihoods']
            MMKF_lsb_post[t, :]   = mmkf['lsb']
            MMKF_X_t_est[t]       = mmkf['X_t_est']

    return {
        'mu':             mu,
        'V':              V,
        'lsb':            lsb,
        'X_t_seq':        X_t_seq,
        'MMKF_lsb_prior':  MMKF_lsb_prior,
        'MMKF_likelihood': MMKF_likelihood,
        'MMKF_lsb_post':   MMKF_lsb_post,
        'MMKF_X_t_est':    MMKF_X_t_est,
        'mmkf':           mmkf,
        'agent':          agent,
        'par':            par,
    }
