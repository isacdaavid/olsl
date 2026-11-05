import numpy as np

from olsl.common.math import SoftMaxChoice

def LatentState(par, agent, world):
    """
    Latent-state learning approach
    Input:
         par   = dictionary of parameters
         agent = dictionary of variables tracked by agent (or None for init)
         world = dictionary of variables generated from task/world
    Output:
         agent = updated dictionary of variables tracked by agent
    """
    # Number of latent causes (ncop x initial # of hypotheses)
    L = par['L'] * par['ls']['ncop']

    # -------------------------------------------------------------------------
    # Initialization of agent
    # -------------------------------------------------------------------------
    if agent is None:
        agent = {}

        if 'initialV' in par:
            agent = par['initialV']
        else:
            # Initial expectations (neutral, unless otherwise specified)
            if 'initialex' in par:
                agent['V'] = par['initialex']
            else:
                agent['V'] = np.zeros((par['D'], par['C'], L))

            # Each hypothesis is initially active
            agent['Lmax'] = 1 

            # Change point statistic
            agent['q'] = 0.0

            # Initial log-beliefs (evenly divided among initial hypotheses)
            agent['lsb'] = np.zeros(L)
            if 'lsb0' in par:
                agent['lsb'][:par['L']] = par['lsb0']
            else:
                agent['lsb'][:par['L']] = 1.0 / par['L']

            # Initial B matrix (identity) for cues per dimension & state 
            agent['B'] = np.zeros((par['C'], par['C'], par['D'], L))
            for d in range(par['D']):
                for l in range(L):
                    agent['B'][:, :, d, l] = np.eye(par['C'])

            # Initial variance matrix of expected mean
            agent['sigmasq'] = par['ls']['sigma0'] ** 2
            
        return agent

    # -------------------------------------------------------------------------
    # Update of agent    
    # -------------------------------------------------------------------------
    else:
        # Get variables
        V         = agent['V']              
        lsb       = agent['lsb']            
        B         = agent['B']              
        Lmax      = agent['Lmax']           
        q         = agent['q']              
        sigmasq0  = agent['sigmasq']        
        c_vec     = world['c_vec']
        # Standardize win to shape (D, A)
        win = world['win'] - par['center']
        if np.isscalar(win) or win.ndim == 0:
            win = np.array([[win]])
        elif win.ndim == 1:
            if len(win) == par['A']:
                win = win.reshape(1, -1)  # (1, A)
            else:
                win = win.reshape(-1, 1)  # (D, 1)

        # Context or temporal shift
        lsb = ContextShift(lsb, Lmax, par)

        # Choice behavior
        # Value function per arm, latent state: (D, L, A)
        mu  = np.reshape(
            np.sum(V[..., None] * c_vec, axis=1),
            (par['D'], L, par['A'])
        )
        
        # Value function per arm: (D, A)
        # We use lsb[None, :, None] to broadcast (D, L, A) * (1, L, 1)
        mu0 = np.reshape(np.sum(mu * lsb[None, :, None], axis=1), (par['D'], par['A']))
        
        # Choose arm and update relevant variables based on arm
        arm, _ = SoftMaxChoice(mu0, par)
        win = win[:, arm]           # (D,)
        mu = mu[:, :, arm]          # (D, L)
        c_vec = c_vec[:, :, :, arm] # (D, C, L)

        # Updates
        LL = LogLikelihood(win, mu, sigmasq0, Lmax, L, par['ls']['ncop'])
        lsb = LatentStateBeliefs(lsb, Lmax, LL, par)
        V, lsb, Lmax, q = NewState(V, lsb, Lmax, c_vec, win, LL, q, par)
        V, B, sigmasq = Update(V, B, sigmasq0, lsb, Lmax, c_vec, win, par)

        # Output updated variables
        agent['V']       = V
        agent['lsb']     = lsb
        agent['B']       = B
        agent['sigmasq'] = sigmasq
        agent['Lmax']    = Lmax
        agent['q']       = q
        agent['arm']     = arm
        agent['win']     = win
        agent['mu']      = mu0

        return agent

# -------------------------------------------------------------------------
# Sub-functions
# -------------------------------------------------------------------------

def ContextShift(lsb, Lmax, par):
    # Temporal shift
    if par['t'] != 1 and 'time' in par:
        # par['t'] is 1-based, so current time is par['time'][par['t'] - 1]
        # previous time is par['time'][par['t'] - 2]
        prob_stay = (1 - par['ls']['gamma']) ** (par['time'][par['t']-1] - par['time'][par['t']-2] - 1)
        lsb[:Lmax*par['L']] = lsb[:Lmax*par['L']] * prob_stay + (1 - prob_stay) / (Lmax * par['L'])

    # Shift based to old context re-initialize beliefs
    if 'context' in par and par['t'] in par['context']:
        lsb[:par['L']*Lmax] = 1.0
        lsb = lsb / np.sum(lsb)
        
    return lsb

def LogLikelihood(win, mu, sigmasq, Lmax, L, ncop):
    # Log-likelihood of each latent state
    LL = np.zeros(L)
    ix_limit = int((L / ncop) * (Lmax + 1))
    LL[ix_limit:] = -np.inf

    # Index of activated states
    ix = int((L / ncop) * Lmax)
    
    # Error: win is (D,), mu is (D, L). win[:, None] is (D, 1)
    LL[:ix] = np.sum(-(win[:, None] - mu[:, :ix])**2 / (2 * sigmasq), axis=0)
    
    return LL

def LatentStateBeliefs(lsb, Lmax, LL, par):
    # Update latent state beliefs
    ix1 = int(Lmax * par['L'])

    # Current (scaled) likelihood
    tmp = np.zeros(par['L'] * par['ls']['ncop'])
    
    # We use np.max(LL[:ix1]) for numerical stability
    tmp[:ix1] = ((1 - par['ls']['gamma']) * lsb[:ix1] + par['ls']['gamma'] / (Lmax * par['L'])) * np.exp(LL[:ix1] - np.max(LL[:ix1]))

    # Normalize
    lsb = tmp / np.sum(tmp)
    return lsb

def NewState(V, lsb, Lmax, c_vec, win, LL, q, par):
    # Check if new state should be added
    if Lmax < par['ls']['ncop']:
        
        tmp = np.zeros(par['L'] * par['ls']['ncop'])

        ix1_end = int(par['L'] * Lmax)
        ix2_start = int(par['L'] * Lmax)
        ix2_end = int(par['L'] * Lmax + par['L'])
        
        # Concatenate indices for ix1 and ix2
        idx = np.r_[0:ix1_end, ix2_start:ix2_end]
        
        mx = np.max(np.exp(LL[idx]))
        tmp[:ix1_end] = ((1 - par['ls']['gamma']) * lsb[:ix1_end] + par['ls']['gamma'] / (Lmax * par['L'])) * np.exp(LL[:ix1_end] - mx)

        # Current (scaled) likelihood
        l0 = np.sum(tmp)
        
        # (Scaled) likelihood if adding new states
        l1 = np.sum((1.0 / par['L']) * np.exp(LL[ix2_start:ix2_end] - mx))
        
        # Update change point statistic
        q = np.maximum(q + np.log(l1) - np.log(l0) - par['ls']['delta'], 0.0)

        # Check if new state is activated
        if q >= par['ls']['eta']:
            Lmax = Lmax + 1
            q = 0.0
            
            # Update beliefs
            tmp[idx] = ((1 - par['ls']['gamma']) * lsb[idx] + par['ls']['gamma'] / (Lmax * par['L'])) * np.exp(LL[idx])
            
            # Add in new associative strength
            # c_vec is (D, C, L). We are at state Lmax-1 (0-based)
            c_current = c_vec[:, :, Lmax-1]
            Vnew = c_current * win[:, None] / np.sum(c_current**2, axis=1, keepdims=True)
            
            V[:, :, Lmax-1] = Vnew

            # Normalize
            lsb = tmp / np.sum(tmp)

    return V, lsb, Lmax, q

def Update(V, B, sigmasq, lsb, Lmax, c_vec, win, par):
    # Update variables V and cov

    # Number of iterations
    if par['t'] == par['ntrials'] or 'time' not in par:
        nIter = 1
    else:
        # par['t'] is 1-based. Current time is par['time'][par['t']-1], next is par['time'][par['t']]
        nIter = min(par['ls']['chi'], round(par['time'][par['t']] - par['time'][par['t']-1]))

    # Save current value
    sigmasq0 = sigmasq

    # Get expectations
    mu = np.reshape(np.sum(V * c_vec, axis=1), (par['D'], par['L'] * par['ls']['ncop']))

    # Loop through reward dimensions
    for d in range(par['D']):
        # Loop through active latent states
        for l1 in range(par['L']):
            for l2 in range(Lmax):
                # Index (0-based)
                l = l2 * par['L'] + l1

                # Weighted covariance (Outer product)
                c_outer = np.outer(c_vec[d, :, l], c_vec[d, :, l])
                B[:, :, d, l] = B[:, :, d, l] + par['ls']['alpha2'] * (lsb[l] * c_outer - B[:, :, d, l])

                # Associability (solve B * x = c)
                tmp = par['ls']['alpha0'] * lsb[l] * np.linalg.solve(B[:, :, d, l], c_vec[d, :, l])

                # Temporal difference
                TD = win[d] - mu[d, l]

                # Associative strength update
                V[d, :, l] = V[d, :, l] + tmp * TD

                # Part of variance update
                sigmasq = sigmasq + par['ls']['alpha1'] * lsb[l] * TD**2

                # Rumination steps
                for n in range(nIter - 1):
                    mu = np.reshape(np.sum(V * c_vec, axis=1), (par['D'], par['L'] * par['ls']['ncop']))
                    TD = win[d] - mu[d, l]
                    V[d, :, l] = V[d, :, l] + tmp * TD

    # Finish variance update
    sigmasq = sigmasq - par['ls']['alpha1'] * sigmasq0

    return V, B, sigmasq
