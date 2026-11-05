"""Tests for simulation/run_task.py.

Tests are isolated from the real LatentState / MMKalmanFilter by using
lightweight fakes. The goal is to pin down run_task's contract:

  - Output array shapes and initial rows
  - Per-trial logging (V, lsb, mu, x_t, MMKF_*)
  - par['t'] is 1-based; par['lastwin'] follows MATLAB's short-circuit
  - Fortran-order flattening of agent['V']
  - Validation of agent['mu'] size
  - Graceful None for absent agent fields / absent observer
"""

import numpy as np
import pytest

from olsl.simulation.run_task import run_task


# =============================================================================
# Fakes
# =============================================================================

class FakeTask:
    """Minimal task module: initialize(par) and randomize(par)."""

    def __init__(self, rewards):
        self.rewards = np.asarray(rewards, dtype=float)

    def initialize(self, par):
        par = dict(par)
        par['ntrials'] = len(self.rewards)
        return par

    def randomize(self, par):
        D = par['D']
        t_idx = par['t'] - 1   # par['t'] is 1-based
        return {
            'win':   np.full(D, self.rewards[t_idx]),
            'c_vec': np.ones((D, par['C'], 1, par['A'])),
        }


def make_agent_fn(D=1, C=1, n_states=1, with_lsb=True, with_mu=True,
                  with_x_t=False, V_init=None):
    """Return a callable (par, agent, world) -> agent, mimicking LatentState's
    init/update dispatch. The 'update' is a trivial RW-style rule so tests
    can predict V[t+1] without depending on LatentState."""
    if V_init is None:
        V_init = np.zeros((D, C, n_states))

    def agent_fn(par, agent, world):
        if agent is None:
            a = {'V': V_init.copy(), 'win': np.zeros(D)}
            if with_lsb:
                a['lsb'] = np.ones(n_states) / n_states
            if with_mu:
                a['mu'] = np.zeros((D, par['A']))
            return a

        a = dict(agent)
        reward = np.atleast_1d(world['win'])
        V = a['V'].copy()
        V += 0.1 * (reward[0] - V.sum())
        a['V'] = V
        a['win'] = reward
        if with_mu:
            a['mu'] = np.array([[V.sum()] * par['A']] * D)
        if with_x_t:
            a['x_t'] = int(par['t']) % 2
        return a

    return agent_fn


def make_observer_fn(L_kf=3):
    """Return a callable (par, mmkf, agent, world) -> mmkf, mimicking
    MMKalmanFilter's init/update dispatch. Uses values that depend on
    par['t'] so tests can verify per-trial logging."""

    def observer_fn(par, mmkf, agent, world):
        if mmkf is None:
            return {
                'lsb_prior':   np.full(L_kf, 1.0 / L_kf),
                'likelihoods': np.ones(L_kf),
                'lsb':         np.full(L_kf, 1.0 / L_kf),
                'X_t_est':     0,
            }
        mm = dict(mmkf)
        mm['lsb_prior']   = np.roll(mmkf['lsb'], 1)
        mm['likelihoods'] = np.full(L_kf, float(par['t']))   # 1, 2, 3, ...
        s = mm['likelihoods'].sum()
        mm['lsb'] = mm['likelihoods'] / s
        mm['X_t_est'] = int(np.argmax(mm['lsb']))
        return mm

    return observer_fn


@pytest.fixture
def base_par():
    return {'D': 1, 'C': 1, 'A': 1, 'center': 0.0, 'ntrials': 1}


# =============================================================================
# 1. Contract: shapes, initial rows, par side effects
# =============================================================================

def test_output_keys_no_observer(base_par):
    task = FakeTask([1.0, 0.0])
    agent_fn = make_agent_fn()
    r = run_task(base_par, task, agent_fn)

    for key in ('mu', 'V', 'lsb', 'X_t_seq',
                'MMKF_lsb_prior', 'MMKF_likelihood',
                'MMKF_lsb_post', 'MMKF_X_t_est',
                'mmkf', 'agent', 'par'):
        assert key in r, f"missing output key: {key}"


def test_shapes_no_observer(base_par):
    task = FakeTask([1.0, 0.0, 1.0])
    agent_fn = make_agent_fn(n_states=2)
    r = run_task(base_par, task, agent_fn)

    ntrials = 3
    assert r['V'].shape == (ntrials + 1, 1 * 1 * 2)   # (4, 2)
    assert r['lsb'].shape == (ntrials + 1, 2)          # (4, 2)
    assert r['mu'].shape == (ntrials, 1 * 1)           # (3, 1)
    assert r['X_t_seq'].shape == (ntrials,)


def test_initial_v_row(base_par):
    """V[0] must equal the flattened initial agent['V'] (order='F')."""
    V_init = np.array([[[0.1, 0.2, 0.3]]])   # (D=1, C=1, L=3)
    agent_fn = make_agent_fn(n_states=3, V_init=V_init)
    task = FakeTask([1.0])
    r = run_task(base_par, task, agent_fn)

    np.testing.assert_array_equal(r['V'][0, :], [0.1, 0.2, 0.3])


def test_par_t_is_one_based(base_par):
    seen = []

    def tracking_agent(par, agent, world):
        if agent is not None:
            seen.append(par['t'])
        return make_agent_fn()(par, agent, world)

    task = FakeTask([1.0, 1.0, 1.0])
    run_task(base_par, task, tracking_agent)
    assert seen == [1, 2, 3]


def test_par_ntrials_set_by_task_init(base_par):
    """Task.initialize may change par['ntrials']; run_task must honor it."""
    task = FakeTask([1.0, 0.0, 1.0, 0.0])   # task will set ntrials=4
    agent_fn = make_agent_fn()
    r = run_task(base_par, task, agent_fn)
    assert r['par']['ntrials'] == 4
    assert r['V'].shape[0] == 5


def test_final_par_t_equals_ntrials(base_par):
    task = FakeTask([1.0, 1.0, 1.0])
    agent_fn = make_agent_fn()
    r = run_task(base_par, task, agent_fn)
    assert r['par']['t'] == 3


# =============================================================================
# 2. Tracking: each log matches the agent/observer state after each trial
# =============================================================================

def test_v_tracking_matches_agent(base_par):
    """V[t+1] must be the flattened agent['V'] after trial t."""
    task = FakeTask([1.0, 0.0, 1.0])
    agent_fn = make_agent_fn(n_states=2)
    r = run_task(base_par, task, agent_fn)

    # Replay the fake agent to get its per-trial V and compare.
    par = task.initialize(dict(base_par))

    # Init is separate from the trial loop — mirrors run_task's own
    # `agent = approach(par, None, None)` before `for t in range(...)`.
    agent = agent_fn(par, None, None)

    for t in range(par['ntrials']):
        par['t'] = t + 1
        if t == 0:
            par['lastwin'] = False
        else:
            par['lastwin'] = bool(not np.all(np.atleast_1d(agent['win']) == 0))
        world = task.randomize(par)
        agent = agent_fn(par, agent, world)
        np.testing.assert_allclose(
            r['V'][t + 1, :],
            agent['V'].flatten(order='F'),
        )

    np.testing.assert_allclose(r['V'][1, :], [0.1, 0.1])
    np.testing.assert_allclose(r['V'][2, :], [0.08, 0.08])
    np.testing.assert_allclose(r['V'][3, :], [0.164, 0.164])


def test_fortran_flatten_order(base_par):
    """V[1,:] must be the *column-major* flatten of agent['V'], matching
    MATLAB's A(:). For a (2,2,1) V this is [1,3,2,4], not [1,2,3,4]."""
    D, C = 2, 2

    def agent_fn(par, agent, world):
        if agent is None:
            return {'V': np.zeros((D, C, 1)), 'win': np.zeros(D)}
        a = dict(agent)
        a['V'] = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])   # (2,2,1)
        return a

    base_par['D'] = D
    base_par['C'] = C
    task = FakeTask([1.0])
    r = run_task(base_par, task, agent_fn)

    expected_col_major = np.array([1.0, 3.0, 2.0, 4.0])
    expected_row_major = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(r['V'][1, :], expected_col_major)
    # Regression guard: must NOT match row-major
    assert not np.array_equal(r['V'][1, :], expected_row_major)


def test_lsb_tracking(base_par):
    """When the agent has 'lsb', lsb[t+1] must equal it."""
    task = FakeTask([1.0, 0.0])
    agent_fn = make_agent_fn(n_states=3)
    r = run_task(base_par, task, agent_fn)

    # Fake agent never changes lsb, so every row after init equals init.
    np.testing.assert_allclose(r['lsb'][0, :], [1 / 3, 1 / 3, 1 / 3])
    for t in range(1, 3):
        np.testing.assert_allclose(r['lsb'][t, :], [1 / 3, 1 / 3, 1 / 3])


def test_lsb_none_when_agent_lacks_lsb(base_par):
    task = FakeTask([1.0])
    agent_fn = make_agent_fn(with_lsb=False)
    r = run_task(base_par, task, agent_fn)
    assert r['lsb'] is None


def test_mu_tracking(base_par):
    task = FakeTask([1.0, 0.0, 1.0])
    agent_fn = make_agent_fn()
    r = run_task(base_par, task, agent_fn)

    # The fake agent sets mu = [[V.sum()]] each trial; V.sum() grows
    # monotonically here, but we just check shape + no NaN to avoid
    # depending on the fake's internals.
    assert r['mu'].shape == (3, 1)
    assert not np.any(np.isnan(r['mu']))


def test_x_t_seq_nan_when_agent_lacks_x_t(base_par):
    task = FakeTask([1.0, 1.0])
    agent_fn = make_agent_fn(with_x_t=False)
    r = run_task(base_par, task, agent_fn)
    assert np.all(np.isnan(r['X_t_seq']))


def test_x_t_seq_filled_when_agent_has_x_t(base_par):
    task = FakeTask([1.0, 1.0, 1.0, 1.0])
    agent_fn = make_agent_fn(with_x_t=True)
    r = run_task(base_par, task, agent_fn)
    # Fake agent sets x_t = par['t'] % 2 -> [1, 0, 1, 0]
    np.testing.assert_array_equal(r['X_t_seq'], [1, 0, 1, 0])


# =============================================================================
# 3. lastwin logic (MATLAB's `~(t==1 || agent.win==0)`)
# =============================================================================

def test_lastwin_logic(base_par):
    """lastwin must be:
        trial 1            -> False  (short-circuit on t==1)
        trial t>1, prev=0  -> False  (agent.win==0 is True)
        trial t>1, prev!=0 -> True
    """
    lastwins = []

    def tracking_agent(par, agent, world):
        if agent is not None:
            lastwins.append(bool(par['lastwin']))
        return make_agent_fn()(par, agent, world)

    # Rewards: trial 1 = 0 (agent stores win=0 after), trial 2 = 1, trial 3 = 1
    task = FakeTask([0.0, 1.0, 1.0])
    run_task(base_par, task, tracking_agent)

    # Trial 1: t==0 short-circuit -> False
    # Trial 2: agent['win'] from trial 1 was [0] -> False
    # Trial 3: agent['win'] from trial 2 was [1] -> True
    assert lastwins == [False, False, True]


# =============================================================================
# 4. Observer (MMKF) integration
# =============================================================================

def test_mmkf_none_without_observer(base_par):
    task = FakeTask([1.0])
    agent_fn = make_agent_fn()
    r = run_task(base_par, task, agent_fn)
    for key in ('MMKF_lsb_prior', 'MMKF_likelihood',
                'MMKF_lsb_post', 'MMKF_X_t_est', 'mmkf'):
        assert r[key] is None


def test_mmkf_shapes(base_par):
    base_par['kf'] = {'L': 4}
    task = FakeTask([1.0, 0.0, 1.0])
    agent_fn = make_agent_fn()
    observer_fn = make_observer_fn(L_kf=4)
    r = run_task(base_par, task, agent_fn, observer=observer_fn)

    ntrials, L_kf = 3, 4
    assert r['MMKF_lsb_prior'].shape  == (ntrials, L_kf)
    assert r['MMKF_likelihood'].shape == (ntrials, L_kf)
    assert r['MMKF_lsb_post'].shape   == (ntrials, L_kf)
    assert r['MMKF_X_t_est'].shape    == (ntrials,)
    # Final mmkf struct is returned
    assert r['mmkf'] is not None
    assert 'lsb_prior' in r['mmkf']


def test_mmkf_logged_per_trial(base_par):
    """Each MMKF_* row must match what the observer returned that trial."""
    base_par['kf'] = {'L': 3}
    task = FakeTask([1.0, 0.0, 1.0])
    agent_fn = make_agent_fn()
    observer_fn = make_observer_fn(L_kf=3)

    # Side channel: record observer outputs
    recorded = []

    def tracking_observer(par, mmkf, agent, world):
        mm = observer_fn(par, mmkf, agent, world)
        recorded.append({
            'lsb_prior':   mm['lsb_prior'].copy(),
            'likelihoods': mm['likelihoods'].copy(),
            'lsb':         mm['lsb'].copy(),
            'X_t_est':     mm['X_t_est'],
        })
        return mm

    r = run_task(base_par, task, agent_fn, observer=tracking_observer)

    for t in range(3):
        np.testing.assert_array_equal(r['MMKF_lsb_prior'][t, :],  recorded[t]['lsb_prior'])
        np.testing.assert_array_equal(r['MMKF_likelihood'][t, :], recorded[t]['likelihoods'])
        np.testing.assert_array_equal(r['MMKF_lsb_post'][t, :],   recorded[t]['lsb'])
        assert r['MMKF_X_t_est'][t] == recorded[t]['X_t_est']


def test_mmkf_likelihoods_use_one_based_t(base_par):
    """The fake observer sets likelihoods = par['t'], so this pins down
    that par['t'] is 1-based (1, 2, 3, ...) and not 0, 1, 2."""
    base_par['kf'] = {'L': 2}
    task = FakeTask([1.0, 1.0, 1.0])
    agent_fn = make_agent_fn()
    observer_fn = make_observer_fn(L_kf=2)
    r = run_task(base_par, task, agent_fn, observer=observer_fn)

    for t in range(3):
        np.testing.assert_array_equal(
            r['MMKF_likelihood'][t, :], [t + 1, t + 1]
        )


# =============================================================================
# 5. Validation
# =============================================================================

def test_mu_size_mismatch_raises(base_par):
    """If agent['mu'] doesn't have D*A elements, run_task must raise."""
    def bad_agent(par, agent, world):
        if agent is None:
            return {'V': np.zeros((1, 1, 1)),
                    'win': np.zeros(1),
                    'mu': np.zeros(1)}
        a = dict(agent)
        a['mu'] = np.zeros(5)   # wrong: D*A == 1
        return a

    task = FakeTask([1.0])
    with pytest.raises(ValueError, match=r"agent\['mu'\]"):
        run_task(base_par, task, bad_agent)
