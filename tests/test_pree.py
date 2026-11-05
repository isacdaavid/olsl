"""Tests for tasks/pree.py.

Two layers:

  1. Unit tests of ``initialize`` and ``randomize`` — verify the trial
     schedule, reward encoding, cue encoding, shape contract, and case
     parsing. These are the interface guarantees that downstream code
     (run_task, LatentState) relies on.

  2. Integration test with ``run_task`` and a minimal fake agent — verify
     that the task plugs into the simulation loop without shape errors.
     This catches drift between the task's c_vec shape and the model's
     expectations.
"""

import numpy as np
import pytest

from olsl.tasks import pree
from olsl.simulation.run_task import run_task


# =============================================================================
# Helpers
# =============================================================================

def make_par(case=1, ncop=1, **overrides):
    """Minimal par dict for PREE, plus any user overrides."""
    par = {
        'case': case,
        'ls': {'ncop': ncop},
    }
    par.update(overrides)
    return par


# =============================================================================
# 1. initialize() — defaults and case structure
# =============================================================================

def test_initialize_sets_core_dimensions():
    par = pree.initialize(make_par())
    assert par['A'] == 1
    assert par['D'] == 1
    assert par['C'] == 1
    assert par['L'] == 1
    assert par['center'] == 0.5


def test_initialize_outcomes_layout():
    """Reward/cue encoding: row 0 = rewarded (1, 1), row 1 = unrewarded (0, 1)."""
    par = pree.initialize(make_par())
    np.testing.assert_array_equal(par['outcomes'], [[1.0, 1.0], [0.0, 1.0]])


def test_initialize_case_1_trialorder():
    """Case 1: 10 alternating pairs then 10 extinction trials."""
    par = pree.initialize(make_par(case=1))
    assert par['ntrials'] == 30
    assert len(par['trialorder']) == 30
    np.testing.assert_array_equal(par['trialorder'][:20], np.tile([1, 2], 10))
    np.testing.assert_array_equal(par['trialorder'][20:], np.full(10, 2))


def test_initialize_case_2_trialorder():
    """Case 2: 20 continuous-reward trials then 10 extinction trials."""
    par = pree.initialize(make_par(case=2))
    assert par['ntrials'] == 30
    np.testing.assert_array_equal(par['trialorder'][:20], np.tile([1, 1], 10))
    np.testing.assert_array_equal(par['trialorder'][20:], np.full(10, 2))


def test_initialize_case_3_trialorder():
    """Case 3: alternating -> rewarded -> extinction (50 trials)."""
    par = pree.initialize(make_par(case=3))
    assert par['ntrials'] == 50
    np.testing.assert_array_equal(par['trialorder'][:20], np.tile([1, 2], 10))
    np.testing.assert_array_equal(par['trialorder'][20:40], np.ones(20, dtype=int))
    np.testing.assert_array_equal(par['trialorder'][40:], np.full(10, 2))


def test_initialize_case_4_trialorder():
    """Case 4: continuous -> rewarded -> extinction (50 trials)."""
    par = pree.initialize(make_par(case=4))
    assert par['ntrials'] == 50
    np.testing.assert_array_equal(par['trialorder'][:20], np.tile([1, 1], 10))
    np.testing.assert_array_equal(par['trialorder'][20:40], np.ones(20, dtype=int))
    np.testing.assert_array_equal(par['trialorder'][40:], np.full(10, 2))


@pytest.mark.parametrize('case', [1, 2, 3, 4])
def test_initialize_trialorder_dtype_is_int(case):
    """trialorder must be integer — randomize does ``trial_type - 1`` on it."""
    par = pree.initialize(make_par(case=case))
    assert np.issubdtype(par['trialorder'].dtype, np.integer)


@pytest.mark.parametrize('case', [1, 2, 3, 4])
def test_initialize_ntrials_matches_trialorder(case):
    par = pree.initialize(make_par(case=case))
    assert par['ntrials'] == len(par['trialorder'])


# =============================================================================
# 2. initialize() — case parsing and error handling
# =============================================================================

@pytest.mark.parametrize('case_val', [
    1,                         # Python int
    np.int64(1),               # numpy scalar
    [1],                       # list
    np.array([1]),             # array
])
def test_initialize_accepts_scalar_or_sequence_case(case_val):
    """MATLAB's par.case(1) accepts scalar or sequence; mirror that."""
    par = pree.initialize(make_par(case=case_val))
    assert par['ntrials'] == 30


def test_initialize_unknown_case_raises():
    with pytest.raises(ValueError, match=r"Unknown PREE case"):
        pree.initialize(make_par(case=99))


def test_initialize_does_not_mutate_input():
    """initialize shallow-copies par, so the caller's dict is untouched."""
    original = make_par(case=1)
    pree.initialize(original)

    assert 'ntrials'    not in original
    assert 'trialorder' not in original
    assert 'A'          not in original
    assert 'outcomes'   not in original


def test_initialize_overrides_user_supplied_dimensions():
    """MATLAB's PREE always resets D, C, L, A — Python must do the same."""
    par = pree.initialize(make_par(case=1, D=5, C=3, L=2, A=4))
    assert par['D'] == 1
    assert par['C'] == 1
    assert par['L'] == 1
    assert par['A'] == 1


# =============================================================================
# 3. randomize() — shapes and content
# =============================================================================

def test_randomize_c_vec_shape():
    """c_vec must have shape (D, C, L*ncop, A)."""
    par = pree.initialize(make_par(case=1, ncop=3))
    par['t'] = 1
    world = pree.randomize(par)

    assert world['c_vec'].shape == (1, 1, 3, 1)


def test_randomize_win_is_python_float():
    par = pree.initialize(make_par())
    par['t'] = 1
    world = pree.randomize(par)
    assert isinstance(world['win'], float)


@pytest.mark.parametrize('case', [1, 2, 3, 4])
def test_randomize_cue_always_one(case):
    """Both PREE outcome rows have cue=1: the cue is always present,
    only the reward changes with trial type."""
    par = pree.initialize(make_par(case=case))
    for t in range(1, par['ntrials'] + 1):
        par['t'] = t
        world = pree.randomize(par)
        assert np.all(world['c_vec'] == 1.0), f"trial {t}: cue should be 1"


@pytest.mark.parametrize('case', [1, 2, 3, 4])
def test_randomize_reward_matches_trialorder(case):
    """Reward at trial t must match outcomes[trialorder[t-1], 0]."""
    par = pree.initialize(make_par(case=case))
    for t in range(1, par['ntrials'] + 1):
        par['t'] = t
        world = pree.randomize(par)
        expected_row = int(par['trialorder'][t - 1]) - 1
        assert world['win'] == par['outcomes'][expected_row, 0], \
            f"case={case}, trial={t}"


def test_randomize_case_1_reward_pattern():
    """Sanity: case 1 has 10 alternating [1,0] pairs, then 10 zeros."""
    par = pree.initialize(make_par(case=1))
    rewards = []
    for t in range(1, 31):
        par['t'] = t
        rewards.append(pree.randomize(par)['win'])

    np.testing.assert_array_equal(
        rewards,
        [1, 0] * 10 + [0] * 10,
    )


def test_randomize_case_2_reward_pattern():
    """Sanity: case 2 has 20 continuous rewards, then 10 zeros."""
    par = pree.initialize(make_par(case=2))
    rewards = []
    for t in range(1, 31):
        par['t'] = t
        rewards.append(pree.randomize(par)['win'])

    np.testing.assert_array_equal(rewards, [1] * 20 + [0] * 10)


def test_randomize_case_3_reward_pattern():
    """Sanity: case 3 has alternation, then 20 continuous rewards, then 10 zeros."""
    par = pree.initialize(make_par(case=3))
    rewards = []
    for t in range(1, 51):
        par['t'] = t
        rewards.append(pree.randomize(par)['win'])

    np.testing.assert_array_equal(
        rewards,
        [1, 0] * 10 + [1] * 20 + [0] * 10,
    )


def test_randomize_cues_identical_across_states():
    """All L*ncop slots in c_vec must hold the same cue."""
    par = pree.initialize(make_par(case=1, ncop=4))
    par['t'] = 1
    world = pree.randomize(par)

    for l in range(4):
        assert world['c_vec'][0, :, l, 0] == 1.0


# =============================================================================
# 4. Integration with run_task
# =============================================================================

def test_integration_with_run_task_minimal_agent():
    """The task's outputs must be consumable by run_task without shape errors."""
    def trivial_agent(par, agent, world):
        D, C = par['D'], par['C']
        L_total = par['L'] * par['ls']['ncop']
        A = par['A']
        if agent is None:
            return {
                'V':   np.zeros((D, C, L_total)),
                'lsb': np.ones(L_total) / L_total,
                'win': np.zeros(D),
                'mu':  np.zeros((D, A)),
            }
        a = dict(agent)
        a['V']   = np.ones((D, C, L_total)) * par['t']
        a['win'] = np.atleast_1d(world['win'])
        a['mu']  = np.array([[a['V'].sum()] * A] * D)
        return a

    par = make_par(case=1, ncop=2)
    r = run_task(par, pree, trivial_agent)

    assert r['par']['ntrials'] == 30
    # V has (ntrials+1) rows and D*C*L*ncop columns
    assert r['V'].shape == (31, 1 * 1 * 1 * 2)
    # MU has (ntrials) rows and D*A columns
    assert r['mu'].shape == (30, 1 * 1)
    # No NaNs in logged arrays
    assert not np.any(np.isnan(r['mu']))


def test_integration_all_cases():
    """Every case must run through run_task end-to-end."""
    def trivial_agent(par, agent, world):
        D, C = par['D'], par['C']
        L_total = par['L'] * par['ls']['ncop']
        A = par['A']
        if agent is None:
            return {
                'V':   np.zeros((D, C, L_total)),
                'lsb': np.ones(L_total) / L_total,
                'win': np.zeros(D),
                'mu':  np.zeros((D, A)),
            }
        a = dict(agent)
        a['V']   = np.ones((D, C, L_total)) * par['t']
        a['win'] = np.atleast_1d(world['win'])
        a['mu']  = np.array([[a['V'].sum()] * A] * D)
        return a

    for case in (1, 2, 3, 4):
        par = make_par(case=case, ncop=2)
        r = run_task(par, pree, trivial_agent)
        assert r['par']['ntrials'] in (30, 50), f"case={case}"
        # trialorder length must match ntrials
        assert len(r['par']['trialorder']) == r['par']['ntrials']
