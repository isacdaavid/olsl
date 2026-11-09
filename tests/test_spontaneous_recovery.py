"""Tests for tasks/spontaneous_recovery.py.

The distinguishing feature of this task (vs. Renewal) is that there's
no explicit context shift — recovery is driven purely by the temporal
gap encoded in ``par['time']``. So beyond the standard shape/reward
tests, this file focuses on the structure of ``par['time']``:

  - Length is ntrials + 1 (indices 0..ntrials)
  - Pre-shift values are 1..40
  - Post-shift values are shifted by (case - 1)
  - The gap at the extinction→renewal boundary equals ``case``
"""

import numpy as np
import pytest

from olsl.tasks import spontaneous_recovery
from olsl.simulation.run_task import run_task


def make_par(case=1, ncop=1, **overrides):
    par = {'case': case, 'ls': {'ncop': ncop}}
    par.update(overrides)
    return par


# =============================================================================
# 1. initialize() — dimensions, outcomes, trialorder
# =============================================================================

def test_initialize_sets_core_dimensions():
    par = spontaneous_recovery.initialize(make_par())
    assert par['A'] == 1
    assert par['D'] == 1
    assert par['C'] == 1
    assert par['L'] == 1
    assert par['ntrials'] == 50
    assert par['center'] == 0.5
    assert par['r0'] == 1.0


def test_initialize_outcomes_layout():
    par = spontaneous_recovery.initialize(make_par())
    np.testing.assert_array_equal(
        par['outcomes'],
        [[1.0, 1.0], [0.0, 1.0]],
    )


def test_initialize_trialorder():
    par = spontaneous_recovery.initialize(make_par())
    np.testing.assert_array_equal(par['trialorder'][:20], np.full(20, 1))
    np.testing.assert_array_equal(par['trialorder'][20:40], np.full(20, 2))
    np.testing.assert_array_equal(par['trialorder'][40:], np.full(10, 1))


def test_initialize_trialorder_dtype():
    par = spontaneous_recovery.initialize(make_par())
    assert np.issubdtype(par['trialorder'].dtype, np.integer)


def test_initialize_does_not_mutate_input():
    original = make_par(case=1)
    spontaneous_recovery.initialize(original)
    for key in ('A', 'D', 'C', 'L', 'center', 'r0', 'ntrials',
                'trialorder', 'outcomes', 'time'):
        assert key not in original


# =============================================================================
# 2. initialize() — time array structure
# =============================================================================

def test_time_array_length():
    par = spontaneous_recovery.initialize(make_par())
    assert len(par['time']) == par['ntrials'] + 1


def test_time_array_case_1_no_shift():
    """case=1 means no gap: times are 1, 2, ..., 51."""
    par = spontaneous_recovery.initialize(make_par(case=1))
    np.testing.assert_array_equal(
        par['time'], np.arange(1, 52, dtype=float),
    )


def test_time_array_case_2_shifts_renewal():
    """case=2: renewal phase (indices 40..50) shifted by 1."""
    par = spontaneous_recovery.initialize(make_par(case=2))
    expected = np.arange(1, 52, dtype=float)
    expected[40:] += 1
    np.testing.assert_array_equal(par['time'], expected)


@pytest.mark.parametrize('case', [1, 2, 3, 5, 10])
def test_time_array_case_k(case):
    """Shift magnitude scales linearly with case - 1."""
    par = spontaneous_recovery.initialize(make_par(case=case))
    expected = np.arange(1, 52, dtype=float)
    expected[40:] += (case - 1)
    np.testing.assert_array_equal(par['time'], expected, err_msg=f"case={case}")


@pytest.mark.parametrize('case', [1, 2, 5, 10])
def test_time_array_monotone(case):
    par = spontaneous_recovery.initialize(make_par(case=case))
    assert np.all(np.diff(par['time']) > 0), f"case={case}"


@pytest.mark.parametrize('case', [1, 2, 3, 10])
def test_gap_at_extinction_renewal_boundary(case):
    """The gap between last extinction (trial 40) and first renewal
    (trial 41) equals ``case``. This is what drives the recovery signal:
    LatentState.Update uses this gap to set nIter (rumination count),
    and LatentState.ContextShift uses it to set prob_stay (belief decay).
    """
    par = spontaneous_recovery.initialize(make_par(case=case))
    # 1-based trial 41 → 0-based index 40; trial 40 → index 39
    gap = par['time'][40] - par['time'][39]
    assert gap == case, f"case={case}: gap={gap}, expected {case}"


@pytest.mark.parametrize('case', [1, 2, 3, 10])
def test_gap_within_renewal_is_unity(case):
    """Once inside the renewal phase, consecutive trials are 1 apart
    (the shift only applies at the boundary)."""
    par = spontaneous_recovery.initialize(make_par(case=case))
    for i in range(41, len(par['time'])):
        assert par['time'][i] - par['time'][i - 1] == 1, (
            f"case={case}: gap at index {i} should be 1"
        )


# =============================================================================
# 3. initialize() — case validation
# =============================================================================

def test_initialize_accepts_scalar_or_sequence_case():
    for case_val in (1, np.int64(1), [1], np.array([1])):
        par = spontaneous_recovery.initialize(make_par(case=case_val))
        assert par['ntrials'] == 50


def test_initialize_case_large_accepted():
    par = spontaneous_recovery.initialize(make_par(case=100))
    assert par['ntrials'] == 50


@pytest.mark.parametrize('bad_case', [0, -1, -100])
def test_initialize_invalid_case_raises(bad_case):
    with pytest.raises(ValueError, match=r"SpontaneousRecovery case"):
        spontaneous_recovery.initialize(make_par(case=bad_case))


# =============================================================================
# 4. randomize()
# =============================================================================

def test_randomize_c_vec_shape():
    par = spontaneous_recovery.initialize(make_par(case=1, ncop=2))
    par['t'] = 1
    world = spontaneous_recovery.randomize(par)
    assert world['c_vec'].shape == (1, 1, 2, 1)


def test_randomize_win_is_python_float():
    par = spontaneous_recovery.initialize(make_par())
    par['t'] = 1
    world = spontaneous_recovery.randomize(par)
    assert isinstance(world['win'], float)


@pytest.mark.parametrize('case', [1, 2, 3])
def test_randomize_reward_pattern(case):
    """Reward=1 during acquisition and renewal, 0 during extinction.
    Case value only affects timing, not the reward schedule."""
    par = spontaneous_recovery.initialize(make_par(case=case))
    rewards = []
    for t in range(1, par['ntrials'] + 1):
        par['t'] = t
        rewards.append(spontaneous_recovery.randomize(par)['win'])
    np.testing.assert_array_equal(
        rewards,
        [1.0] * 20 + [0.0] * 20 + [1.0] * 10,
        err_msg=f"case={case}",
    )


def test_randomize_cue_always_one():
    """Cue is on-screen on every trial (both outcome rows have cue=1)."""
    par = spontaneous_recovery.initialize(make_par(case=1))
    for t in (1, 20, 21, 40, 41, 50):
        par['t'] = t
        cue = spontaneous_recovery.randomize(par)['c_vec'][0, :, 0, 0]
        np.testing.assert_array_equal(cue, [1.0], err_msg=f"trial {t}")


def test_randomize_cue_identical_across_states():
    par = spontaneous_recovery.initialize(make_par(case=1, ncop=3))
    par['t'] = 1
    world = spontaneous_recovery.randomize(par)
    for l in range(3):
        np.testing.assert_array_equal(
            world['c_vec'][0, :, l, 0],
            world['c_vec'][0, :, 0, 0],
        )


# =============================================================================
# 5. Integration with run_task
# =============================================================================

def _trivial_agent(par, agent, world):
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


@pytest.mark.parametrize('case', [1, 2, 3])
def test_integration_with_run_task(case):
    par = make_par(case=case, ncop=2)
    r = run_task(par, spontaneous_recovery, _trivial_agent)
    assert r['par']['ntrials'] == 50
    assert r['V'].shape == (51, 1 * 1 * 1 * 2)
    assert r['mu'].shape == (50, 1)
    assert not np.any(np.isnan(r['mu']))


def test_integration_time_survives_round_trip():
    """par['time'] must still be present and unmodified after run_task."""
    par = make_par(case=3, ncop=1)
    r = run_task(par, spontaneous_recovery, _trivial_agent)

    assert 'time' in r['par']
    assert len(r['par']['time']) == 51
    # Gap of 3 at the extinction → renewal boundary should be preserved
    assert r['par']['time'][40] - r['par']['time'][39] == 3


def test_integration_par_t_ends_at_ntrials():
    par = make_par(case=1, ncop=1)
    r = run_task(par, spontaneous_recovery, _trivial_agent)
    assert r['par']['t'] == 50
