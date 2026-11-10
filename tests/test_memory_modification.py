"""Tests for tasks/memory_modification.py.

The interesting structure in this task is `par['time']`: it encodes
four phases separated by gaps of 20, `case`, and 200 units. Tests
verify:

  - The exact time-array values for representative cases.
  - The gap at each phase boundary.
  - The reward pattern (only Phase 1 is reinforced).
  - That `case` is honored, validated, and doesn't leak between calls.
"""

import numpy as np
import pytest

from olsl.tasks import memory_modification as mm
from olsl.simulation.run_task import run_task


def make_par(case=5, ncop=1, **overrides):
    par = {'case': case, 'ls': {'ncop': ncop}}
    par.update(overrides)
    return par


# =============================================================================
# 1. Dimensions and parameter metadata
# =============================================================================

def test_core_dimensions():
    par = mm.initialize(make_par())
    assert par['A'] == 1
    assert par['D'] == 1
    assert par['C'] == 1
    assert par['L'] == 1
    assert par['center'] == 0.5
    assert par['ruminate'] == 3


def test_ntrials_is_24():
    """sum([3, 1, 19, 2]) - 1 = 24 — the last time entry is the boundary."""
    par = mm.initialize(make_par())
    assert par['ntrials'] == 24


# =============================================================================
# 2. Time array — exact values for representative cases
# =============================================================================

def test_time_array_length_is_25():
    """3+1+19+2 = 25 time entries; one per trial slot + a boundary."""
    for case in (0, 1, 5, 100):
        par = mm.initialize(make_par(case=case))
        assert len(par['time']) == 25, f"case={case}"


def test_time_array_case_5_explicit():
    """Hand-computed expected values for case=5."""
    par = mm.initialize(make_par(case=5))
    expected = np.array(
        [1.0, 2.0, 3.0, 24.0]
        + [float(t) for t in range(30, 49)]       # extinction 30..48
        + [249.0, 250.0]
    )
    np.testing.assert_array_equal(par['time'], expected)


def test_time_array_case_1_explicit():
    """Hand-computed expected values for case=1."""
    par = mm.initialize(make_par(case=1))
    expected = np.array(
        [1.0, 2.0, 3.0, 24.0]
        + [float(t) for t in range(26, 45)]       # extinction 26..44
        + [245.0, 246.0]
    )
    np.testing.assert_array_equal(par['time'], expected)


def test_time_array_case_0_explicit():
    """Case=0 → extinction starts one unit after retrieval."""
    par = mm.initialize(make_par(case=0))
    expected = np.array(
        [1.0, 2.0, 3.0, 24.0]
        + [float(t) for t in range(25, 44)]       # extinction 25..43
        + [244.0, 245.0]
    )
    np.testing.assert_array_equal(par['time'], expected)


@pytest.mark.parametrize('case', [0, 1, 5, 20, 100])
def test_time_array_monotone(case):
    par = mm.initialize(make_par(case=case))
    assert np.all(np.diff(par['time']) > 0), f"case={case}"


def test_time_array_dtype_is_float():
    par = mm.initialize(make_par())
    assert par['time'].dtype == np.float64


# =============================================================================
# 3. Phase boundary gaps
# =============================================================================

def test_gap_before_retrieval():
    """Gap between last acquisition trial (t=3) and retrieval (t=24) = 21."""
    for case in (0, 1, 5, 100):
        par = mm.initialize(make_par(case=case))
        gap = par['time'][3] - par['time'][2]
        assert gap == 21, f"case={case}"


@pytest.mark.parametrize('case', [0, 1, 5, 20, 100])
def test_gap_after_retrieval(case):
    """Gap after retrieval (index 3 → index 4) = case + 1."""
    par = mm.initialize(make_par(case=case))
    gap = par['time'][4] - par['time'][3]
    assert gap == case + 1, f"case={case}: {gap} != {case + 1}"


def test_gap_before_test():
    """Gap between last extinction (index 22) and first test (index 23) = 201."""
    for case in (0, 1, 5, 100):
        par = mm.initialize(make_par(case=case))
        gap = par['time'][23] - par['time'][22]
        assert gap == 201, f"case={case}"


def test_gap_between_retrieval_and_extinction_scales_with_case():
    """Sanity: doubling case doubles the retrieval→extinction gap."""
    par5  = mm.initialize(make_par(case=5))
    par10 = mm.initialize(make_par(case=10))
    gap5  = par5['time'][4] - par5['time'][3]
    gap10 = par10['time'][4] - par10['time'][3]
    assert gap10 == 2 * gap5 - 1   # (10+1) = 2*(5+1) - 1


# =============================================================================
# 4. Case handling
# =============================================================================

def test_case_accepts_list():
    par = mm.initialize({'case': [5], 'ls': {'ncop': 1}})
    assert par['ntrials'] == 24


def test_case_accepts_numpy_scalar():
    par = mm.initialize({'case': np.int64(5), 'ls': {'ncop': 1}})
    assert par['ntrials'] == 24


@pytest.mark.parametrize('bad_case', [-1, -100])
def test_negative_case_raises(bad_case):
    with pytest.raises(ValueError, match=r"MemoryModification case"):
        mm.initialize(make_par(case=bad_case))


def test_initialize_does_not_mutate_input():
    original = make_par(case=5)
    mm.initialize(original)
    for key in ('A', 'D', 'C', 'L', 'center', 'ruminate',
                'time', 'ntrials'):
        assert key not in original


def test_different_cases_produce_different_time_arrays():
    par1   = mm.initialize(make_par(case=1))
    par100 = mm.initialize(make_par(case=100))
    assert not np.array_equal(par1['time'], par100['time'])


# =============================================================================
# 5. randomize() — shape, reward pattern, cues
# =============================================================================

def test_randomize_c_vec_shape():
    par = mm.initialize(make_par(case=5, ncop=2))
    par['t'] = 1
    world = mm.randomize(par)
    assert world['c_vec'].shape == (1, 1, 2, 1)   # D, C, L*ncop, A


def test_randomize_win_is_float():
    par = mm.initialize(make_par())
    par['t'] = 1
    world = mm.randomize(par)
    assert isinstance(world['win'], float)


@pytest.mark.parametrize('case', [0, 1, 5, 100])
def test_reward_pattern(case):
    """Trials 1–3 are rewarded; trials 4+ (retrieval, extinction,
    test) are not."""
    par = mm.initialize(make_par(case=case))
    rewards = []
    for t in range(1, par['ntrials'] + 1):
        par['t'] = t
        rewards.append(mm.randomize(par)['win'])

    expected = [1.0, 1.0, 1.0] + [0.0] * (par['ntrials'] - 3)
    np.testing.assert_array_equal(rewards, expected, err_msg=f"case={case}")


def test_c_vec_all_ones():
    """The CS is present on every trial (single cue)."""
    par = mm.initialize(make_par(case=5))
    for t in (1, 3, 4, 24):
        par['t'] = t
        cue = mm.randomize(par)['c_vec'][0, :, 0, 0]
        np.testing.assert_array_equal(cue, [1.0], err_msg=f"trial {t}")


def test_c_vec_identical_across_states():
    par = mm.initialize(make_par(case=5, ncop=3))
    par['t'] = 1
    world = mm.randomize(par)
    for l in range(3):
        np.testing.assert_array_equal(
            world['c_vec'][0, :, l, 0],
            world['c_vec'][0, :, 0, 0],
        )


# =============================================================================
# 6. Integration with run_task
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


@pytest.mark.parametrize('case', [1, 5, 100])
def test_integration_with_run_task(case):
    par = make_par(case=case, ncop=2)
    r = run_task(par, mm, _trivial_agent)
    assert r['par']['ntrials'] == 24
    assert r['V'].shape == (25, 1 * 1 * 1 * 2)    # ntrials+1 rows, D*C*L*ncop cols
    assert r['mu'].shape == (24, 1)


def test_integration_time_survives_round_trip():
    """par['time'] must be present and unchanged after run_task."""
    par = make_par(case=5, ncop=1)
    r = run_task(par, mm, _trivial_agent)
    assert 'time' in r['par']
    assert len(r['par']['time']) == 25
    # Retrieval → extinction gap must still be case+1 = 6
    assert r['par']['time'][4] - r['par']['time'][3] == 6


def test_integration_par_t_ends_at_ntrials():
    par = make_par(case=5, ncop=1)
    r = run_task(par, mm, _trivial_agent)
    assert r['par']['t'] == 24
