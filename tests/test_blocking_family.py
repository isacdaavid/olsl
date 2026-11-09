"""Tests for the four blocking-family task modules.

All four tasks share the same outcomes matrix and randomize logic
(tested in test_blocking_common.py); only the trialorder differs. So
this file is:

  - Parametrized over the four tasks for properties that hold for all
    of them (ntrials, dtype, no-mutation, run_task integration).
  - Task-specific for trialorder patterns, since that is the sole
    thing that distinguishes each task.
"""

import numpy as np
import pytest

from olsl.tasks import (
    blocking, backwards_blocking, overexpectation, conditioned_inhibition,
)
from olsl.simulation.run_task import run_task


# (task_module, expected_ntrials) for parametrized tests
ALL_TASKS = [
    (blocking,               36),
    (backwards_blocking,     28),
    (overexpectation,       110),
    (conditioned_inhibition, 200),
]


# =============================================================================
# 1. Common properties (parametrized)
# =============================================================================

@pytest.mark.parametrize('task,ntrials', ALL_TASKS,
                         ids=[t.__name__.split('.')[-1] for t, _ in ALL_TASKS])
def test_ntrials(task, ntrials):
    par = task.initialize({'ls': {'ncop': 1}, 'modelname': 'RescorlaWagner'})
    assert par['ntrials'] == ntrials


@pytest.mark.parametrize('task,ntrials', ALL_TASKS,
                         ids=[t.__name__.split('.')[-1] for t, _ in ALL_TASKS])
def test_trialorder_length_matches_ntrials(task, ntrials):
    par = task.initialize({'ls': {'ncop': 1}})
    assert len(par['trialorder']) == ntrials


@pytest.mark.parametrize('task,_', ALL_TASKS,
                         ids=[t.__name__.split('.')[-1] for t, _ in ALL_TASKS])
def test_trialorder_dtype_is_int(task, _):
    par = task.initialize({'ls': {'ncop': 1}})
    assert np.issubdtype(par['trialorder'].dtype, np.integer)


@pytest.mark.parametrize('task,_', ALL_TASKS,
                         ids=[t.__name__.split('.')[-1] for t, _ in ALL_TASKS])
def test_core_dimensions_nonlatent(task, _):
    par = task.initialize({'ls': {'ncop': 1},
                           'modelname': 'RescorlaWagner'})
    assert par['A'] == 1
    assert par['D'] == 1
    assert par['L'] == 1
    assert par['center'] == 0.5
    assert par['C'] == 4


@pytest.mark.parametrize('task,_', ALL_TASKS,
                         ids=[t.__name__.split('.')[-1] for t, _ in ALL_TASKS])
def test_core_dimensions_latent(task, _):
    par = task.initialize({'ls': {'ncop': 1},
                           'modelname': 'LatentState'})
    assert par['C'] == 6


@pytest.mark.parametrize('task,_', ALL_TASKS,
                         ids=[t.__name__.split('.')[-1] for t, _ in ALL_TASKS])
def test_initialize_does_not_mutate_input(task, _):
    original = {'ls': {'ncop': 1}, 'modelname': 'RescorlaWagner'}
    task.initialize(original)
    for key in ('A', 'D', 'C', 'L', 'center', 'ntrials',
                'nphases', 'outcomes', 'trialorder'):
        assert key not in original, f"{key} leaked into input par"


@pytest.mark.parametrize('task,_', ALL_TASKS,
                         ids=[t.__name__.split('.')[-1] for t, _ in ALL_TASKS])
def test_randomize_returns_float_win(task, _):
    par = task.initialize({'ls': {'ncop': 1}})
    par['t'] = 1
    world = task.randomize(par)
    assert isinstance(world['win'], float)
    assert 'c_vec' in world


@pytest.mark.parametrize('task,_', ALL_TASKS,
                         ids=[t.__name__.split('.')[-1] for t, _ in ALL_TASKS])
def test_randomize_shape(task, _):
    par = task.initialize({'ls': {'ncop': 3},
                           'modelname': 'RescorlaWagner'})
    par['t'] = 1
    world = task.randomize(par)
    assert world['c_vec'].shape == (1, 4, 3, 1)   # D, C, L*ncop, A


# =============================================================================
# 2. Task-specific trialorder patterns
# =============================================================================

def test_blocking_trialorder():
    """20×(A+) then 8×(AB+, CD+)."""
    par = blocking.initialize({'ls': {'ncop': 1}})
    np.testing.assert_array_equal(par['trialorder'][:20], np.full(20, 1))
    np.testing.assert_array_equal(par['trialorder'][20:],
                                  np.tile([3, 4], 8))


def test_backwards_blocking_trialorder():
    """20×(AB+) then 8×(A+)."""
    par = backwards_blocking.initialize({'ls': {'ncop': 1}})
    np.testing.assert_array_equal(par['trialorder'][:20], np.full(20, 3))
    np.testing.assert_array_equal(par['trialorder'][20:], np.full(8, 1))


def test_overexpectation_trialorder():
    """50×(A+, B+) then 10×(AB+)."""
    par = overexpectation.initialize({'ls': {'ncop': 1}})
    np.testing.assert_array_equal(par['trialorder'][:100],
                                  np.tile([1, 2], 50))
    np.testing.assert_array_equal(par['trialorder'][100:], np.full(10, 3))


def test_conditioned_inhibition_trialorder():
    """100×(A+, ABweak+)."""
    par = conditioned_inhibition.initialize({'ls': {'ncop': 1}})
    np.testing.assert_array_equal(par['trialorder'], np.tile([1, 5], 100))


def test_phase_counts():
    """nphases matches the number of loop iterations in each schedule."""
    assert blocking.initialize({'ls': {'ncop': 1}})['nphases'] == 28
    assert backwards_blocking.initialize({'ls': {'ncop': 1}})['nphases'] == 28
    assert overexpectation.initialize({'ls': {'ncop': 1}})['nphases'] == 60
    assert conditioned_inhibition.initialize({'ls': {'ncop': 1}})['nphases'] == 100


# =============================================================================
# 3. Reward patterns (integration-flavored sanity checks)
# =============================================================================

def _reward_sequence(task):
    par = task.initialize({'ls': {'ncop': 1}})
    rewards = []
    for t in range(1, par['ntrials'] + 1):
        par['t'] = t
        rewards.append(task.randomize(par)['win'])
    return np.array(rewards)


def test_blocking_reward_pattern():
    """Acquisition rewards all 1.0, compound rewards all 1.0."""
    r = _reward_sequence(blocking)
    np.testing.assert_array_equal(r, np.ones(36))


def test_backwards_blocking_reward_pattern():
    r = _reward_sequence(backwards_blocking)
    np.testing.assert_array_equal(r, np.ones(28))


def test_overexpectation_reward_pattern():
    r = _reward_sequence(overexpectation)
    np.testing.assert_array_equal(r, np.ones(110))


def test_conditioned_inhibition_reward_pattern():
    """Alternating [A+ = 1, ABweak+ = 0.5]."""
    r = _reward_sequence(conditioned_inhibition)
    np.testing.assert_array_equal(r, np.tile([1.0, 0.5], 100))


# =============================================================================
# 4. Integration with run_task
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


@pytest.mark.parametrize('task,ntrials', ALL_TASKS,
                         ids=[t.__name__.split('.')[-1] for t, _ in ALL_TASKS])
def test_integration_run_task_nonlatent(task, ntrials):
    par = {'ls': {'ncop': 2}, 'modelname': 'RescorlaWagner', 'tau': 1.0}
    r = run_task(par, task, _trivial_agent)
    assert r['par']['ntrials'] == ntrials
    assert r['V'].shape == (ntrials + 1, 1 * 4 * 1 * 2)
    assert r['mu'].shape == (ntrials, 1)


@pytest.mark.parametrize('task,ntrials', ALL_TASKS,
                         ids=[t.__name__.split('.')[-1] for t, _ in ALL_TASKS])
def test_integration_run_task_latent(task, ntrials):
    par = {'ls': {'ncop': 2}, 'modelname': 'LatentState', 'tau': 1.0}
    r = run_task(par, task, _trivial_agent)
    assert r['par']['ntrials'] == ntrials
    assert r['V'].shape == (ntrials + 1, 1 * 6 * 1 * 2)


@pytest.mark.parametrize('task,_', ALL_TASKS,
                         ids=[t.__name__.split('.')[-1] for t, _ in ALL_TASKS])
def test_integration_par_case_ignored(task, _):
    """The task modules don't consume par['case'] — verify a stray value
    from run_example's caller doesn't break anything."""
    par = {'ls': {'ncop': 1}, 'modelname': 'RescorlaWagner',
           'case': 99, 'tau': 1.0}
    r = run_task(par, task, _trivial_agent)
    assert r['par']['ntrials'] > 0
