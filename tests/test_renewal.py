"""Tests for tasks/renewal.py.

The Renewal task has two cases with different extinction contexts, and
behaves differently for LatentState models (which treat context as
hypotheses) vs. other models (which treat context as cues). The tests
cover both paths plus the integration with run_task.
"""

import numpy as np
import pytest

from olsl.tasks import renewal
from olsl.simulation.run_task import run_task


# =============================================================================
# Helpers
# =============================================================================

def make_par(case=1, ncop=1, modelname='', **overrides):
    par = {'case': case, 'ls': {'ncop': ncop}}
    if modelname:
        par['modelname'] = modelname
    par.update(overrides)
    return par


# =============================================================================
# 1. initialize() — base dimensions and trialorder
# =============================================================================

def test_initialize_sets_core_dimensions_nonlatent():
    par = renewal.initialize(make_par(modelname='RescorlaWagner'))
    assert par['A'] == 1
    assert par['D'] == 1
    assert par['C'] == 3                # non-LatentState keeps 3 cues
    assert par['nphases'] == 3
    assert par['ntrials'] == 50
    assert par['L'] == 1
    assert par['center'] == 0.5
    assert par['r0'] == 1.0


def test_initialize_case_1_trialorder():
    """Case 1: 20 row-1, 20 row-2, 10 row-1."""
    par = renewal.initialize(make_par(case=1))
    np.testing.assert_array_equal(par['trialorder'][:20], np.full(20, 1))
    np.testing.assert_array_equal(par['trialorder'][20:40], np.full(20, 2))
    np.testing.assert_array_equal(par['trialorder'][40:], np.full(10, 1))


def test_initialize_case_2_trialorder():
    """Case 2: 20 row-1, 20 row-3, 10 row-1."""
    par = renewal.initialize(make_par(case=2))
    np.testing.assert_array_equal(par['trialorder'][:20], np.full(20, 1))
    np.testing.assert_array_equal(par['trialorder'][20:40], np.full(20, 3))
    np.testing.assert_array_equal(par['trialorder'][40:], np.full(10, 1))


def test_initialize_outcomes_layout():
    """Rows = [reward, ctx_A, ctx_B, cue]; acquisition/renewal in context A."""
    par = renewal.initialize(make_par(modelname='RescorlaWagner'))
    np.testing.assert_array_equal(
        par['outcomes'],
        [[1.0, 1.0, 0.0, 1.0],
         [0.0, 1.0, 0.0, 1.0],
         [0.0, 0.0, 1.0, 1.0]],
    )


def test_initialize_accepts_scalar_or_sequence_case():
    for case_val in (1, np.int64(1), [1], np.array([1])):
        par = renewal.initialize(make_par(case=case_val))
        assert par['ntrials'] == 50


def test_initialize_unknown_case_raises():
    with pytest.raises(ValueError, match=r"Renewal case"):
        renewal.initialize(make_par(case=3))


def test_initialize_does_not_mutate_input():
    original = make_par(case=1)
    renewal.initialize(original)
    for key in ('A', 'D', 'C', 'trialorder', 'outcomes'):
        assert key not in original


# =============================================================================
# 2. initialize() — LatentState-specific adjustments
# =============================================================================

def test_latent_state_uses_one_cue():
    """LatentState models treat context as hypotheses, not cues → C=1."""
    par = renewal.initialize(make_par(case=1, modelname='LatentState'))
    assert par['C'] == 1


def test_latent_state_outcomes_have_two_columns():
    """[reward, cue] only — context columns are dropped."""
    par = renewal.initialize(make_par(case=1, modelname='LatentState'))
    assert par['outcomes'].shape == (3, 2)
    np.testing.assert_array_equal(
        par['outcomes'],
        [[1.0, 1.0],
         [0.0, 1.0],
         [0.0, 1.0]],
    )


def test_latent_state_case_1_has_no_context_shift():
    par = renewal.initialize(make_par(case=1, modelname='LatentState'))
    assert 'context' not in par


def test_latent_state_case_2_has_context_shift():
    """Case 2 signals context change at trials 21 and 41 (1-based)."""
    par = renewal.initialize(make_par(case=2, modelname='LatentState'))
    assert 'context' in par
    np.testing.assert_array_equal(par['context'], [21, 41])


def test_non_latent_state_has_no_context_shift():
    """Non-LatentState models use explicit context cues, no belief reset."""
    for model in ('RescorlaWagner', 'fRLDecay', 'InfiniteMixture',
                  'Gershman2017', 'Redish2007'):
        par = renewal.initialize(make_par(case=2, modelname=model))
        assert 'context' not in par, f"{model} should not set par['context']"


def test_missing_modelname_treated_as_non_latent():
    """Direct call with no modelname should not crash and should not
    trigger the LatentState branch."""
    par = renewal.initialize({'case': 1, 'ls': {'ncop': 1}})
    assert par['C'] == 3
    assert 'context' not in par


# =============================================================================
# 3. randomize() — reward sequence and shapes
# =============================================================================

def test_randomize_c_vec_shape_nonlatent():
    par = renewal.initialize(make_par(case=1, ncop=2,
                                      modelname='RescorlaWagner'))
    par['t'] = 1
    world = renewal.randomize(par)
    assert world['c_vec'].shape == (1, 3, 2, 1)   # (D, C, L*ncop, A)


def test_randomize_c_vec_shape_latent():
    par = renewal.initialize(make_par(case=1, ncop=2,
                                      modelname='LatentState'))
    par['t'] = 1
    world = renewal.randomize(par)
    assert world['c_vec'].shape == (1, 1, 2, 1)   # C=1 for LatentState


def test_randomize_win_is_python_float():
    par = renewal.initialize(make_par(case=1))
    par['t'] = 1
    world = renewal.randomize(par)
    assert isinstance(world['win'], float)


@pytest.mark.parametrize('case', [1, 2])
def test_randomize_reward_pattern_latent(case):
    """After LatentState column reduction, both cases produce
    reward=1 (trials 1-20), reward=0 (21-40), reward=1 (41-50)."""
    par = renewal.initialize(make_par(case=case, modelname='LatentState'))
    rewards = []
    for t in range(1, par['ntrials'] + 1):
        par['t'] = t
        rewards.append(renewal.randomize(par)['win'])

    np.testing.assert_array_equal(
        rewards,
        [1.0] * 20 + [0.0] * 20 + [1.0] * 10,
    )


def test_randomize_context_cues_case_1():
    """Case 1: ctx_A=1 throughout; ctx_B=0 throughout."""
    par = renewal.initialize(make_par(case=1, modelname='RescorlaWagner'))
    for t in (1, 20, 21, 40, 41, 50):
        par['t'] = t
        cue = renewal.randomize(par)['c_vec'][0, :, 0, 0]
        np.testing.assert_array_equal(cue, [1.0, 0.0, 1.0],
                                      err_msg=f"case 1, trial {t}")


def test_randomize_context_cues_case_2():
    """Case 2: ctx_A=1 during acquisition/renewal, ctx_B=1 during extinction."""
    par = renewal.initialize(make_par(case=2, modelname='RescorlaWagner'))
    for t in (1, 20, 41, 50):
        par['t'] = t
        cue = renewal.randomize(par)['c_vec'][0, :, 0, 0]
        np.testing.assert_array_equal(cue, [1.0, 0.0, 1.0],
                                      err_msg=f"case 2, trial {t} (ctx A)")
    for t in (21, 30, 40):
        par['t'] = t
        cue = renewal.randomize(par)['c_vec'][0, :, 0, 0]
        np.testing.assert_array_equal(cue, [0.0, 1.0, 1.0],
                                      err_msg=f"case 2, trial {t} (ctx B)")


def test_randomize_cue_always_one():
    """The cue column is 1 on every trial — it's always on."""
    for case in (1, 2):
        par = renewal.initialize(make_par(case=case,
                                          modelname='RescorlaWagner'))
        for t in range(1, par['ntrials'] + 1):
            par['t'] = t
            cue = renewal.randomize(par)['c_vec'][0, :, 0, 0]
            assert cue[-1] == 1.0, f"case={case}, trial={t}: cue should be 1"


def test_randomize_cues_identical_across_states():
    par = renewal.initialize(make_par(case=1, ncop=3,
                                      modelname='RescorlaWagner'))
    par['t'] = 1
    world = renewal.randomize(par)
    for l in range(3):
        np.testing.assert_array_equal(
            world['c_vec'][0, :, l, 0],
            world['c_vec'][0, :, 0, 0],
        )


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


@pytest.mark.parametrize('case', [1, 2])
def test_integration_with_run_task(case):
    par = make_par(case=case, ncop=2, modelname='LatentState')
    r = run_task(par, renewal, _trivial_agent)

    assert r['par']['ntrials'] == 50
    assert r['V'].shape == (51, 1 * 1 * 1 * 2)   # C=1 for LatentState
    assert r['mu'].shape == (50, 1)
    assert not np.any(np.isnan(r['mu']))


def test_integration_preserves_context_key():
    """par['context'] must survive the round-trip through run_task."""
    par = make_par(case=2, ncop=2, modelname='LatentState')
    r = run_task(par, renewal, _trivial_agent)
    assert 'context' in r['par']
    np.testing.assert_array_equal(r['par']['context'], [21, 41])
