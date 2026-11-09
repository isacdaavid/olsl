"""Tests for tasks/wilson_boumphrey_pearce.py.

The critical invariants are:
  - Case 'E' has three phases (10 + 40 + 10 trials); case 'C' has two
    (50 + 10).
  - The interaction column equals light × tone in every outcome row.
  - LatentState keeps C=3 (with interaction); other models get C=2.
  - Reward patterns match the phase structure.
"""

import numpy as np
import pytest

from olsl.tasks import wilson_boumphrey_pearce as wbp
from olsl.simulation.run_task import run_task


def make_par(case='E', ncop=1, modelname='', **overrides):
    par = {'case': case, 'ls': {'ncop': ncop}}
    if modelname:
        par['modelname'] = modelname
    par.update(overrides)
    return par


# =============================================================================
# 1. initialize() — dimensions
# =============================================================================

def test_core_dimensions_nonlatent():
    par = wbp.initialize(make_par(case='E', modelname='RescorlaWagner'))
    assert par['A'] == 1
    assert par['D'] == 1
    assert par['C'] == 2
    assert par['nphases'] == 9
    assert par['ntrials'] == 60
    assert par['L'] == 1
    assert par['center'] == 0.5


def test_core_dimensions_latent():
    par = wbp.initialize(make_par(case='E', modelname='LatentState'))
    assert par['C'] == 3


def test_latent_state_hard_inherits_latent_branch():
    par = wbp.initialize(make_par(case='E', modelname='LatentStateHard'))
    assert par['C'] == 3


# =============================================================================
# 2. initialize() — outcomes matrix
# =============================================================================

def test_outcomes_full_layout_latent():
    par = wbp.initialize(make_par(case='E', modelname='LatentState'))
    np.testing.assert_array_equal(
        par['outcomes'],
        [[1.0, 1.0, 1.0, 1.0],
         [0.0, 1.0, 1.0, 1.0],
         [0.0, 1.0, 0.0, 0.0],
         [1.0, 1.0, 0.0, 0.0]],
    )


def test_outcomes_reduced_for_nonlatent():
    par = wbp.initialize(make_par(case='E', modelname='RescorlaWagner'))
    np.testing.assert_array_equal(
        par['outcomes'],
        [[1.0, 1.0, 1.0],
         [0.0, 1.0, 1.0],
         [0.0, 1.0, 0.0],
         [1.0, 1.0, 0.0]],
    )


def test_interaction_equals_light_and_tone():
    """Column 3 (interaction) should equal column 1 (light) * column 2 (tone)."""
    par = wbp.initialize(make_par(case='E', modelname='LatentState'))
    light = par['outcomes'][:, 1]
    tone  = par['outcomes'][:, 2]
    inter = par['outcomes'][:, 3]
    np.testing.assert_array_equal(inter, light * tone)


# =============================================================================
# 3. initialize() — trial orders
# =============================================================================

def test_case_e_trialorder():
    par = wbp.initialize(make_par(case='E'))
    assert len(par['trialorder']) == 60
    np.testing.assert_array_equal(par['trialorder'][:10],  np.tile([1, 2], 5))
    np.testing.assert_array_equal(par['trialorder'][10:50], np.tile([1, 3], 20))
    np.testing.assert_array_equal(par['trialorder'][50:],  np.full(10, 4))


def test_case_c_trialorder():
    par = wbp.initialize(make_par(case='C'))
    assert len(par['trialorder']) == 60
    np.testing.assert_array_equal(par['trialorder'][:50], np.tile([1, 2], 25))
    np.testing.assert_array_equal(par['trialorder'][50:], np.full(10, 4))


def test_trialorder_dtype():
    for case in ('E', 'C'):
        par = wbp.initialize(make_par(case=case))
        assert np.issubdtype(par['trialorder'].dtype, np.integer)


def test_trialorder_length_matches_ntrials():
    for case in ('E', 'C'):
        par = wbp.initialize(make_par(case=case))
        assert len(par['trialorder']) == par['ntrials']


# =============================================================================
# 4. initialize() — case handling
# =============================================================================

def test_case_accepts_list_wrapper():
    par = wbp.initialize({'case': ['E'], 'ls': {'ncop': 1}})
    assert par['ntrials'] == 60


def test_case_accepts_numpy_array():
    par = wbp.initialize({'case': np.array(['C']), 'ls': {'ncop': 1}})
    assert par['ntrials'] == 60


def test_invalid_case_raises():
    with pytest.raises(ValueError, match=r"WilsonBoumphreyPearce case"):
        wbp.initialize(make_par(case='X'))


def test_initialize_does_not_mutate_input():
    original = make_par(case='E', modelname='RescorlaWagner')
    wbp.initialize(original)
    for key in ('A', 'D', 'C', 'L', 'center', 'ntrials',
                'nphases', 'outcomes', 'trialorder'):
        assert key not in original


# =============================================================================
# 5. randomize() — shapes and reward patterns
# =============================================================================

def test_randomize_c_vec_shape_latent():
    par = wbp.initialize(make_par(case='E', ncop=2, modelname='LatentState'))
    par['t'] = 1
    world = wbp.randomize(par)
    assert world['c_vec'].shape == (1, 3, 2, 1)


def test_randomize_c_vec_shape_nonlatent():
    par = wbp.initialize(make_par(case='E', ncop=2, modelname='RescorlaWagner'))
    par['t'] = 1
    world = wbp.randomize(par)
    assert world['c_vec'].shape == (1, 2, 2, 1)


def test_randomize_win_is_python_float():
    par = wbp.initialize(make_par(case='E'))
    par['t'] = 1
    world = wbp.randomize(par)
    assert isinstance(world['win'], float)


@pytest.mark.parametrize('case', ['E', 'C'])
def test_reward_pattern(case):
    par = wbp.initialize(make_par(case=case))
    rewards = []
    for t in range(1, 61):
        par['t'] = t
        rewards.append(wbp.randomize(par)['win'])

    if case == 'E':
        expected = [1, 0] * 5 + [1, 0] * 20 + [1] * 10
    else:
        expected = [1, 0] * 25 + [1] * 10

    np.testing.assert_array_equal(rewards, expected, err_msg=f"case={case}")


# =============================================================================
# 6. randomize() — cue content at key trials
# =============================================================================

def test_compound_trial_cue_latent():
    """Trial type 1 (compound, rewarded): cue = [light, tone, interaction] = [1,1,1]."""
    par = wbp.initialize(make_par(case='E', modelname='LatentState'))
    par['t'] = 1
    cue = wbp.randomize(par)['c_vec'][0, :, 0, 0]
    np.testing.assert_array_equal(cue, [1.0, 1.0, 1.0])


def test_light_only_trial_cue_latent():
    """Trial type 3 (light alone, unrewarded): cue = [1, 0, 0]."""
    par = wbp.initialize(make_par(case='E', modelname='LatentState'))
    par['t'] = 12   # Phase 2, even index → trial type 3
    assert int(par['trialorder'][par['t'] - 1]) == 3
    cue = wbp.randomize(par)['c_vec'][0, :, 0, 0]
    np.testing.assert_array_equal(cue, [1.0, 0.0, 0.0])


def test_rewarded_light_trial_cue_latent():
    """Trial type 4 (light alone, rewarded): cue = [1, 0, 0], reward = 1."""
    par = wbp.initialize(make_par(case='E', modelname='LatentState'))
    par['t'] = 51   # Phase 3
    assert int(par['trialorder'][par['t'] - 1]) == 4
    world = wbp.randomize(par)
    np.testing.assert_array_equal(world['c_vec'][0, :, 0, 0], [1.0, 0.0, 0.0])
    assert world['win'] == 1.0


def test_cue_drops_interaction_for_nonlatent():
    """Trial type 1 for non-LatentState: [light, tone] = [1, 1]."""
    par = wbp.initialize(make_par(case='E', modelname='RescorlaWagner'))
    par['t'] = 1
    cue = wbp.randomize(par)['c_vec'][0, :, 0, 0]
    np.testing.assert_array_equal(cue, [1.0, 1.0])


def test_randomize_cue_identical_across_states():
    par = wbp.initialize(make_par(case='E', ncop=3, modelname='LatentState'))
    par['t'] = 1
    world = wbp.randomize(par)
    for l in range(3):
        np.testing.assert_array_equal(
            world['c_vec'][0, :, l, 0],
            world['c_vec'][0, :, 0, 0],
        )


# =============================================================================
# 7. Integration with run_task
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


@pytest.mark.parametrize('case', ['E', 'C'])
def test_integration_with_run_task_latent(case):
    par = make_par(case=case, ncop=2, modelname='LatentState')
    r = run_task(par, wbp, _trivial_agent)
    assert r['par']['ntrials'] == 60
    assert r['V'].shape == (61, 1 * 3 * 1 * 2)
    assert r['mu'].shape == (60, 1)


@pytest.mark.parametrize('case', ['E', 'C'])
def test_integration_with_run_task_nonlatent(case):
    par = make_par(case=case, ncop=2, modelname='RescorlaWagner')
    r = run_task(par, wbp, _trivial_agent)
    assert r['par']['ntrials'] == 60
    assert r['V'].shape == (61, 1 * 2 * 1 * 2)
