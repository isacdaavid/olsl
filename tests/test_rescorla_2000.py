"""Tests for tasks/rescorla_2000.py.

The critical invariants:
  - Total trial count is 512 (500 + 8 + 4).
  - Training phase trial types are always a permutation of {1..5}.
  - Compound phase trial type is 6 (case 1) or 7 (case 2).
  - Test phase covers both rows 8 and 9.
  - Interaction columns equal the product of their single-cue columns.
  - LatentState keeps C=10; other models drop to C=5.
  - rng_seed controls reproducibility.
"""

import numpy as np
import pytest

from olsl.tasks import rescorla_2000 as r2000
from olsl.simulation.run_task import run_task


def make_par(case=1, ncop=1, modelname='', rng_seed=1, **overrides):
    par = {
        'case': case,
        'ls':   {'ncop': ncop},
        'rng_seed': rng_seed,
    }
    if modelname:
        par['modelname'] = modelname
    par.update(overrides)
    return par


# =============================================================================
# 1. initialize() — dimensions
# =============================================================================

def test_core_dimensions_nonlatent():
    par = r2000.initialize(make_par(case=1, modelname='RescorlaWagner'))
    assert par['A'] == 1
    assert par['D'] == 1
    assert par['C'] == 5
    assert par['nphases'] == 110
    assert par['ntrials'] == 512
    assert par['L'] == 1
    assert par['center'] == 0.5


def test_core_dimensions_latent():
    par = r2000.initialize(make_par(case=1, modelname='LatentState'))
    assert par['C'] == 10


def test_latent_state_hard_inherits_branch():
    par = r2000.initialize(make_par(case=1, modelname='LatentStateHard'))
    assert par['C'] == 10


# =============================================================================
# 2. initialize() — outcomes matrix
# =============================================================================

def test_outcomes_full_shape_latent():
    par = r2000.initialize(make_par(modelname='LatentState'))
    assert par['outcomes'].shape == (9, 11)


def test_outcomes_reduced_shape_nonlatent():
    par = r2000.initialize(make_par(modelname='RescorlaWagner'))
    assert par['outcomes'].shape == (9, 6)


def test_outcomes_reward_column():
    """Row order: A+, C+, X+, BX-, DX-, AB+, AB-, AD-, BC-."""
    par = r2000.initialize(make_par(modelname='LatentState'))
    np.testing.assert_array_equal(
        par['outcomes'][:, 0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    )


def test_outcomes_single_cue_columns():
    """A=col 1, B=col 2, C=col 3, D=col 4, X=col 5."""
    par = r2000.initialize(make_par(modelname='LatentState'))
    np.testing.assert_array_equal(
        par['outcomes'][:, 1],   # A
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
    )
    np.testing.assert_array_equal(
        par['outcomes'][:, 2],   # B
        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    )
    np.testing.assert_array_equal(
        par['outcomes'][:, 3],   # C
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    )
    np.testing.assert_array_equal(
        par['outcomes'][:, 4],   # D
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    )
    np.testing.assert_array_equal(
        par['outcomes'][:, 5],   # X
        [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )


def test_interaction_columns_are_products():
    """Each interaction column = product of its constituent single cues."""
    par = r2000.initialize(make_par(modelname='LatentState'))
    O = par['outcomes']
    A, B, C, D, X = O[:, 1], O[:, 2], O[:, 3], O[:, 4], O[:, 5]
    np.testing.assert_array_equal(O[:, 6],  B * X)   # BX_int
    np.testing.assert_array_equal(O[:, 7],  D * X)   # DX_int
    np.testing.assert_array_equal(O[:, 8],  A * B)   # AB_int
    np.testing.assert_array_equal(O[:, 9],  A * D)   # AD_int
    np.testing.assert_array_equal(O[:, 10], B * C)   # BC_int


# =============================================================================
# 3. initialize() — trial order structure
# =============================================================================

def test_trialorder_length():
    for case in (1, 2):
        par = r2000.initialize(make_par(case=case))
        assert len(par['trialorder']) == par['ntrials'] == 512


def test_trialorder_dtype():
    par = r2000.initialize(make_par())
    assert np.issubdtype(par['trialorder'].dtype, np.integer)


def test_training_phase_is_permutation_of_1_to_5():
    """Every block of 5 training trials contains {1, 2, 3, 4, 5}."""
    par = r2000.initialize(make_par(case=1, rng_seed=42))
    training = par['trialorder'][:500]
    for block in range(100):
        chunk = training[block * 5:(block + 1) * 5]
        assert sorted(chunk.tolist()) == [1, 2, 3, 4, 5], (
            f"block {block} is not a permutation: {chunk}"
        )


def test_compound_phase_uses_correct_row():
    """Trials 501-508 (0-based 500:508) use row 6 (case 1) or 7 (case 2)."""
    for case, expected_row in [(1, 6), (2, 7)]:
        par = r2000.initialize(make_par(case=case))
        np.testing.assert_array_equal(
            par['trialorder'][500:508],
            np.full(8, expected_row),
            err_msg=f"case={case}",
        )


def test_test_phase_covers_rows_8_and_9():
    """Last 4 trials (0-based 508:512) must include both 8 and 9, twice."""
    for case in (1, 2):
        par = r2000.initialize(make_par(case=case))
        test_trials = par['trialorder'][508:]
        assert len(test_trials) == 4
        assert (test_trials == 8).sum() == 2, f"case={case}"
        assert (test_trials == 9).sum() == 2, f"case={case}"


# =============================================================================
# 4. initialize() — rng_seed reproducibility
# =============================================================================

def test_same_seed_same_trialorder():
    par1 = r2000.initialize(make_par(rng_seed=7))
    par2 = r2000.initialize(make_par(rng_seed=7))
    np.testing.assert_array_equal(par1['trialorder'], par2['trialorder'])


def test_different_seed_different_trialorder():
    par1 = r2000.initialize(make_par(rng_seed=1))
    par2 = r2000.initialize(make_par(rng_seed=2))
    assert not np.array_equal(par1['trialorder'], par2['trialorder'])


def test_default_seed_is_1():
    """Matches MATLAB's hard-coded rng(1)."""
    par_default = r2000.initialize({'case': 1, 'ls': {'ncop': 1}})
    par_explicit = r2000.initialize(make_par(rng_seed=1))
    np.testing.assert_array_equal(
        par_default['trialorder'], par_explicit['trialorder'],
    )


def test_training_phase_identical_across_cases():
    """Training phase is the same for case 1 and 2 (only compound and
    test phases differ)."""
    par1 = r2000.initialize(make_par(case=1, rng_seed=5))
    par2 = r2000.initialize(make_par(case=2, rng_seed=5))
    np.testing.assert_array_equal(
        par1['trialorder'][:500], par2['trialorder'][:500],
    )


# =============================================================================
# 5. initialize() — case validation
# =============================================================================

def test_case_accepts_list_wrapper():
    par = r2000.initialize({'case': [1], 'ls': {'ncop': 1}})
    assert par['ntrials'] == 512


def test_case_accepts_numpy_scalar():
    par = r2000.initialize({'case': np.int64(2), 'ls': {'ncop': 1}})
    assert par['ntrials'] == 512


@pytest.mark.parametrize('bad_case', [0, 3, -1])
def test_invalid_case_raises(bad_case):
    with pytest.raises(ValueError, match=r"Rescorla2000 case"):
        r2000.initialize(make_par(case=bad_case))


# =============================================================================
# 6. initialize() — input isolation
# =============================================================================

def test_does_not_mutate_input():
    original = make_par(case=1, modelname='RescorlaWagner')
    r2000.initialize(original)
    for key in ('A', 'D', 'C', 'L', 'center', 'ntrials',
                'nphases', 'outcomes', 'trialorder'):
        assert key not in original


# =============================================================================
# 7. randomize() — shapes and content
# =============================================================================

def test_randomize_shape_latent():
    par = r2000.initialize(make_par(case=1, ncop=2, modelname='LatentState'))
    par['t'] = 1
    world = r2000.randomize(par)
    assert world['c_vec'].shape == (1, 10, 2, 1)


def test_randomize_shape_nonlatent():
    par = r2000.initialize(make_par(case=1, ncop=2, modelname='RescorlaWagner'))
    par['t'] = 1
    world = r2000.randomize(par)
    assert world['c_vec'].shape == (1, 5, 2, 1)


def test_randomize_win_is_python_float():
    par = r2000.initialize(make_par())
    par['t'] = 1
    world = r2000.randomize(par)
    assert isinstance(world['win'], float)


@pytest.mark.parametrize('case', [1, 2])
def test_randomize_cue_identical_across_states(case):
    par = r2000.initialize(make_par(case=case, ncop=3,
                                    modelname='LatentState'))
    par['t'] = 1
    world = r2000.randomize(par)
    for l in range(3):
        np.testing.assert_array_equal(
            world['c_vec'][0, :, l, 0],
            world['c_vec'][0, :, 0, 0],
        )


def test_randomize_compound_phase_cue(case=1):
    """At trial 501 (0-based 500), the compound trial for case 1 is AB+:
    cues A=1, B=1, AB_int=1, everything else 0."""
    par = r2000.initialize(make_par(case=1, modelname='LatentState'))
    par['t'] = 501
    cue = r2000.randomize(par)['c_vec'][0, :, 0, 0]
    # Columns: A, B, C, D, X, BX, DX, AB, AD, BC
    np.testing.assert_array_equal(cue, [1.0, 1.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 1.0, 0.0, 0.0])


def test_randomize_compound_phase_reward_case_2():
    """Case 2 compound is AB-: reward=0."""
    par = r2000.initialize(make_par(case=2, modelname='LatentState'))
    par['t'] = 501
    assert r2000.randomize(par)['win'] == 0.0


# =============================================================================
# 8. Integration with run_task
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
def test_integration_latent(case):
    par = make_par(case=case, ncop=2, modelname='LatentState')
    r = run_task(par, r2000, _trivial_agent)
    assert r['par']['ntrials'] == 512
    assert r['V'].shape == (513, 1 * 10 * 1 * 2)
    assert r['mu'].shape == (512, 1)


@pytest.mark.parametrize('case', [1, 2])
def test_integration_nonlatent(case):
    par = make_par(case=case, ncop=2, modelname='RescorlaWagner')
    r = run_task(par, r2000, _trivial_agent)
    assert r['par']['ntrials'] == 512
    assert r['V'].shape == (513, 1 * 5 * 1 * 2)


def test_integration_preserves_rng_seed():
    """rng_seed must survive the round-trip so subsequent reinitializations
    reproduce the same trial order."""
    par = make_par(case=1, ncop=1, modelname='LatentState', rng_seed=42)
    r = run_task(par, r2000, _trivial_agent)
    assert r['par']['rng_seed'] == 42
