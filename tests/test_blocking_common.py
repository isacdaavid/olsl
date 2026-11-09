"""Tests for tasks/_blocking_common.py.

The shared module provides two things:
  - ``OUTCOMES_FULL``: the 5×7 matrix [reward, A, B, C, D, AB_int, CD_int]
  - ``finalize_outcomes``: reduces columns for non-LatentState models
  - ``randomize``: shared trial-time cue/reward builder

These are exercised indirectly by the four task tests, but testing them
directly catches drift before it propagates to four downstream files.
"""

import numpy as np
import pytest

from olsl.tasks._blocking_common import (
    OUTCOMES_FULL, finalize_outcomes, randomize,
)


# =============================================================================
# OUTCOMES_FULL layout
# =============================================================================

def test_outcomes_full_shape():
    assert OUTCOMES_FULL.shape == (5, 7)


def test_outcomes_full_reward_column():
    """Row order: A+, B+, AB+, CD+, ABweak+."""
    np.testing.assert_array_equal(
        OUTCOMES_FULL[:, 0],
        [1.0, 1.0, 1.0, 1.0, 0.5],
    )


def test_outcomes_full_a_and_b_columns():
    """A column is [A+, B+, AB+, CD+, ABweak+] = [1,0,1,0,1]."""
    np.testing.assert_array_equal(
        OUTCOMES_FULL[:, 1],
        [1.0, 0.0, 1.0, 0.0, 1.0],
    )
    np.testing.assert_array_equal(
        OUTCOMES_FULL[:, 2],
        [0.0, 1.0, 1.0, 0.0, 1.0],
    )


def test_outcomes_full_c_and_d_columns():
    """C, D only appear in the CD+ row."""
    np.testing.assert_array_equal(
        OUTCOMES_FULL[:, 3],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    )
    np.testing.assert_array_equal(
        OUTCOMES_FULL[:, 4],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    )


def test_outcomes_full_interaction_columns():
    """AB_int fires on AB+ and ABweak+; CD_int fires on CD+ only."""
    np.testing.assert_array_equal(
        OUTCOMES_FULL[:, 5],
        [0.0, 0.0, 1.0, 0.0, 1.0],
    )
    np.testing.assert_array_equal(
        OUTCOMES_FULL[:, 6],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    )


def test_outcomes_full_is_not_mutated_by_finalize():
    """finalize_outcomes must copy, not alias, OUTCOMES_FULL."""
    before = OUTCOMES_FULL.copy()
    finalize_outcomes({'modelname': 'LatentState', 'ls': {'ncop': 1}})
    finalize_outcomes({'modelname': 'RescorlaWagner', 'ls': {'ncop': 1}})
    np.testing.assert_array_equal(OUTCOMES_FULL, before)


# =============================================================================
# finalize_outcomes
# =============================================================================

def test_finalize_latent_state_keeps_all_columns():
    par = finalize_outcomes({'modelname': 'LatentState', 'ls': {'ncop': 1}})
    assert par['C'] == 6
    assert par['outcomes'].shape == (5, 7)


def test_finalize_non_latent_drops_interactions():
    par = finalize_outcomes({'modelname': 'RescorlaWagner',
                             'ls': {'ncop': 1}})
    assert par['C'] == 4
    assert par['outcomes'].shape == (5, 5)


def test_finalize_missing_modelname_defaults_to_non_latent():
    par = finalize_outcomes({'ls': {'ncop': 1}})
    assert par['C'] == 4
    assert par['outcomes'].shape == (5, 5)


def test_finalize_latent_state_hard_inherits_branch():
    """LatentStateHard contains 'LatentState' as substring → 6 cues."""
    par = finalize_outcomes({'modelname': 'LatentStateHard',
                             'ls': {'ncop': 1}})
    assert par['C'] == 6


def test_finalize_returns_copied_outcomes():
    """Mutating the returned outcomes must not touch OUTCOMES_FULL."""
    par = finalize_outcomes({'modelname': 'LatentState', 'ls': {'ncop': 1}})
    par['outcomes'][0, 0] = 999.0
    assert OUTCOMES_FULL[0, 0] == 1.0


# =============================================================================
# randomize
# =============================================================================

def _make_par(trialorder, t=1, C=4, ncop=1, D=1, A=1, L=1,
              modelname='RescorlaWagner'):
    return finalize_outcomes({
        'trialorder': np.asarray(trialorder, dtype=int),
        't':          t,
        'C':          C,
        'ncop':       ncop,
        'D':          D,
        'A':          A,
        'L':          L,
        'ls':         {'ncop': ncop},
        'modelname':  modelname,
    })


def test_randomize_c_vec_shape_nonlatent():
    par = _make_par(trialorder=[1], ncop=2)
    par['t'] = 1
    world = randomize(par)
    assert world['c_vec'].shape == (1, 4, 2, 1)   # D, C, L*ncop, A


def test_randomize_c_vec_shape_latent():
    par = _make_par(trialorder=[1], ncop=2, modelname='LatentState')
    par['t'] = 1
    world = randomize(par)
    assert world['c_vec'].shape == (1, 6, 2, 1)


def test_randomize_win_is_python_float():
    par = _make_par(trialorder=[1])
    par['t'] = 1
    world = randomize(par)
    assert isinstance(world['win'], float)
    assert world['win'] == 1.0


def test_randomize_1based_trial_indexing():
    """par['t']=1 picks trialorder[0], par['t']=2 picks trialorder[1]."""
    # A+ on trial 1, B+ on trial 2
    par = _make_par(trialorder=[1, 2])
    par['t'] = 1
    assert randomize(par)['win'] == 1.0
    cue1 = randomize(par)['c_vec'][0, :, 0, 0]
    np.testing.assert_array_equal(cue1, [1.0, 0.0, 0.0, 0.0])

    par['t'] = 2
    assert randomize(par)['win'] == 1.0
    cue2 = randomize(par)['c_vec'][0, :, 0, 0]
    np.testing.assert_array_equal(cue2, [0.0, 1.0, 0.0, 0.0])


def test_randomize_cue_identical_across_states():
    par = _make_par(trialorder=[3], ncop=4)   # AB+
    par['t'] = 1
    world = randomize(par)
    for l in range(4):
        np.testing.assert_array_equal(
            world['c_vec'][0, :, l, 0],
            world['c_vec'][0, :, 0, 0],
        )


def test_randomize_abweak_reward_is_half():
    par = _make_par(trialorder=[5])   # ABweak+
    par['t'] = 1
    assert randomize(par)['win'] == 0.5
