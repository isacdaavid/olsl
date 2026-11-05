import sys
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import patch

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from olsl.models.latent_state import LatentState

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def base_par():
    """Minimal default parameter dict matching MATLAB PatientParameters."""
    return {
        'L': 1,             # base hypotheses
        'D': 1,             # outcome dimension
        'C': 1,             # cue dimension (size of cue vector)
        'A': 1,             # number of arms
        'center': 0.5,
        'ntrials': 1,
        't': 1,
        'tau': 10.0,
        'ls': {
            'ncop': 1,      # copies (total states = L * ncop)
            'sigma0': 1.0,
            'gamma': 0.1,
            'alpha0': 0.5,
            'alpha1': 0.1,
            'alpha2': 0.1,
            'chi': 1,
            'delta': 1.0,
            'eta': 3.0,
        },
    }

@pytest.fixture
def base_world():
    """One trial: cue present, reward = 1."""
    return {
        'win': np.array([1.0]),
        'c_vec': np.ones((1, 1, 1, 1)),  # (D, C, L, A)
    }

@pytest.fixture(autouse=True)
def deterministic_choice():
    """Force SoftMaxChoice to always return arm 0 for reproducibility."""
    with patch(
            'olsl.common.math.SoftMaxChoice',
            return_value=(0, np.array([1.0]))
    ):
        yield

# =============================================================================
# 1. Initialization tests
# =============================================================================

def test_init_shapes(base_par):
    agent = LatentState(base_par, None, None)
    L = base_par['L'] * base_par['ls']['ncop']
    assert agent['V'].shape == (base_par['D'], base_par['C'], L)
    assert agent['lsb'].shape == (L,)
    assert agent['B'].shape == (base_par['C'], base_par['C'], base_par['D'], L)
    assert agent['Lmax'] == 1
    assert agent['q'] == 0.0

def test_init_defaults(base_par):
    agent = LatentState(base_par, None, None)
    np.testing.assert_array_equal(agent['V'], 0)
    np.testing.assert_allclose(agent['lsb'][:base_par['L']], 1.0 / base_par['L'])
    np.testing.assert_allclose(agent['sigmasq'], base_par['ls']['sigma0'] ** 2)
    for d in range(base_par['D']):
        for l in range(base_par['L']):
            np.testing.assert_array_equal(agent['B'][:, :, d, l], np.eye(base_par['C']))

def test_init_with_initialex(base_par):
    base_par['initialex'] = np.array([[[0.5]]])
    agent = LatentState(base_par, None, None)
    np.testing.assert_array_equal(agent['V'], 0.5)

def test_init_with_lsb0(base_par):
    """lsb0 semantics match MATLAB: scalar broadcasts, vector assigns element-wise."""
    base_par['L'] = 2
    base_par['ls']['ncop'] = 1

    # Scalar → broadcast to every initial hypothesis
    base_par['lsb0'] = 1.0
    agent = LatentState(base_par, None, None)
    np.testing.assert_array_equal(agent['lsb'], [1.0, 1.0])

    # Vector → element-wise assignment
    base_par['lsb0'] = np.array([1.0, 0.0])
    agent = LatentState(base_par, None, None)
    np.testing.assert_array_equal(agent['lsb'], [1.0, 0.0])

    # Vector with different weights → uniform
    base_par['lsb0'] = np.array([0.3, 0.7])
    agent = LatentState(base_par, None, None)
    np.testing.assert_array_equal(agent['lsb'], [0.3, 0.7])

# =============================================================================
# 2. Single-trial analytical tests (L = 1)
# =============================================================================

def test_rw_single_trial(base_par, base_world):
    """With L=1, LatentState reduces to standard RW.
    Analytical result for one trial:
        V_new = alpha0 * (reward - center)   (since V_old = 0, c = 1, B = I)
        sigmasq_new = (1-alpha1)*sigma0^2 + alpha1*(reward-center)^2
    """
    agent = LatentState(base_par, None, None)
    base_world['win'] = np.array([1.0])
    agent = LatentState(base_par, agent, base_world)

    reward = 1.0 - base_par['center']
    alpha0 = base_par['ls']['alpha0']
    alpha1 = base_par['ls']['alpha1']
    sigma0_sq = base_par['ls']['sigma0'] ** 2

    expected_V = alpha0 * reward
    expected_sigmasq = (1 - alpha1) * sigma0_sq + alpha1 * (reward ** 2)

    np.testing.assert_allclose(agent['V'][0, 0, 0], expected_V, rtol=1e-10)
    np.testing.assert_allclose(agent['sigmasq'], expected_sigmasq, rtol=1e-10)
    # B should remain identity (since c*c^T = 1 and B=I: B_new = B + alpha2*(lsb*1 - B) = I)
    np.testing.assert_allclose(agent['B'][0, 0, 0, 0], 1.0, rtol=1e-10)

def test_rw_two_trials(base_par, base_world):
    """Recursive check over two trials."""
    agent = LatentState(base_par, None, None)
    alpha0 = base_par['ls']['alpha0']
    c = base_par['center']

    r1, r2 = 1.0, 0.0
    base_world['win'] = np.array([r1])
    base_par['t'] = 1
    agent = LatentState(base_par, agent, base_world)
    V1 = alpha0 * (r1 - c)

    base_world['win'] = np.array([r2])
    base_par['t'] = 2
    base_par['ntrials'] = 2
    agent = LatentState(base_par, agent, base_world)
    V2 = V1 + alpha0 * ((r2 - c) - V1)

    np.testing.assert_allclose(agent['V'][0, 0, 0], V2, rtol=1e-10)

# =============================================================================
# 3. Belief collapse test (L > 1, ncop = 1)
# =============================================================================

def test_all_par_L_states_active_at_init(base_par):
    """With par.L=3, ncop=1, all 3 states are active from trial 1.
    No belief collapse: the model distributes belief over all hypotheses.
    This mirrors LatentStateBeliefs when Lmax*par.L == L_total.
    """
    base_par['L'] = 3
    base_par['ls']['ncop'] = 1  # L_total = 3

    agent = LatentState(base_par, None, None)
    world = {
        'win': np.array([1.0]),
        'c_vec': np.ones((1, 1, 3, 1)),  # (D, C, L_total, A) — 3 states, not 1
    }
    agent = LatentState(base_par, agent, world)

    # All 3 states receive equal treatment
    np.testing.assert_allclose(agent['lsb'], [1/3, 1/3, 1/3], atol=1e-10)
    # All 3 states learned the same V (since cues/priors were symmetric)
    np.testing.assert_allclose(
        agent['V'][0, 0, :], [0.08928571] * 3, rtol=1e-6
    )
    assert agent['Lmax'] == 1  # nothing was added


def test_belief_collapse_with_spare_slots(base_par):
    """With par.L=1 but ncop>1, only state 0 is active (Lmax=1).
    The pre-allocated spare slots stay at belief 0 — this is the
    'collapse' behavior: belief concentrates on the active copy.
    """
    base_par['L'] = 1
    base_par['ls']['ncop'] = 3  # L_total = 3, but only 1 active

    agent = LatentState(base_par, None, None)
    world = {
        'win': np.array([1.0]),
        'c_vec': np.ones((1, 1, 3, 1)),
    }
    agent = LatentState(base_par, agent, world)

    # Active state gets all the belief; spare slots are zeroed out
    np.testing.assert_allclose(agent['lsb'], [1.0, 0.0, 0.0], atol=1e-10)
    # Only the active state learned anything
    assert agent['V'][0, 0, 0] != 0
    assert agent['V'][0, 0, 1] == 0
    assert agent['V'][0, 0, 2] == 0
    assert agent['Lmax'] == 1

# =============================================================================
# 4. New-state detection test (ncop > 1)
# =============================================================================

def test_new_state_triggered_by_surprise(base_par, base_world):
    """A large surprise with ncop > 1 should trigger state addition.
    q = -log(likelihood) - delta. With reward=10, sigma0=1:
        LL[0] ≈ -50 => q ≈ 49 >> eta => new state created.
    """
    base_par['L'] = 1
    base_par['ls']['ncop'] = 2   # total L = 2
    base_par['ls']['delta'] = 1.0
    base_par['ls']['eta'] = 3.0
    # Use a very small gamma so prior is almost entirely on state 0
    base_par['ls']['gamma'] = 0.001

    agent = LatentState(base_par, None, None)
    base_world['win'] = np.array([10.0])
    base_world['c_vec'] = np.ones((1, 1, 2, 1))  # (D, C, L_total, A)
    agent = LatentState(base_par, agent, base_world)

    assert agent['Lmax'] == 2, f"Expected new state, got Lmax={agent['Lmax']}"
    # New state (index 1) should learn V = c * (win - center) / |c|^2
    # With win=10, center=0.5, c=1: V_new = 9.5
    expected_V_new = 10.0 - base_par['center']
    np.testing.assert_allclose(agent['V'][0, 0, 1], expected_V_new, rtol=1e-6)
    # Belief should have shifted to the new state
    assert agent['lsb'][1] > 0.9

def test_new_state_not_triggered_when_ncop_1(base_par, base_world):
    """If ncop=1, Lmax can never grow, even after a huge surprise."""
    base_par['L'] = 1
    base_par['ls']['ncop'] = 1
    agent = LatentState(base_par, None, None)
    base_world['win'] = np.array([100.0])
    agent = LatentState(base_par, agent, base_world)
    assert agent['Lmax'] == 1

# =============================================================================
# 5. Context shift test
# =============================================================================

def test_context_shift_temporal(base_par, base_world):
    """After a long time gap, beliefs should be pulled toward uniform."""
    base_par['L'] = 2
    base_par['ls']['ncop'] = 1
    base_par['ls']['gamma'] = 0.1
    base_par['time'] = np.array([1, 100])  # trial 1 at t=1, trial 2 at t=100
    base_par['ntrials'] = 2
    agent = LatentState(base_par, None, None)

    # Force lsb to be [1, 0] before trial 2, then trial 2 will apply context shift
    agent['lsb'] = np.array([1.0, 0.0])

    base_par['t'] = 2
    base_world['win'] = np.array([1.0])
    base_world['c_vec'] = np.ones((1, 1, 2, 1))
    agent = LatentState(base_par, agent, base_world)

    # After the gap, lsb should be closer to 0.5/0.5 than [1,0]
    # (exact value depends on alpha0 and lsb update, but belief in state 0 should be < 1)
    assert agent['lsb'][0] < 1.0

# =============================================================================
# 6. Golden trajectory test (MATLAB → Python equality)
# =============================================================================

GOLDEN_FILE = Path(__file__).parent / "fixtures" / "golden_latent_state.mat"

@pytest.mark.skipif(
    not GOLDEN_FILE.exists(),
    reason=(
        "Golden fixture not found. Generate with:\n"
        "  >> out = run_golden_LatentState();\n"
        "  >> save('tests/fixtures/golden_latent_state.mat', '-struct', 'out');\n"
        "The generator should use: L=1, ncop=2, ntrials=10, deterministic arm=0, "
        "and save V (11,D,C,L), lsb (11,L), sigmasq (11,), Lmax (11,), q (11,)."
    )
)
def test_golden_trajectory():
    """Compare the full trajectory against a MATLAB-saved fixture.
    The MATLAB .mat should contain these arrays (1-based; MATLAB order):
        V       -> (ntrials+1, D, C, L_total)
        lsb     -> (ntrials+1, L_total)
        sigmasq -> (ntrials+1,)
        Lmax    -> (ntrials+1,)
        q       -> (ntrials+1,)
    """
    import scipy.io
    mat = scipy.io.loadmat(str(GOLDEN_FILE))

    def _s(m, key):
        """Extract a MATLAB scalar from a loaded .mat dict."""
         # .item() works on any size-1 array (shape (1,1), (1,), scalar)
        return m[key].item()

    # Reconstruct par/world from the fixture
    par = {
        'L': 1, 'D': 1, 'C': 1, 'A': 1, 'center': 0.5,
        'ntrials': int(_s(mat, 'ntrials')),
        'tau':    float(_s(mat, 'tau')),
        'ls': {
            'ncop':   int(_s(mat, 'ncop')),
            'sigma0': float(_s(mat, 'sigma0')),
            'gamma':  float(_s(mat, 'gamma')),
            'alpha0': float(_s(mat, 'alpha0')),
            'alpha1': float(_s(mat, 'alpha1')),
            'alpha2': float(_s(mat, 'alpha2')),
            'chi':    int(_s(mat, 'chi')),
            'delta':  float(_s(mat, 'delta')),
            'eta':    float(_s(mat, 'eta')),
        },
    }

    rewards = mat['rewards'].flatten()   # shape (ntrials,)

    agent = LatentState(par, None, None)
    V_traj, lsb_traj = [agent['V'].copy()], [agent['lsb'].copy()]
    sigmasq_traj, Lmax_traj, q_traj = [agent['sigmasq']], [agent['Lmax']], [agent['q']]

    for t in range(par['ntrials']):
        par['t'] = t + 1
        world = {
            'win': np.array([rewards[t]]),
            'c_vec': np.ones((1, 1, par['L'] * par['ls']['ncop'], 1)),
        }
        agent = LatentState(par, agent, world)
        V_traj.append(agent['V'].copy())
        lsb_traj.append(agent['lsb'].copy())
        sigmasq_traj.append(agent['sigmasq'])
        Lmax_traj.append(agent['Lmax'])
        q_traj.append(agent['q'])

    V_traj = np.array(V_traj)
    lsb_traj = np.array(lsb_traj)
    sigmasq_traj = np.array(sigmasq_traj)
    Lmax_traj = np.array(Lmax_traj)
    q_traj = np.array(q_traj)

    # MATLAB convention: (ntrials+1, D, C, L_total). Numpy: (ntrials+1, D, C, L_total).
    assert V_traj.shape == mat['V'].shape, \
        f"V shape mismatch: python {V_traj.shape} vs matlab {mat['V'].shape}"
    assert lsb_traj.shape == mat['lsb'].shape, \
        f"lsb shape mismatch: python {lsb_traj.shape} vs matlab {mat['lsb'].shape}"
    np.testing.assert_allclose(
        V_traj,
        mat['V'],
        rtol=1e-8,
        atol=1e-10
    )
    np.testing.assert_allclose(
        lsb_traj,
        mat['lsb'],
        rtol=1e-8,
        atol=1e-10
    )
    np.testing.assert_allclose(
        sigmasq_traj,
        mat['sigmasq'].flatten(),
        rtol=1e-8
    )
    np.testing.assert_array_equal(
        Lmax_traj,
        mat['Lmax'].flatten().astype(int)
    )
    np.testing.assert_allclose(
        q_traj,
        mat['q'].flatten(),
        rtol=1e-8
    )
