"""Tests for simulation/run_example.py.

Two layers of tests:

  1. Direct tests of ``_postprocess`` — the shape-and-weighting helper that
     reimplements MATLAB's 4-D ``reshape(... .* ... ./ nrm, ...)`` chain. This
     is the trickiest part of the port (column-major vs. row-major) and
     deserves hand-worked numeric examples.

  2. Integration tests of ``run_example`` — verify the orchestration contract
     (dict keys, shapes, deepcopy isolation, MMKF presence/absence, pickling).
     These use lightweight fakes so they don't depend on LatentState internals.
"""

import pickle

import numpy as np
import pytest

from olsl.simulation.run_example import run_example, save_results, _postprocess


# =============================================================================
# Fakes (mirror the ones in test_run_task.py; kept inline for independence)
# =============================================================================

class FakeTask:
    def __init__(self, rewards):
        self.rewards = np.asarray(rewards, dtype=float)

    def initialize(self, par):
        par = dict(par)
        par['ntrials'] = len(self.rewards)
        return par

    def randomize(self, par):
        D = par['D']
        t_idx = par['t'] - 1
        return {
            'win':   np.full(D, self.rewards[t_idx]),
            'c_vec': np.ones((D, par['C'], 1, par['A'])),
        }


def make_agent_fn(D=1, C=1, n_states=1):
    """Fake agent: sets V[t] = ones * par['t'] so postprocess results are
    predictable without depending on a specific learning rule."""
    def agent_fn(par, agent, world):
        if agent is None:
            return {
                'V':   np.zeros((D, C, n_states)),
                'lsb': np.ones(n_states) / n_states,
                'win': np.zeros(D),
                'mu':  np.zeros((D, par['A'])),
            }
        a = dict(agent)
        reward = np.atleast_1d(world['win'])
        a['V']   = np.ones((D, C, n_states)) * par['t']
        a['win'] = reward
        a['mu']  = np.array([[a['V'].sum()] * par['A']] * D)
        return a
    return agent_fn


def make_observer_fn(L_kf=3):
    def observer_fn(par, mmkf, agent, world):
        if mmkf is None:
            return {
                'lsb_prior':   np.full(L_kf, 1.0 / L_kf),
                'likelihoods': np.ones(L_kf),
                'lsb':         np.full(L_kf, 1.0 / L_kf),
                'X_t_est':     0,
            }
        return {
            'lsb_prior':   np.full(L_kf, float(par['t'])),
            'likelihoods': np.full(L_kf, float(par['t'])),
            'lsb':         np.full(L_kf, 1.0 / L_kf),
            'X_t_est':     int(par['t']) % L_kf,
        }
    return observer_fn


@pytest.fixture
def base_par():
    return {
        'D': 1, 'C': 1, 'A': 1, 'center': 0.0,
        'L': 1, 'ls': {'ncop': 1},
        'tau': 1.0,
    }


# =============================================================================
# 1. Direct _postprocess tests (LatentState weighted averaging)
# =============================================================================

def test_postprocess_latent_state_L1_ncop2():
    """L=1, ncop=2, D=C=1. Weighted avg of two copies of the same hypothesis.

    Layout: l = l1 + l2*L = 0*1 + l2, so V[t] = [v(l2=0), v(l2=1)].
    Expected v_proc[t, 0] = (V[t,0]*lsb[t,0] + V[t,1]*lsb[t,1]) / (lsb[t,0]+lsb[t,1]).
    """
    V   = np.array([[0.0, 0.0], [1.0, 2.0]])
    LSB = np.array([[0.5, 0.5], [0.3, 0.7]])
    par = {'D': 1, 'C': 1, 'L': 1, 'ls': {'ncop': 2}}

    v_proc, lsb_proc = _postprocess('LatentState', V, LSB, par)

    assert v_proc.shape == (2, 1)           # D*C*L = 1
    np.testing.assert_allclose(lsb_proc, LSB)   # pass-through
    np.testing.assert_allclose(v_proc[0, 0], 0.0)                          # 0*0.5+0*0.5
    np.testing.assert_allclose(v_proc[1, 0], 0.3*1.0 + 0.7*2.0)            # = 1.7


def test_postprocess_latent_state_L2_ncop1_is_identity():
    """L=2, ncop=1 → one copy per hypothesis → v_proc == V exactly."""
    V   = np.array([[0.0, 0.0], [1.5, 2.5]])
    LSB = np.array([[0.5, 0.5], [0.4, 0.6]])
    par = {'D': 1, 'C': 1, 'L': 2, 'ls': {'ncop': 1}}

    v_proc, _ = _postprocess('LatentState', V, LSB, par)

    assert v_proc.shape == (2, 2)           # D*C*L = 2
    np.testing.assert_allclose(v_proc, V)


def test_postprocess_latent_state_L2_ncop2():
    """General case: 4 states, hypotheses x copies.

    Layout: l = l1 + l2*L, so within a trial V = [l1=0,l2=0 | l1=1,l2=0 |
                                                 l1=0,l2=1 | l1=1,l2=1].
    Output index: l1 fastest (last dim after transpose).
    """
    V   = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0, 4.0],   # a0 a1 b0 b1
    ])
    LSB = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [0.10, 0.20, 0.30, 0.40],   # copy0=0.1 (l1=0), 0.2 (l1=1); copy1=0.3, 0.4
    ])
    par = {'D': 1, 'C': 1, 'L': 2, 'ls': {'ncop': 2}}

    v_proc, _ = _postprocess('LatentState', V, LSB, par)

    assert v_proc.shape == (2, 2)           # D*C*L = 2

    # t=0: uniform copies of 0 → v_proc[0] = [0, 0]
    np.testing.assert_allclose(v_proc[0], [0.0, 0.0])

    # t=1:
    #   l1=0 (copies l2=0,1 with values 1.0, 3.0, weights 0.1, 0.3):
    #       (0.1*1.0 + 0.3*3.0) / 0.4 = 1.0/0.4 = 2.5
    #   l1=1 (copies l2=0,1 with values 2.0, 4.0, weights 0.2, 0.4):
    #       (0.2*2.0 + 0.4*4.0) / 0.6 = 2.0/0.6 = 3.3333...
    np.testing.assert_allclose(v_proc[1], [2.5, 2.0 / 0.6], rtol=1e-10)


# =============================================================================
# 2. Direct _postprocess tests (Gershman2017, passthrough, unknown)
# =============================================================================

def test_postprocess_gershman2017():
    """K=2, D=C=1: weighted sum across K states."""
    V   = np.array([[0.0, 0.0], [1.0, 3.0]])
    LSB = np.array([[0.5, 0.5], [0.4, 0.6]])
    par = {'D': 1, 'C': 1, 'gershman': {'K': 2}}

    v_proc, lsb_proc = _postprocess('Gershman2017', V, LSB, par)

    assert v_proc.shape == (2, 1)           # D*C (no L dim)
    np.testing.assert_allclose(lsb_proc, LSB)
    np.testing.assert_allclose(v_proc[0, 0], 0.0)
    np.testing.assert_allclose(v_proc[1, 0], 0.4 * 1.0 + 0.6 * 3.0)   # = 2.2


@pytest.mark.parametrize('model_name', ['InfiniteMixture', 'Redish2007'])
def test_postprocess_passthrough(model_name):
    V   = np.array([[0.0, 0.0], [1.5, 2.5]])
    LSB = np.array([[0.5, 0.5], [0.4, 0.6]])
    par = {'D': 1, 'C': 1, 'L': 2, 'ls': {'ncop': 1}}

    v_proc, lsb_proc = _postprocess(model_name, V, LSB, par)

    np.testing.assert_array_equal(v_proc, V)
    np.testing.assert_array_equal(lsb_proc, LSB)


def test_postprocess_unknown_model():
    """Unrecognized model: V unchanged, LSB replaced with zeros (T, 1)."""
    V   = np.array([[0.0, 0.0], [1.5, 2.5]])
    LSB = np.array([[0.5, 0.5], [0.4, 0.6]])
    par = {'D': 1, 'C': 1}

    v_proc, lsb_proc = _postprocess('SomeUnknownModel', V, LSB, par)

    np.testing.assert_array_equal(v_proc, V)
    assert lsb_proc.shape == (2, 1)
    np.testing.assert_array_equal(lsb_proc, 0.0)


# =============================================================================
# 3. Integration: output structure and shapes
# =============================================================================

def test_output_keys_present(base_par):
    task = FakeTask([1.0])
    models = [('LatentState', make_agent_fn())]
    r = run_example(base_par, task, models, nrep=1, cases=[1], progress=False)

    for key in ('MU', 'LSB', 'V', 'XT',
                'MMKF_PRIOR', 'MMKF_LIKE', 'MMKF_POST', 'MMKF_XT',
                'par_by_key'):
        assert key in r


def test_shapes_latent_state_no_observer(base_par):
    task = FakeTask([1.0, 0.0, 1.0])
    models = [('LatentState', make_agent_fn(n_states=2))]
    base_par['L'] = 1
    base_par['ls']['ncop'] = 2

    r = run_example(base_par, task, models, nrep=3, cases=[1], progress=False)
    key = ('LatentState', 1)

    ntrials = 3
    # MU: (T, D*A)
    assert r['MU'][key]['mean'].shape == (ntrials, 1)
    assert r['MU'][key]['std'].shape  == (ntrials, 1)
    # LSB: (T+1, L*ncop) — starts at init
    assert r['LSB'][key]['mean'].shape == (ntrials + 1, 2)
    # V (postprocessed): (T+1, D*C*L)
    assert r['V'][key]['mean'].shape == (ntrials + 1, 1)
    assert r['V'][key]['std'].shape  == (ntrials + 1, 1)
    # XT: (T, nrep) — always populated by FakeAgent? No, only if x_t exists.
    # FakeAgent does NOT set x_t, so XT is absent for this key.
    assert key not in r['XT']


def test_shapes_with_observer(base_par):
    base_par['kf'] = {'L': 4}
    task = FakeTask([1.0, 0.0, 1.0])
    models = [('LatentState', make_agent_fn())]
    observer = make_observer_fn(L_kf=4)

    r = run_example(base_par, task, models, nrep=3, cases=[1],
                    observer=observer, progress=False)
    key = ('LatentState', 1)

    ntrials, L_kf, nrep = 3, 4, 3
    assert r['MMKF_PRIOR'][key].shape == (ntrials, L_kf, nrep)
    assert r['MMKF_LIKE'][key].shape  == (ntrials, L_kf, nrep)
    assert r['MMKF_POST'][key].shape  == (ntrials, L_kf, nrep)
    assert r['MMKF_XT'][key].shape    == (ntrials, nrep)


def test_no_mmkf_when_observer_absent(base_par):
    task = FakeTask([1.0])
    models = [('LatentState', make_agent_fn())]
    r = run_example(base_par, task, models, nrep=1, cases=[1],
                    observer=None, progress=False)

    assert r['MMKF_PRIOR'] == {}
    assert r['MMKF_LIKE']  == {}
    assert r['MMKF_POST']  == {}
    assert r['MMKF_XT']    == {}


def test_multiple_models_and_cases(base_par):
    task = FakeTask([1.0, 0.0])
    models = [('LatentState', make_agent_fn()), ('RW', make_agent_fn())]
    r = run_example(base_par, task, models, nrep=2, cases=[1, 2], progress=False)

    for name in ('LatentState', 'RW'):
        for case in (1, 2):
            key = (name, case)
            assert key in r['MU']
            assert key in r['LSB']
            assert key in r['V']
            assert key in r['par_by_key']


def test_xt_populated_when_agent_has_x_t(base_par):
    def agent_with_x_t(par, agent, world):
        a = make_agent_fn()(par, agent, world)
        if agent is not None:
            a['x_t'] = int(par['t']) % 2
        return a

    task = FakeTask([1.0, 1.0, 1.0, 1.0])
    models = [('LatentState', agent_with_x_t)]
    r = run_example(base_par, task, models, nrep=2, cases=[1], progress=False)

    key = ('LatentState', 1)
    assert key in r['XT']
    assert r['XT'][key].shape == (4, 2)   # (T, nrep)


# =============================================================================
# 4. Deepcopy isolation and par_by_key
# =============================================================================

def test_deepcopy_isolates_cases(base_par):
    """par['case'] and par['t'] must not leak across cases."""
    task = FakeTask([1.0, 0.0, 1.0])   # ntrials = 3
    models = [('LatentState', make_agent_fn())]

    r = run_example(base_par, task, models, nrep=1, cases=[1, 2], progress=False)

    for case in (1, 2):
        par = r['par_by_key'][('LatentState', case)]
        assert par['case'] == case
        assert par['modelname'] == 'LatentState'
        assert par['t'] == 3        # final trial index, per case
        assert par['Lmax'] == par['ls']['ncop'] * par['L']


def test_par_by_key_has_lmax(base_par):
    base_par['L'] = 2
    base_par['ls']['ncop'] = 3
    task = FakeTask([1.0])
    # Fake must allocate L*ncop states to match the real LatentState contract
    n_states = base_par['L'] * base_par['ls']['ncop']
    models = [('LatentState', make_agent_fn(n_states=n_states))]
    r = run_example(base_par, task, models, nrep=1, cases=[1], progress=False)
    assert r['par_by_key'][('LatentState', 1)]['Lmax'] == 6


# =============================================================================
# 5. Output serialization and progress printing
# =============================================================================

def test_save_results_writes_pickle(tmp_path, base_par):
    task = FakeTask([1.0])
    models = [('LatentState', make_agent_fn())]
    r = run_example(base_par, task, models, nrep=1, cases=[1], progress=False)

    path = save_results(r, 'test_example', out_dir=str(tmp_path))
    assert path.exists()
    with open(path, 'rb') as f:
        loaded = pickle.load(f)
    assert 'MU' in loaded
    assert ('LatentState', 1) in loaded['MU']


def test_progress_output(base_par, capsys):
    task = FakeTask([1.0] * 10)
    models = [('TestModel', make_agent_fn())]
    run_example(base_par, task, models, nrep=10, cases=[1], progress=True)

    captured = capsys.readouterr()
    assert 'Running ... TestModel' in captured.out
    assert 'Run #10' in captured.out


def test_no_progress_output(base_par, capsys):
    task = FakeTask([1.0])
    models = [('TestModel', make_agent_fn())]
    run_example(base_par, task, models, nrep=1, cases=[1], progress=False)

    captured = capsys.readouterr()
    assert 'Running' not in captured.out
    assert 'Run #' not in captured.out
