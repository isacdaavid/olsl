"""simulation/run_example.py — batch simulation across models, cases, replicates.

Literal translation of MATLAB ``RunExample.m``. Deviations from MATLAB
(forced by the absence of ``eval`` and by Python's lack of cell arrays):

  - Task dispatch: MATLAB passes a task *name* (string) and calls it via
    ``eval``. Here ``example`` is the task module itself (exposing
    ``.initialize`` and ``.randomize``). Callers resolve the name to a
    module once, at the call site.
  - Model dispatch: MATLAB passes ``mdls`` as a cell of strings and calls
    each via ``eval``. Here ``models`` is an iterable of either
    ``(name, callable)`` pairs or bare callables (name taken from
    ``__name__``).
  - Result storage: MATLAB uses nested cell arrays with ``(:,:,r)`` triple
    indexing. Here we return a dict keyed by ``(model_name, case)``; each
    entry has ``'mean'`` and ``'std'`` (or, for XT / MMKF_*, the raw
    replicate array ``(T, nrep)``, matching MATLAB's "store raw" behavior).
  - Save: MATLAB calls ``save('Results_' + exnme)`` at the end. Here we
    return the results dict; ``save_results`` is provided as a helper.
"""

import copy
import pickle
from pathlib import Path

import numpy as np

from .run_task import run_task


def run_example(par, example, models, nrep, cases, observer=None,
                progress=True):
    """Run one simulated example across models, cases, and replicates.

    Input:
        par      : dict of parameters. Deep-copied per (model, case).
        example  : task module exposing ``initialize(par)`` and
                   ``randomize(par)``. Replaces MATLAB's ``exnme`` string.
        models   : iterable of ``(name, callable)`` pairs or bare
                   callables. Replaces MATLAB's ``mdls`` cell of strings.
        nrep     : number of simulation replicates.
        cases    : iterable of case identifiers (typically ints).
        observer : optional ``(par, mmkf, agent, world) -> mmkf`` callable.
        progress : if True, print per-model / every-10-rep progress.

    Output:
        dict with keys:
            'MU', 'LSB', 'V'          -> each maps (name, case) ->
                                         {'mean': (T, K), 'std': (T, K)}
            'XT'                      -> maps (name, case) -> (T, nrep)
            'MMKF_PRIOR', 'MMKF_LIKE',
            'MMKF_POST', 'MMKF_XT'    -> maps (name, case) -> (T, L, nrep)
                                         or (T, nrep) for MMKF_XT
            'par_by_key'              -> maps (name, case) -> final par
    """
    results = {
        'MU':         {},
        'LSB':        {},
        'V':          {},
        'XT':         {},
        'MMKF_PRIOR': {},
        'MMKF_LIKE':  {},
        'MMKF_POST':  {},
        'MMKF_XT':    {},
        'par_by_key': {},
    }

    for entry in models:
        if isinstance(entry, tuple):
            model_name, approach = entry
        else:
            model_name = entry.__name__
            approach   = entry

        if progress:
            print(f"Running ... {model_name}")

        for case in cases:
            par_run = copy.deepcopy(par)
            par_run['case']      = case
            par_run['modelname'] = model_name
            par_run['Lmax']      = par_run['ls']['ncop'] * par_run['L']

            mu_list         = []
            v_list          = []
            lsb_list        = []
            x_t_list        = []
            mmkf_prior_list = []
            mmkf_like_list  = []
            mmkf_post_list  = []
            mmkf_xt_list    = []

            for r in range(nrep):
                if progress and (r + 1) % 10 == 0:
                    print(f"   Run #{r + 1}")

                out = run_task(par_run, example, approach, observer=observer)
                par_run = out['par']
                par_run['Lmax'] = par_run['ls']['ncop'] * par_run['L']

                v_proc, lsb_proc = _postprocess(
                    model_name, out['V'], out['lsb'], par_run
                )
                mu_list.append(out['mu'])
                v_list.append(v_proc)
                lsb_list.append(lsb_proc)

                if (out['X_t_seq'] is not None
                    and out['X_t_seq'].size > 0
                    and not np.all(np.isnan(out['X_t_seq']))):
                    x_t_list.append(out['X_t_seq'])

                if out['MMKF_lsb_prior'] is not None:
                    mmkf_prior_list.append(out['MMKF_lsb_prior'])
                    mmkf_like_list.append(out['MMKF_likelihood'])
                    mmkf_post_list.append(out['MMKF_lsb_post'])
                    mmkf_xt_list.append(out['MMKF_X_t_est'])

            key = (model_name, case)
            results['par_by_key'][key] = par_run

            MU = np.stack(mu_list, axis=-1)
            results['MU'][key] = {
                'mean': np.mean(MU, axis=-1),
                'std':  np.std(MU,  axis=-1),
            }

            LSB = np.stack(lsb_list, axis=-1)
            results['LSB'][key] = {
                'mean': np.mean(LSB, axis=-1),
                'std':  np.std(LSB,  axis=-1),
            }

            V = np.stack(v_list, axis=-1)
            results['V'][key] = {
                'mean': np.mean(V, axis=-1),
                'std':  np.std(V,  axis=-1),
            }

            if x_t_list:
                results['XT'][key] = np.stack(x_t_list, axis=-1)

            if mmkf_prior_list:
                results['MMKF_PRIOR'][key] = np.stack(mmkf_prior_list, axis=-1)
                results['MMKF_LIKE'][key]  = np.stack(mmkf_like_list,  axis=-1)
                results['MMKF_POST'][key]  = np.stack(mmkf_post_list,  axis=-1)
                results['MMKF_XT'][key]    = np.stack(mmkf_xt_list,    axis=-1)

    return results

def _postprocess(model_name, V, LSB, par):
    """Model-specific post-processing of V and LSB (see RunExample.m).

    LatentState:  weighted-average V across the ``ncop`` copies of each
                  hypothesis, weighted by the copy's belief.
    Gershman2017: weighted-average V across Gershman's K states.
    InfiniteMixture / Redish2007: pass through.
    Unknown:      V unchanged; LSB replaced with zeros (matches MATLAB).
    """
    T = V.shape[0]
    D = par['D']
    C = par['C']

    if 'LatentState' in model_name:
        # --- LatentState-only keys ---
        L    = par['L']
        ncop = par['ls']['ncop']

        expected_cols = D * C * L * ncop

        if V.shape[1] != expected_cols:
            raise ValueError(
                f"_postprocess got V with {V.shape[1]} columns, "
                f"but L*ncop*D*C = {expected_cols}. "
                f"Check that the agent allocates V of shape "
                f"(D={D}, C={C}, L*ncop={L*ncop})."
            )

        expected_lsb_cols = L * ncop

        if LSB.shape[1] != expected_lsb_cols:
            raise ValueError(
                f"_postprocess got LSB with {LSB.shape[1]} columns, "
                f"but L*ncop = {expected_lsb_cols}."
            )

        # run_task returns V flattened with order='F' from (D, C, L*ncop),
        # so the linear index within a trial is
        #     k = dc + l1*D*C + l2*par.L*D*C   (dc fastest, then l1, then l2)
        # where l = l1 + l2*par.L (0-based) and dc = d + c*D.
        #
        # Reshape to (T, ncop, L, D*C) row-major puts the fastest axis last,
        # i.e., (t, l2, l1, dc) — transpose to (t, dc, l1, l2) to match MATLAB.
        v_5d   = V.reshape(T, ncop, L, D * C).transpose(0, 3, 2, 1)  # (T, D*C, L, ncop)
        lsb_4d = LSB.reshape(T, ncop, L).transpose(0, 2, 1)          # (T, L, ncop)

        numerator = (v_5d * lsb_4d[:, None, :, :]).sum(axis=-1)      # (T, D*C, L)
        nrm       = lsb_4d.sum(axis=-1)                              # (T, L)

        with np.errstate(divide='ignore', invalid='ignore'):
            v_avg = numerator / nrm[:, None, :]                      # (T, D*C, L)

        # Flatten to (T, D*C*L) with dc fastest, then l1 (matches MATLAB).
        v_proc = v_avg.transpose(0, 2, 1).reshape(T, D * C * L)
        return v_proc, LSB

    elif 'Gershman2017' in model_name:
        K = par['gershman']['K']
        v_3d = V.reshape(T, K, D * C).transpose(0, 2, 1)   # (T, D*C, K)
        v_proc = (v_3d * LSB[:, None, :]).sum(axis=-1)      # (T, D*C)
        return v_proc, LSB

    elif 'InfiniteMixture' in model_name or 'Redish2007' in model_name:
        return V, LSB

    else:
        # MATLAB does LSB0(:,:,r) = 0 -> scalar; we return a (T, 1) zeros
        # array. Downstream code skips LSB plotting for these models anyway.
        return V, np.zeros((T, 1))


def save_results(results, exnme, out_dir='.'):
    """Replacement for MATLAB's ``save(['Results_', exnme])``."""
    path = Path(out_dir) / f'Results_{exnme}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    return path
