"""Online latent-state (approximate Bayesian) reinforcement learning.

Python class-based port of the MATLAB implementation accompanying

Cochran, A. L., & Cisler, J. M. (2019).
A flexible and generalizable model of online latent-state learning.
PLOS Computational Biology, 15(9), e1007331.
https://doi.org/10.1371/journal.pcbi.1007331

The agent keeps the **full history** of every persistent variable along a
leading time axis. Slot ``0`` is the initial state; slot ``k`` is the state
after ``k`` completed trials. ``self.step`` is the index of the last
completed state, so:

    agent.V[0:agent.step + 1]   # full history
    agent.V[agent.step]         # current state

State
    V       : (T, REWARDS, CUES, LATENT_TOTAL)       associative strengths
    B       : (T, CUES, CUES, REWARDS, LATENT_TOTAL) associability covariance
    lsb     : (T, LATENT_TOTAL)                      latent-state beliefs
    sigmasq : (T,)                                   running reward variance
    Lmax    : (T,)                                   number of active latent states
    q       : (T,)                                   change-point statistic

Per-trial record
    arm               : (T,)                       chosen arm index
    reward            : (T, REWARDS)               realized reward
    expected_reward_0 : (T, REWARDS, ARMS)         pre-selection value per arm
    expected_reward   : (T, REWARDS, LATENT_TOTAL) per-state value for chosen arm

``LATENT_TOTAL = par.LATENT * par.LATENT_MAX`` throughout.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from olsl.common.math import SoftMaxChoice


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class Parameters:
    """Hyperparameters and (temporary) task-schedule fields.

    The ``t``, ``ntrials``, ``time`` and ``context`` fields are currently
    carried here because the model still reads them during ``update``.
    They will move to :class:`Task` in the next refactor step.
    """

    # --- dimensions ---
    REWARDS: int = 1
    """Number of reward dimensions."""

    CUES: int = 1
    """Number of cues."""

    ARMS: int = 1
    """Number of arms / actions."""

    LATENT: int = 3
    """Number of latent-state hypotheses per copy."""

    LATENT_MAX: int = 1
    """Copies per latent state (capacity of the change-point process)."""

    # --- learning dynamics ---
    alpha0: float = 0.1
    """Associability scale."""

    alpha1: float = 0.1
    """Variance-tracking rate."""

    alpha2: float = 0.1
    """Covariance (B) tracking rate."""

    gamma: float = 0.1
    """Belief stickiness / transition rate."""

    delta: float = 0.1
    """Change-point offset."""

    eta: float = 0.1
    """Change-point threshold."""

    sigma0: float = 1.0
    """Initial reward standard deviation."""

    chi: int = 10
    """Maximum rumination iterations."""

    center: float = 0.0
    """Reward baseline subtracted from ``world['reward']``."""

    beta: float = 1.0
    """Softmax inverse temperature (consumed by ``SoftMaxChoice``)."""

    # --- optional initialization overrides ---
    initialex: Optional[np.ndarray] = None
    """Initial associative strengths, shape ``(REWARDS, CUES, LATENT)``."""

    lsb0: Optional[np.ndarray] = None
    """Initial beliefs over the first ``LATENT`` states, shape ``(LATENT,)``."""

    initialV: Optional[dict] = None
    """Full saved agent state to restore into slot 0 (see ``save_state``)."""

    # --- task-schedule fields (TODO: move to Task) ---
    t: int = 1
    """1-based trial index (will move to Task)."""

    ntrials: int = 1000
    """Total number of trials (allocation and schedule hint)."""

    time: Optional[list] = None
    """Per-trial timestamps, length ``ntrials`` (will move to Task)."""

    context: Optional[list] = None
    """1-based trial indices at which beliefs reset (will move to Task)."""

    @property
    def LATENT_TOTAL(self) -> int:
        """Total number of latent-state slots, ``LATENT * LATENT_MAX``."""
        return self.LATENT * self.LATENT_MAX


# ---------------------------------------------------------------------------
# Array specifications (name, shape, dtype)
# ---------------------------------------------------------------------------

_ARRAY_SPECS = [
    ("V",       lambda p: (p.REWARDS, p.CUES, p.LATENT_TOTAL),         np.float64),
    ("B",       lambda p: (p.CUES, p.CUES, p.REWARDS, p.LATENT_TOTAL),    np.float64),
    ("lsb",     lambda p: (p.LATENT_TOTAL,),                   np.float64),
    ("sigmasq", lambda p: (),                             np.float64),
    ("Lmax",    lambda p: (),                             np.int64),
    ("q",       lambda p: (),                             np.float64),
    ("arm",     lambda p: (),                             np.int64),
    ("reward",     lambda p: (p.REWARDS,),                         np.float64),
    ("expected_reward_0",     lambda p: (p.REWARDS, p.ARMS),                     np.float64),
    ("expected_reward",      lambda p: (p.REWARDS, p.LATENT_TOTAL),               np.float64),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_2d(reward: np.ndarray, n_rewards: int, n_arms: int) -> np.ndarray:
    """Coerce a scalar / 1-D / 2-D reward into shape ``(REWARDS, ARMS)``.

    Args:
        reward: Reward array of arbitrary rank ≤ 2.
        n_rewards: Target first dimension.
        n_arms: Target second dimension.

    Returns:
        Array of shape ``(REWARDS, ARMS)``.
    """
    if np.isscalar(reward) or np.ndim(reward) == 0:
        return np.array([[float(reward)]])
    reward = np.asarray(reward)
    if reward.ndim == 1:
        if len(reward) == n_arms:
            return reward.reshape(1, -1)
        return reward.reshape(-1, 1)
    return reward


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LatentState:
    """Approximate Bayesian latent-state reinforcement learning agent.

    Every persistent variable is stored as a history array with a leading
    time axis. ``self.step`` indexes the last completed trial; the next
    call to :meth:`update` writes into slot ``self.step + 1``.

    Args:
        parameters: Hyperparameters. If ``None``, defaults are used.

    Attributes:
        par: The :class:`Parameters` instance in use.
        step: Index of the last completed trial (``0`` = initial state).
        V, B, lsb, sigmasq, Lmax, q: Persistent history arrays.
        arm, reward, expected_reward_0, expected_reward: Per-trial record arrays.
    """

    def __init__(self, parameters: Optional[Parameters] = None) -> None:
        self.par: Parameters = parameters or Parameters()
        p = self.par

        # Initial capacity: enough for the declared schedule, with a floor.
        self._capacity: int = max(p.ntrials + 1, 128)
        for name, shape_fn, dtype in _ARRAY_SPECS:
            arr = np.full((self._capacity,) + shape_fn(p), np.nan, dtype=dtype)
            setattr(self, name, arr)

        self._initialize_slot(0)
        self.step: int = 0

    # -- slot initialization / growth ------------------------------------

    def _initialize_slot(self, s: int) -> None:
        """Populate slot ``s`` with the initial (or restored) state.

        Args:
            s: History index to write.
        """
        p = self.par
        latent_total = p.LATENT_TOTAL

        if p.initialV is not None:
            init = p.initialV
            self.V[s] = init["V"]
            self.B[s] = init["B"]
            self.lsb[s] = init["lsb"]
            self.Lmax[s] = init["Lmax"]
            self.q[s] = init["q"]
            self.sigmasq[s] = init["sigmasq"]
            return

        # Associative strengths
        self.V[s] = p.initialex if p.initialex is not None else 0.0

        # Beliefs
        self.lsb[s] = 0.0
        if p.lsb0 is not None:
            self.lsb[s, : p.LATENT] = p.lsb0
        else:
            self.lsb[s, : p.LATENT] = 1.0 / p.LATENT

        # Covariance: identity per (dim, state)
        self.B[s] = 0.0
        for d in range(p.REWARDS):
            for l in range(latent_total):
                self.B[s, :, :, d, l] = np.eye(p.CUES)

        # Scalars
        self.Lmax[s] = 1
        self.q[s] = 0.0
        self.sigmasq[s] = p.sigma0 ** 2

        # Per-trial record: unset
        self.arm[s] = -1
        self.reward[s] = np.nan
        self.expected_reward_0[s] = np.nan
        self.expected_reward[s] = np.nan

    def _grow_if_needed(self, required_len: int) -> None:
        """Extend history arrays so slot ``required_len - 1`` is addressable.

        Args:
            required_len: Minimum number of history slots required.
        """
        if required_len <= self._capacity:
            return
        new_capacity = max(required_len, 2 * self._capacity)
        pad = new_capacity - self._capacity
        p = self.par
        for name, shape_fn, dtype in _ARRAY_SPECS:
            old = getattr(self, name)
            pad_arr = np.full((pad,) + shape_fn(p), np.nan, dtype=dtype)
            setattr(self, name, np.concatenate([old, pad_arr], axis=0))
        self._capacity = new_capacity

    # -- public API ------------------------------------------------------

    def update(self, world: dict, t: Optional[int] = None) -> "LatentState":
        """Run one trial and append the new state to history.

        Args:
            world: Observation dict with keys:

                * ``'c_vec'`` — cues, shape ``(REWARDS, CUES, LATENT_TOTAL, ARMS)``.
                * ``'reward'`` — reward, shape ``(REWARDS, ARMS)`` or coercible.

            t: Optional 1-based trial index. If provided, overwrites
                ``self.par.t`` (transitional — will move to Task).

        Returns:
            ``self`` (for chaining).
        """
        p = self.par
        if t is not None:
            p.t = t

        s = self.step
        s1 = s + 1
        self._grow_if_needed(s1 + 1)

        # Copy persistent state forward (in-place at s1)
        for name in ("V", "B", "lsb", "sigmasq", "Lmax", "q"):
            getattr(self, name)[s1] = getattr(self, name)[s]

        # Reset per-trial record
        self.arm[s1] = -1
        self.reward[s1] = np.nan
        self.expected_reward_0[s1] = np.nan
        self.expected_reward[s1] = np.nan

        # Inputs
        reward = _as_2d(np.asarray(world["reward"]) - p.center, p.REWARDS, p.ARMS)
        c_vec = np.asarray(world["c_vec"])            # (REWARDS, CUES, LATENT_TOTAL, ARMS)

        # (1) Context / temporal shift on beliefs
        self._context_shift(s1)

        # (2) Predictions
        V_s1 = self.V[s1]
        expected_reward = np.reshape(
            np.sum(V_s1[..., None] * c_vec, axis=1),
            (p.REWARDS, p.LATENT_TOTAL, p.ARMS),
        )
        expected_reward_0 = np.reshape(
            np.sum(expected_reward * self.lsb[s1][None, :, None], axis=1),
            (p.REWARDS, p.ARMS),
        )
        self.expected_reward_0[s1] = expected_reward_0

        # (3) Arm selection
        arm, _ = SoftMaxChoice(expected_reward_0, self._legacy_par())
        self.arm[s1] = arm

        # (4) Slice by chosen arm
        reward_sl = reward[:, arm]                          # (REWARDS,)
        expected_reward_sl = expected_reward[:, :, arm]                         # (REWARDS, LATENT_TOTAL)
        c_vec_sl = c_vec[:, :, :, arm]                # (REWARDS, CUES, LATENT_TOTAL)
        self.reward[s1] = reward_sl
        self.expected_reward[s1] = expected_reward_sl

        # (5) Prior variance (before this trial's update)
        sigmasq0 = self.sigmasq[s]

        # (6) Belief update, state creation, associative update
        LL = self._log_likelihood(s1, reward_sl, expected_reward_sl, sigmasq0)
        self.lsb[s1] = self._update_beliefs(s1, LL)
        self._maybe_add_state(s1, c_vec_sl, reward_sl, LL)
        self._update_v_b(s1, c_vec_sl, reward_sl, sigmasq0)

        self.step = s1
        return self

    def reset(self) -> "LatentState":
        """Reset to the initial state, keeping the allocated capacity.

        Returns:
            ``self``.
        """
        self._initialize_slot(0)
        self.step = 0
        return self

    def save_state(self) -> dict:
        """Snapshot the current state for ``Parameters.initialV``.

        Returns:
            Dict with copies of ``V``, ``B``, ``lsb``, ``Lmax``, ``q``
            and ``sigmasq`` at ``self.step``.
        """
        s = self.step
        return {
            "V": self.V[s].copy(),
            "B": self.B[s].copy(),
            "lsb": self.lsb[s].copy(),
            "Lmax": int(self.Lmax[s]),
            "q": float(self.q[s]),
            "sigmasq": float(self.sigmasq[s]),
        }

    # -- internal: model pieces -----------------------------------------

    def _context_shift(self, s: int) -> None:
        """Apply temporal decay and/or explicit context reset to beliefs.

        Args:
            s: History slot whose beliefs are updated in place.
        """
        p = self.par
        lsb = self.lsb[s]                             # view into history
        Lmax = int(self.Lmax[s])

        # Temporal decay toward uniform (per elapsed time between trials)
        if p.t != 1 and p.time is not None:
            elapsed = p.time[p.t - 1] - p.time[p.t - 2]
            prob_stay = (1 - p.gamma) ** (elapsed - 1)
            n = Lmax * p.LATENT
            lsb[:n] = lsb[:n] * prob_stay + (1 - prob_stay) / n

        # Explicit context reset: uniform over active states, normalized
        if p.context is not None and p.t in p.context:
            lsb[: p.LATENT * Lmax] = 1.0
            lsb[:] = lsb / np.sum(lsb)

    def _log_likelihood(
        self,
        s: int,
        reward: np.ndarray,
        expected_reward: np.ndarray,
        sigmasq: float,
    ) -> np.ndarray:
        """Per-state Gaussian log-likelihood of the observed reward.

        Args:
            s: History slot whose ``Lmax`` is read.
            reward: Realized reward, shape ``(REWARDS,)``.
            expected_reward: Per-state predictions for the chosen arm, shape ``(REWARDS, LATENT_TOTAL)``.
            sigmasq: Variance used to scale squared error.

        Returns:
            Log-likelihood vector of shape ``(LATENT_TOTAL,)``. Active states
            get the data term; the next inactive block is neutral (0);
            everything beyond is ``-inf``.
        """
        p = self.par
        latent_total = p.LATENT_TOTAL
        Lmax = int(self.Lmax[s])

        LL = np.zeros(latent_total)
        ix_limit = p.LATENT * (Lmax + 1)
        LL[ix_limit:] = -np.inf
        ix = p.LATENT * Lmax
        LL[:ix] = np.sum(
            -(reward[:, None] - expected_reward[:, :ix]) ** 2 / (2 * sigmasq),
            axis=0,
        )
        return LL

    def _update_beliefs(self, s: int, LL: np.ndarray) -> np.ndarray:
        """Prior × likelihood, renormalized over active states.

        Args:
            s: History slot whose prior beliefs are read.
            LL: Log-likelihood vector from :meth:`_log_likelihood`.

        Returns:
            New belief vector of shape ``(LATENT_TOTAL,)``.
        """
        p = self.par
        Lmax = int(self.Lmax[s])
        ix = Lmax * p.LATENT

        prior = (
            (1 - p.gamma) * self.lsb[s][:ix] + p.gamma / (Lmax * p.LATENT)
        )
        scaled_lik = np.exp(LL[:ix] - np.max(LL[:ix]))

        out = np.zeros(p.LATENT_TOTAL)
        out[:ix] = prior * scaled_lik
        return out / np.sum(out)

    def _maybe_add_state(
        self,
        s: int,
        c_vec: np.ndarray,
        reward: np.ndarray,
        LL: np.ndarray,
    ) -> None:
        """Change-point test; spawn a new latent state if ``q`` crosses ``eta``.

        Faithful port of the MATLAB original. Note that the freshly
        activated state's associations are written to slot
        ``new_Lmax - 1`` of the last axis — this coincides with the first
        slot of the new block only when ``LATENT == 1``. Behaviour is preserved
        as-is for compatibility with the original implementation.

        Args:
            s: History slot being updated in place.
            c_vec: Cues for the chosen arm, shape ``(REWARDS, CUES, LATENT_TOTAL)``.
            reward: Realized reward, shape ``(REWARDS,)``.
            LL: Log-likelihood vector (see :meth:`_log_likelihood`).
        """
        p = self.par
        LATENT = p.LATENT
        Lmax = int(self.Lmax[s])
        if Lmax >= p.LATENT_MAX:
            return

        ix_active = Lmax * LATENT
        ix_new_end = (Lmax + 1) * LATENT

        mx = np.max(np.exp(LL[:ix_new_end]))

        # Scaled likelihood if we keep the current active set
        stay = (
            (1 - p.gamma) * self.lsb[s][:ix_active] + p.gamma / (Lmax * LATENT)
        ) * np.exp(LL[:ix_active] - mx)
        l0 = np.sum(stay)

        # Scaled likelihood if we spawn a new state
        l1 = np.sum((1.0 / LATENT) * np.exp(LL[ix_active:ix_new_end] - mx))

        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log(l1) - np.log(l0)
        if np.isnan(log_ratio):
            log_ratio = 0.0

        new_q = max(self.q[s] + log_ratio - p.delta, 0.0)
        if new_q < p.eta:
            self.q[s] = new_q
            return

        # --- spawn ---
        new_Lmax = Lmax + 1
        self.Lmax[s] = new_Lmax
        self.q[s] = 0.0

        tmp = np.zeros(p.LATENT_TOTAL)
        tmp[:ix_new_end] = (
            (1 - p.gamma) * self.lsb[s][:ix_new_end]
            + p.gamma / (new_Lmax * LATENT)
        ) * np.exp(LL[:ix_new_end])

        # Initialize associations for the freshly activated state
        c_current = c_vec[:, :, new_Lmax - 1]                 # (REWARDS, CUES)
        Vnew = c_current * reward[:, None] / np.sum(
            c_current ** 2, axis=1, keepdims=True
        )
        self.V[s, :, :, new_Lmax - 1] = Vnew

        self.lsb[s] = tmp / np.sum(tmp)

    def _update_v_b(
        self,
        s: int,
        c_vec: np.ndarray,
        reward: np.ndarray,
        sigmasq0: float,
    ) -> None:
        """Rescorla–Wagner update of ``V`` and ``B``, plus variance tracking.

        Args:
            s: History slot being updated in place.
            c_vec: Cues for the chosen arm, shape ``(REWARDS, CUES, LATENT_TOTAL)``.
            reward: Realized reward, shape ``(REWARDS,)``.
            sigmasq0: Prior variance (before this trial's update); used
                both inside and at the end of the EMA step.
        """
        p = self.par
        V = self.V[s]                                 # view
        B = self.B[s]                                 # view
        lsb = self.lsb[s]                             # view
        Lmax = int(self.Lmax[s])
        sigmasq = float(self.sigmasq[s])

        # Rumination iterations (fewer on the last trial)
        if p.t == p.ntrials or p.time is None:
            nIter = 1
        else:
            nIter = min(p.chi, round(p.time[p.t] - p.time[p.t - 1]))

        expected_reward = np.reshape(np.sum(V * c_vec, axis=1), (p.REWARDS, p.LATENT_TOTAL))

        for d in range(p.REWARDS):
            for l1 in range(p.LATENT):
                for l2 in range(Lmax):
                    l = l2 * p.LATENT + l1

                    # Exponentially weighted covariance
                    c_dl = c_vec[d, :, l]
                    c_outer = np.outer(c_dl, c_dl)
                    B[:, :, d, l] = B[:, :, d, l] + p.alpha2 * (
                        lsb[l] * c_outer - B[:, :, d, l]
                    )

                    # Associability: alpha0 * lsb * B^{-1} c
                    tmp = p.alpha0 * lsb[l] * np.linalg.solve(
                        B[:, :, d, l], c_dl
                    )

                    # Rescorla–Wagner
                    TD = reward[d] - expected_reward[d, l]
                    V[d, :, l] = V[d, :, l] + tmp * TD
                    sigmasq = sigmasq + p.alpha1 * lsb[l] * TD ** 2

                    # Rumination
                    for _ in range(nIter - 1):
                        expected_reward = np.reshape(
                            np.sum(V * c_vec, axis=1),
                            (p.REWARDS, p.LATENT_TOTAL),
                        )
                        TD = reward[d] - expected_reward[d, l]
                        V[d, :, l] = V[d, :, l] + tmp * TD

        # Finish variance EMA step
        sigmasq = sigmasq - p.alpha1 * sigmasq0
        self.sigmasq[s] = sigmasq

    # -- compatibility shim (removed in task refactor) ------------------

    def _legacy_par(self) -> dict:
        """Rebuild the flat+``ls`` dict shape that ``SoftMaxChoice`` expects.

        Returns:
            Dict matching the original ``par`` layout.
        """
        p = self.par
        return {
            # Legacy keys retained for SoftMaxChoice compatibility.
            "D": p.REWARDS, "C": p.CUES, "A": p.ARMS, "L": p.LATENT,
            "t": p.t, "ntrials": p.ntrials,
            "time": p.time, "context": p.context,
            "center": p.center, "beta": p.beta,
            "ls": {
                "ncop": p.LATENT_MAX, "gamma": p.gamma, "delta": p.delta,
                "eta": p.eta, "sigma0": p.sigma0, "chi": p.chi,
                "alpha0": p.alpha0, "alpha1": p.alpha1, "alpha2": p.alpha2,
            },
        }

    def __repr__(self) -> str:
        p = self.par
        return (
            f"LatentState(REWARDS={p.REWARDS}, CUES={p.CUES}, ARMS={p.ARMS}, "
            f"LATENT={p.LATENT}, LATENT_MAX={p.LATENT_MAX}, "
            f"step={self.step}/{self._capacity - 1})"
        )
