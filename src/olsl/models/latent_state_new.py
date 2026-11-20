"""Online latent-state (approximate Bayesian) reinforcement learning.

Python class-based port of the MATLAB implementation accompanying

Cochran, A. L., & Cisler, J. M. (2019).
A flexible and generalizable model of online latent-state learning.
PLOS Computational Biology, 15(9), e1007331.
https://doi.org/10.1371/journal.pcbi.1007331

The agent keeps the **full history** of every persistent variable. Slot ``0``
is the initial state; slot ``t`` is the state after ``t`` completed trials.
``self.step`` is the index of the last completed state, so:

    agent.V[0:agent.step + 1]   # list of per-trial arrays
    agent.V[agent.step]         # current associative strengths

The number of latent states is dynamic. A change-point event adds one state,
so the latent-state dimension can differ between history slots.

State
    V       : list[(REWARDS, CUES, L_t)]       associative strengths
    B       : list[(CUES, CUES, REWARDS, L_t)] associability covariance
    lsb     : list[(L_t,)]                     latent-state beliefs
    sigmasq : list[float]                      running reward variance
    q       : list[float]                      change-point statistic

Per-trial record
    arm               : list[int]             chosen arm index
    reward            : list[(REWARDS,)]      realized reward
    expected_reward_0 : list[(REWARDS, ARMS)]  pre-selection value per arm
    expected_reward   : list[(REWARDS, L_before)] predictions for chosen arm,
                                               before any state is added

Here ``L_t`` is the number of latent states represented at history slot ``t``.
The cue representation supplied by the task is shared across latent states:
``world['c_vec']`` has shape ``(REWARDS, CUES, ARMS)``.
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
    """Initial associative strengths, shape ``(REWARDS, CUES, L_0)``."""

    lsb0: Optional[np.ndarray] = None
    """Initial beliefs over ``L_0`` states, shape ``(L_0,)``.

    If omitted, one initial state with belief 1 is used. When ``initialex`` is
    supplied without ``lsb0``, its final dimension determines ``L_0`` and the
    initial beliefs are uniform over those states.
    """

    initialV: Optional[dict] = None
    """Full saved agent state to restore into the initial history slot."""

    # --- task-schedule fields (TODO: move to Task) ---
    t: int = 1
    """1-based trial index (will move to Task)."""

    ntrials: int = 1000
    """Total number of trials (allocation and schedule hint)."""

    time: Optional[list] = None
    """Per-trial timestamps, length ``ntrials`` (will move to Task)."""

    context: Optional[list] = None
    """1-based trial indices at which beliefs reset (will move to Task)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_2d(reward: np.ndarray, n_rewards: int, n_arms: int) -> np.ndarray:
    """Coerce a scalar / 1-D / 2-D reward into shape ``(REWARDS, ARMS)``.

    Args:
        reward: Reward array of arbitrary rank <= 2.
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

    Every trial-indexed variable is stored as a Python list with one entry
    per history slot. Array-valued entries are independent NumPy arrays;
    scalar-valued entries are Python scalars. This gives every history the
    same append-based convention, including variables whose shapes are fixed.

    ``self.step`` indexes the last completed trial; the next call to
    :meth:`update` appends slot ``self.step + 1``.

    Args:
        parameters: Hyperparameters. If ``None``, defaults are used.

    Attributes:
        par: The :class:`Parameters` instance in use.
        step: Index of the last completed trial (``0`` = initial state).
        V, B, lsb, expected_reward: Per-trial array histories.
        sigmasq, q, arm: Per-trial scalar histories.
        reward, expected_reward_0: Per-trial array histories.
        Lmax: Derived history of active latent-state counts, obtained from
            ``len(lsb[t])`` rather than stored independently.
    """

    def __init__(self, parameters: Optional[Parameters] = None) -> None:
        self.par: Parameters = parameters or Parameters()

        # Every model variable uses an append-only full-history list.
        self.V: list[np.ndarray] = []
        self.B: list[np.ndarray] = []
        self.lsb: list[np.ndarray] = []
        self.sigmasq: list[float] = []
        self.q: list[float] = []
        self.arm: list[int] = []
        self.reward: list[np.ndarray] = []
        self.expected_reward_0: list[np.ndarray] = []
        self.expected_reward: list[np.ndarray] = []

        self.step: int = 0
        self._initialize_history()


    @property
    def Lmax(self) -> list[int]:
        """Derived number of active latent states at each history slot."""
        return [len(beliefs) for beliefs in self.lsb]

    # -- initialization -------------------------------------------------


    def _initialize_history(self) -> None:
        """Create the initial state and initialize its history records."""
        p = self.par

        if p.initialV is not None:
            init = p.initialV
            lsb0 = np.asarray(init["lsb"], dtype=np.float64).copy()
            V0 = np.asarray(init["V"], dtype=np.float64).copy()
            B0 = np.asarray(init["B"], dtype=np.float64).copy()

            if lsb0.ndim != 1 or lsb0.size == 0:
                raise ValueError("initialV['lsb'] must be a non-empty 1-D array.")
            n_states = lsb0.size
            if V0.shape != (p.REWARDS, p.CUES, n_states):
                raise ValueError(
                    "initialV['V'] must have shape "
                    f"({p.REWARDS}, {p.CUES}, {n_states}); got {V0.shape}."
                )
            if B0.shape != (p.CUES, p.CUES, p.REWARDS, n_states):
                raise ValueError(
                    "initialV['B'] must have shape "
                    f"({p.CUES}, {p.CUES}, {p.REWARDS}, {n_states}); "
                    f"got {B0.shape}."
                )
            if "Lmax" in init and int(init["Lmax"]) != n_states:
                raise ValueError(
                    "initialV uses the previous fixed/block latent-state "
                    "allocation. Its stored Lmax does not match len(lsb); "
                    "create a state snapshot using this dynamic-state version."
                )

            q0 = float(init["q"])
            sigmasq0 = float(init["sigmasq"])
        else:
            if p.lsb0 is not None:
                lsb0 = np.asarray(p.lsb0, dtype=np.float64).copy()
                if lsb0.ndim != 1 or lsb0.size == 0:
                    raise ValueError("lsb0 must be a non-empty 1-D array.")
                n_states = lsb0.size
            elif p.initialex is not None:
                initialex = np.asarray(p.initialex, dtype=np.float64)
                if initialex.ndim != 3:
                    raise ValueError(
                        "initialex must have shape (REWARDS, CUES, L_0)."
                    )
                n_states = initialex.shape[-1]
                if n_states == 0:
                    raise ValueError("initialex must contain at least one state.")
                lsb0 = np.full(n_states, 1.0 / n_states, dtype=np.float64)
            else:
                n_states = 1
                lsb0 = np.ones(1, dtype=np.float64)

            # Associative strengths; initialex can also determine L_0.
            if p.initialex is None:
                V0 = np.zeros((p.REWARDS, p.CUES, n_states), dtype=np.float64)
            else:
                V0 = np.asarray(p.initialex, dtype=np.float64).copy()
                if V0.shape != (p.REWARDS, p.CUES, n_states):
                    raise ValueError(
                        "initialex must have shape "
                        f"({p.REWARDS}, {p.CUES}, {n_states}); got {V0.shape}."
                    )

            # Covariance: identity per reward dimension and latent state.
            B0 = np.zeros(
                (p.CUES, p.CUES, p.REWARDS, n_states), dtype=np.float64
            )
            for d in range(p.REWARDS):
                for state in range(n_states):
                    B0[:, :, d, state] = np.eye(p.CUES)

            q0 = 0.0
            sigmasq0 = p.sigma0 ** 2

        if p.lsb0 is not None and p.initialex is not None and p.initialV is None:
            if np.asarray(p.initialex).shape[-1] != lsb0.size:
                raise ValueError(
                    "The final dimension of initialex must match len(lsb0)."
                )

        # Reset all histories. Every variable has exactly one entry per slot.
        self.V = [V0.copy()]
        self.B = [B0.copy()]
        self.lsb = [lsb0.copy()]
        self.sigmasq = [float(sigmasq0)]
        self.q = [float(q0)]
        self.arm = [-1]
        self.reward = [np.full(p.REWARDS, np.nan, dtype=np.float64)]
        self.expected_reward_0 = [
            np.full((p.REWARDS, p.ARMS), np.nan, dtype=np.float64)
        ]
        self.expected_reward = [
            np.full((p.REWARDS, n_states), np.nan, dtype=np.float64)
        ]

    # -- public API ------------------------------------------------------

    def update(self, world: dict, t: Optional[int] = None) -> "LatentState":
        """Run one trial and append the new state to history.

        Args:
            world: Observation dict with keys:

                * ``'c_vec'`` -- cues, shape ``(REWARDS, CUES, ARMS)``.
                  The same cue representation is used for every latent state.
                * ``'reward'`` -- reward, shape ``(REWARDS, ARMS)`` or coercible.

            t: Optional 1-based trial index. If provided, overwrites
                ``self.par.t`` (transitional -- will move to Task).

        Returns:
            ``self`` (for chaining).
        """
        p = self.par
        if t is not None:
            p.t = t

        # Inputs
        reward = _as_2d(
            np.asarray(world["reward"]) - p.center, p.REWARDS, p.ARMS
        )
        c_vec = np.asarray(world["c_vec"], dtype=np.float64)
        expected_cue_shape = (p.REWARDS, p.CUES, p.ARMS)
        if c_vec.shape != expected_cue_shape:
            raise ValueError(
                f"world['c_vec'] must have shape {expected_cue_shape}; "
                f"got {c_vec.shape}."
            )
        expected_reward_shape = (p.REWARDS, p.ARMS)
        if reward.shape != expected_reward_shape:
            raise ValueError(
                f"world['reward'] must resolve to shape {expected_reward_shape}; "
                f"got {reward.shape}."
            )

        s = self.step
        s1 = s + 1

        # Append independent copies / values for every history variable.
        self.V.append(self.V[s].copy())
        self.B.append(self.B[s].copy())
        self.lsb.append(self.lsb[s].copy())
        self.sigmasq.append(float(self.sigmasq[s]))
        self.q.append(float(self.q[s]))
        self.arm.append(-1)
        self.reward.append(np.full(p.REWARDS, np.nan, dtype=np.float64))
        self.expected_reward_0.append(
            np.full((p.REWARDS, p.ARMS), np.nan, dtype=np.float64)
        )
        self.expected_reward.append(
            np.full(
                (p.REWARDS, self.lsb[s1].size),
                np.nan,
                dtype=np.float64,
            )
        )

        # (1) Context / temporal shift on beliefs.
        self._context_shift(s1)

        # (2) Predictions under each latent state and each available arm.
        V_s1 = self.V[s1]
        # Value function per arm, latent state: (REWARDS, K, ARMS)
        expected_reward_all = np.sum(
            V_s1[..., None] * c_vec[:, :, None, :], axis=1
        )
        # Value function per arm: (REWARDS, ARMS)
        # We use lsb[None, :, None] to broadcast (R, K, A) * (1, K, 1)
        expected_reward_0 = np.sum(
            expected_reward_all * self.lsb[s1][None, :, None], axis=1
        )
        self.expected_reward_0[s1] = expected_reward_0

        # (3) Arm selection.
        arm, _ = SoftMaxChoice(expected_reward_0, beta=p.beta)
        self.arm[s1] = arm

        # (4) Slice by chosen arm. Cue representation is shared by states.
        reward_arm = reward[:, arm]                           # (REWARDS,)
        expected_reward_arm = expected_reward_all[:, :, arm]  # (REWARDS, K)
        c_vec_arm = c_vec[:, :, arm]                          # (REWARDS, CUES)
        self.reward[s1] = reward_arm
        self.expected_reward[s1] = expected_reward_arm.copy()

        # (5) Prior variance (before this trial's update).
        sigmasq0 = self.sigmasq[s]

        # (6) Belief update, state creation, associative update.
        ll = self._log_likelihood(s1, reward_arm, expected_reward_arm, sigmasq0)
        self.lsb[s1] = self._update_beliefs(s1, ll)
        self._maybe_add_state(s1, c_vec_arm, reward_arm, ll)
        self._update_v_b(s1, c_vec_arm, reward_arm, sigmasq0)

        self.step = s1
        return self


    def reset(self) -> "LatentState":
        """Reset all histories to the initial state.

        Returns:
            ``self``.
        """
        self._initialize_history()
        self.step = 0
        return self


    def save_state(self) -> dict:
        """Snapshot the current state for ``Parameters.initialV``.

        Returns:
            Dict with copies of ``V``, ``B``, ``lsb``, derived ``Lmax``, ``q``,
            and ``sigmasq`` at ``self.step``.
        """
        s = self.step
        return {
            "V": self.V[s].copy(),
            "B": self.B[s].copy(),
            "lsb": self.lsb[s].copy(),
            "Lmax": int(self.lsb[s].size),
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
        lsb = self.lsb[s]
        n_states = lsb.size

        # Temporal decay toward uniform (per elapsed time between trials).
        if p.t != 1 and p.time is not None:
            elapsed = p.time[p.t - 1] - p.time[p.t - 2]
            prob_stay = (1 - p.gamma) ** (elapsed - 1)
            lsb[:] = lsb * prob_stay + (1 - prob_stay) / n_states

        # Explicit context reset: uniform over currently active states.
        if p.context is not None and p.t in p.context:
            lsb[:] = 1.0
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
            s: History slot whose active-state count is used for validation.
            reward: Realized reward, shape ``(REWARDS,)``.
            expected_reward: Per-state predictions for the chosen arm, shape
                ``(REWARDS, K)``.
            sigmasq: Variance used to scale squared error.

        Returns:
            Log-likelihood vector of shape ``(K,)`` for currently active states.
        """
        n_states = self.lsb[s].size
        if expected_reward.shape != (self.par.REWARDS, n_states):
            raise ValueError(
                "expected_reward must have shape "
                f"({self.par.REWARDS}, {n_states}); got {expected_reward.shape}."
            )
        return np.sum(
            -(reward[:, None] - expected_reward) ** 2 / (2 * sigmasq),
            axis=0,
        )


    def _update_beliefs(self, s: int, ll: np.ndarray) -> np.ndarray:
        """Prior times likelihood, renormalized over active states.

        Args:
            s: History slot whose prior beliefs are read.
            ll: Log-likelihood vector from :meth:`_log_likelihood`.

        Returns:
            New belief vector with the same length as the current state support.
        """
        p = self.par
        prior_beliefs = self.lsb[s]
        n_states = prior_beliefs.size

        prior = (
            (1 - p.gamma) * prior_beliefs + p.gamma / n_states
        )
        scaled_ll = np.exp(ll - np.max(ll))
        out = prior * scaled_ll
        return out / np.sum(out)


    def _maybe_add_state(
        self,
        s: int,
        c_vec: np.ndarray,
        reward: np.ndarray,
        ll: np.ndarray,
    ) -> None:
        """Test for a change point and append one new latent state if needed.

        The candidate state's log-likelihood is neutral (zero), as in the
        original implementation's inactive candidate block. When a change
        point is detected, one new belief, associative-strength slice, and
        covariance slice are appended.

        Args:
            s: History slot being updated in place.
            c_vec: Shared cues for the chosen arm, shape ``(REWARDS, CUES)``.
            reward: Realized reward, shape ``(REWARDS,)``.
            ll: Log-likelihood vector over the currently active states.
        """
        p = self.par
        beliefs = self.lsb[s]
        n_states = beliefs.size

        # Include the neutral candidate state's likelihood (log-likelihood 0)
        # when obtaining the common scale, matching the previous comparison.
        candidate_ll = 0.0
        mx = np.max(np.exp(np.append(ll, candidate_ll)))

        # Scaled likelihood if we keep the current active set.
        stay_prior = (1 - p.gamma) * beliefs + p.gamma / n_states
        stay = stay_prior * np.exp(ll - mx)
        l0 = np.sum(stay)

        # Scaled likelihood if we spawn one new state.
        l1 = np.exp(candidate_ll - mx)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log(l1) - np.log(l0)
        if np.isnan(log_ratio):
            log_ratio = 0.0

        new_q = max(self.q[s] + log_ratio - p.delta, 0.0)
        if new_q < p.eta:
            self.q[s] = new_q
            return

        # --- spawn one state ---
        n_new_states = n_states + 1
        self.q[s] = 0.0

        # Update beliefs over the old states plus the newly created state.
        new_state_prior = np.concatenate(
            (
                (1 - p.gamma) * beliefs + p.gamma / n_new_states,
                np.array([p.gamma / n_new_states]),
            )
        )
        all_ll = np.append(ll, candidate_ll)
        updated_beliefs = new_state_prior * np.exp(all_ll)
        self.lsb[s] = updated_beliefs / np.sum(updated_beliefs)

        # Initialize associative strengths for the new state's shared cues.
        Vnew = c_vec * reward[:, None] / np.sum(
            c_vec ** 2, axis=1, keepdims=True
        )
        self.V[s] = np.concatenate((self.V[s], Vnew[:, :, None]), axis=-1)

        # Initialize the new state's covariance to the identity per reward dim.
        Bnew = np.zeros(
            (p.CUES, p.CUES, p.REWARDS, 1), dtype=np.float64
        )
        for d in range(p.REWARDS):
            Bnew[:, :, d, 0] = np.eye(p.CUES)
        self.B[s] = np.concatenate((self.B[s], Bnew), axis=-1)

    def _update_v_b(
        self,
        s: int,
        c_vec: np.ndarray,
        reward: np.ndarray,
        sigmasq0: float,
    ) -> None:
        """Rescorla-Wagner update of ``V`` and ``B``, plus variance tracking.

        Args:
            s: History slot being updated in place.
            c_vec: Shared cues for the chosen arm, shape ``(REWARDS, CUES)``.
            reward: Realized reward, shape ``(REWARDS,)``.
            sigmasq0: Prior variance (before this trial's update); used both
                inside and at the end of the exp. moving avg. step.
        """
        p = self.par
        V = self.V[s]
        B = self.B[s]
        lsb = self.lsb[s]
        n_states = lsb.size
        sigmasq = float(self.sigmasq[s])

        # Rumination iterations (fewer on the last trial).
        if p.t == p.ntrials or p.time is None:
            nIter = 1
        else:
            nIter = min(p.chi, round(p.time[p.t] - p.time[p.t - 1]))

        expected_reward = np.sum(V * c_vec[:, :, None], axis=1)

        for d in range(p.REWARDS):
            for state in range(n_states):
                # Exponentially weighted covariance.
                c_d = c_vec[d, :]
                c_outer = np.outer(c_d, c_d)
                B[:, :, d, state] = B[:, :, d, state] + p.alpha2 * (
                    lsb[state] * c_outer - B[:, :, d, state]
                )

                # Associability: alpha0 * belief * B^{-1} c.
                tmp = p.alpha0 * lsb[state] * np.linalg.solve(
                    B[:, :, d, state], c_d
                )

                # Rescorla-Wagner update.
                TD = reward[d] - expected_reward[d, state]
                V[d, :, state] = V[d, :, state] + tmp * TD
                sigmasq = sigmasq + p.alpha1 * lsb[state] * TD ** 2

                # Rumination.
                for _ in range(nIter - 1):
                    expected_reward = np.sum(V * c_vec[:, :, None], axis=1)
                    TD = reward[d] - expected_reward[d, state]
                    V[d, :, state] = V[d, :, state] + tmp * TD

        # Finish variance exp. moving avg. step.
        sigmasq = sigmasq - p.alpha1 * sigmasq0
        self.sigmasq[s] = sigmasq


    def __repr__(self) -> str:
        p = self.par
        return (
            f"LatentState(REWARDS={p.REWARDS}, CUES={p.CUES}, ARMS={p.ARMS}, "
            f"n_states={self.lsb[self.step].size}, "
            f"step={self.step}/{len(self.lsb) - 1})"
        )
