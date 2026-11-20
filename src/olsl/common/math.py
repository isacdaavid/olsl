"""Common math functions."""

import numpy as np


def SoftMaxChoice(mu: np.ndarray, beta: float) -> tuple[int, np.ndarray]:
    """Sample an action according to a softmax choice rule.

    Args:
        mu: Expected values. These are flattened to one value per action.
        beta: Softmax inverse temperature. Larger values make choices more
            deterministic; beta=0 gives a uniform distribution.

    Returns:
        The sampled action index and the action-probability vector.
    """
    # Normalize to 1D shape (A,). Handles scalars, (1,), (1, A), and (D, 1).
    mu_1d = np.atleast_1d(mu).flatten()

    # Numerically stable softmax.
    z = beta * mu_1d
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)

    # Categorical sample — inverse-CDF sampling.
    arm = int(np.random.choice(len(p), p=p))

    return arm, p
