import numpy as np

def SoftMaxChoice(mu, par):
    """
    Assuming mu is (D, A) and we want a 1D choice.
    """
    tau = par['tau']

    # Normalize to 1D shape (A,). Handles scalars, (1,), (1, A), (D, 1).
    mu_1d = np.atleast_1d(mu).flatten()

    # Numerically stable softmax
    z = tau * mu_1d
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)

    # Categorical sample — same distribution as MATLAB's inverse-CDF sampling.
    arm = int(np.random.choice(len(p), p=p))

    return arm, p
