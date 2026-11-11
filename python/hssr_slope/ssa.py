"""Singular Spectrum Analysis (Vautard and Ghil, 1989)."""

import numpy as np


def ssa(x, order):
    """Decompose x into `order` reconstructed components.

    Parameters
    ----------
    x : array of shape (n,)
    order : int
        Embedding dimension (maximum lag), called Reconstruction Order in the
        paper.

    Returns
    -------
    rc : array of shape (n, order)
        Reconstructed components, sorted by decreasing eigenvalue. Their sum
        gives back the standardised input signal.
    """
    x = np.asarray(x, dtype=float)
    n, m = len(x), int(order)
    x = (x - x.mean()) / x.std()

    # Toeplitz lag-covariance matrix from the unbiased autocorrelation
    acov = np.array([np.dot(x[: n - k], x[k:]) / (n - k) for k in range(m)])
    lags = np.abs(np.subtract.outer(np.arange(m), np.arange(m)))
    c = acov[lags]

    eigval, rho = np.linalg.eigh(c)
    rho = rho[:, np.argsort(eigval)[::-1]]

    # Principal components: projection of the lagged copies on the eigenvectors
    y = np.lib.stride_tricks.sliding_window_view(x, m)
    pc = y @ rho

    # Reconstructed components by diagonal averaging
    counts = np.convolve(np.ones(n - m + 1), np.ones(m))
    rc = np.empty((n, m))
    for k in range(m):
        rc[:, k] = np.convolve(pc[:, k], rho[:, k]) / counts
    return rc
