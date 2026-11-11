"""SLOPE beat detector: adaptive threshold with a linear decay after each peak."""

import numpy as np
from scipy import signal


def slope_beat_detection(ppg, fs, refractory=0.4, sr=0.6, alpha=0.8):
    """Detect the systolic peaks of a PPG signal.

    The threshold follows the signal while it rises. Once a local maximum is
    found, the threshold decays linearly with an adaptive slope until it
    crosses the signal again (not earlier than the refractory period).

    Parameters
    ----------
    ppg : array of shape (n,)
    fs : float
        Sampling frequency in Hz.
    refractory : float
        Refractory period in seconds.
    sr, alpha : float
        Update rate and scaling of the slope coefficient.

    Returns
    -------
    peaks : array of int
        Sample indices of the detected peaks.
    """
    ppg = np.asarray(ppg, dtype=float)
    x = signal.detrend(ppg)
    n = len(x)

    slope = alpha * x.max()

    # Start from the first trough below zero
    troughs, _ = signal.find_peaks(-ppg, height=0, width=0.3)
    if len(troughs) == 0:
        raise ValueError("No starting trough found in the PPG signal.")

    peaks = []
    i = int(troughs[0])
    j = None

    while i < n - 1:
        # Follow the signal up to the next local maximum
        i2 = i + 1
        while i2 < n - 1:
            if x[i2] > x[i2 + 1]:
                j = i2
                break
            i2 += 1

        if j is None or (peaks and peaks[-1] == j):
            break
        peaks.append(j)

        slope = (1.0 - sr) * slope + sr * alpha * x[j]

        # Linear decay of the threshold until it meets the signal again
        i3 = j + 1
        th = x[j] - slope / fs
        while i3 <= n - 2:
            if th < x[i3] and (i3 - j) / fs >= refractory:
                i = i3 + 1
                break
            i3 += 1
            th -= slope / fs

    return np.asarray(peaks, dtype=int)
