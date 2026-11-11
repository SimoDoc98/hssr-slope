"""Signal processing helpers used by HSSR-SLOPE.

Several of these functions stand in for MATLAB built-ins used by the original
implementation (envelope, bandpass, pspectrum, pburg, isoutlier). They follow
the documented behaviour of those functions but are not bit-exact copies.
"""

from fractions import Fraction

import numpy as np
from scipy import signal, stats


def zscore(x):
    """Standardise with the sample standard deviation (as MATLAB zscore)."""
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std(ddof=1)


def moving_mean(x, k):
    """Centred moving average with shrinking windows at the edges (movmean)."""
    n = len(x)
    before = k // 2
    after = k - before - 1
    c = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(n)
    lo = np.clip(idx - before, 0, n)
    hi = np.clip(idx + after + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)


def rms_envelope(x, window):
    """Upper and lower RMS envelopes over a sliding window of `window` samples."""
    m = x.mean()
    rms = np.sqrt(moving_mean((x - m) ** 2, int(window)))
    return m + rms, m - rms


def bandpass(x, band, fs, steepness=0.85, attenuation=60.0):
    """Zero-phase band-pass filter.

    Mirrors MATLAB ``bandpass(x, band, fs, 'Steepness', s)``: the transition
    widths are (1 - s) * f_low below the pass band and (1 - s) * (fs/2 - f_high)
    above it. A linear-phase FIR filter is used when the signal is long enough,
    otherwise an elliptic IIR filter applied forwards and backwards.
    """
    nyq = fs / 2.0
    f_lo, f_hi = band
    w_lo = (1.0 - steepness) * f_lo
    w_hi = (1.0 - steepness) * (nyq - f_hi)
    numtaps, beta = signal.kaiserord(attenuation, min(w_lo, w_hi) / nyq)
    numtaps |= 1  # odd length, integer group delay

    if len(x) > 2 * numtaps:
        h = signal.firwin(numtaps, [f_lo - w_lo / 2, f_hi + w_lo / 2],
                          window=("kaiser", beta), pass_zero=False, fs=fs)
        return np.convolve(x, h, mode="same")

    sos = signal.iirdesign([f_lo, f_hi], [f_lo - w_lo, f_hi + w_hi],
                           gpass=0.1, gstop=attenuation, ftype="ellip",
                           output="sos", fs=fs)
    return signal.sosfiltfilt(sos, x)


def _default_segment_length(n):
    """Default spectrogram time resolution of MATLAB pspectrum, in samples."""
    for limit, divisor in ((64, 2), (256, 8), (2048, 8), (4096, 16),
                           (8192, 32), (16384, 64)):
        if n < limit:
            return int(np.ceil(n / divisor))
    return int(np.ceil(n / 128))


def average_spectrum(x, fs, leakage=0.75, overlap_percent=0.5):
    """Time-averaged spectrogram (Kaiser window), as mean(pspectrum(...), 2)."""
    nperseg = max(_default_segment_length(len(x)), 8)
    noverlap = int(np.floor(overlap_percent / 100.0 * nperseg))
    nfft = max(2048, int(2 ** np.ceil(np.log2(nperseg))))
    f, p = signal.welch(x, fs=fs, window=("kaiser", 40.0 * (1.0 - leakage)),
                        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                        detrend=False, scaling="spectrum")
    return f, p


def rescale01(x):
    return (x - x.min()) / (x.max() - x.min())


def burg_psd(x, order, freqs, fs):
    """Power spectral density from a Burg autoregressive model (pburg)."""
    x = np.asarray(x, dtype=float)
    ef = x[1:].copy()
    eb = x[:-1].copy()
    a = np.array([1.0])
    e = np.dot(x, x) / len(x)
    for _ in range(order):
        k = -2.0 * np.dot(eb, ef) / (np.dot(ef, ef) + np.dot(eb, eb))
        a = np.concatenate((a, [0.0])) + k * np.concatenate(([0.0], a[::-1]))
        ef, eb = ef[1:] + k * eb[1:], eb[:-1] + k * ef[:-1]
        e *= 1.0 - k * k
    w = 2.0 * np.pi * np.asarray(freqs) / fs
    h = np.exp(-1j * np.outer(w, np.arange(order + 1))) @ a
    return e / (fs * np.abs(h) ** 2)


def resample_to(x, fs_in, fs_out):
    """Polyphase resampling of each column of x from fs_in to fs_out."""
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    return signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=0)


def welch_power_db(x, fs):
    """Total power in dB, with the default settings of MATLAB pwelch."""
    if not np.any(x):
        return -np.inf
    nperseg = int(len(x) / 4.5)
    nfft = max(256, int(2 ** np.ceil(np.log2(nperseg))))
    _, p = signal.welch(x, fs=fs, window="hamming", nperseg=nperseg,
                        noverlap=nperseg // 2, nfft=nfft, detrend=False)
    return 20.0 * np.log10(p.sum())


def grubbs_outliers(x, alpha=0.05):
    """Iterative Grubbs test; returns a boolean mask of the outliers."""
    x = np.asarray(x, dtype=float)
    mask = np.zeros(len(x), dtype=bool)
    while True:
        keep = np.flatnonzero(~mask)
        n = len(keep)
        if n < 3:
            break
        v = x[keep]
        sd = v.std(ddof=1)
        if sd == 0:
            break
        dev = np.abs(v - v.mean())
        t = stats.t.ppf(1.0 - alpha / (2.0 * n), n - 2)
        g_crit = (n - 1) / np.sqrt(n) * np.sqrt(t * t / (n - 2 + t * t))
        worst = dev.argmax()
        if dev[worst] / sd <= g_crit:
            break
        mask[keep[worst]] = True
    return mask
