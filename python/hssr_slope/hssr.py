"""HSSR-SLOPE: Harmonics-enhanced Singular Spectrum Reconstruction of the PPG
signal followed by SLOPE beat detection."""

import warnings
from dataclasses import dataclass

import numpy as np

from . import _dsp
from .slope import slope_beat_detection
from .ssa import ssa


@dataclass
class HSSRResult:
    """Output of :func:`hssr_slope`.

    Attributes
    ----------
    peaks : array of int
        Sample indices of the detected beats in the PPG signal.
    ibi : array of shape (n_beats - 1, 2)
        Inter-beat intervals: time of each beat (s) and interval (ms).
    ibi_outliers : array of bool
        Intervals flagged by the outlier test (all False if disabled).
    fpeak : float
        Characteristic frequency of the PPG signal (Hz).
    ppg_reconstructed : array of shape (n,)
        PPG waveform reconstructed from the first and second harmonic.
    fallback : bool
        True if beat detection on the reconstructed signal failed and the
        beats were detected on the pre-processed PPG signal instead.
    """

    peaks: np.ndarray
    ibi: np.ndarray
    ibi_outliers: np.ndarray
    fpeak: float
    ppg_reconstructed: np.ndarray
    fallback: bool


def hssr_slope(ppg, acc, fs_ppg, fs_acc, *, fpeak=None,
               reconstruction_order=None, harmonics_tolerance=0.15,
               correlation_tolerance=0.025, ppg_band=(0.67, 8.0),
               second_harmonic_min_db=-20.0, ibi_outlier_test="grubbs",
               visual_inspection=True):
    """Reconstruct a wrist PPG signal and detect its beats.

    Parameters
    ----------
    ppg : array of shape (n,)
        Raw PPG signal.
    acc : array of shape (m, 3)
        Three-axis acceleration recorded together with the PPG signal. It is
        used as a surrogate of the motion noise.
    fs_ppg, fs_acc : float
        Sampling frequencies (Hz) of the two signals.
    fpeak : float, optional
        Characteristic frequency of the PPG signal (Hz). If not given, it is
        estimated from the Wiener-filtered spectrum.
    reconstruction_order : int, optional
        Number of SSA components (RO). Default: twice the PPG sampling
        frequency.
    harmonics_tolerance : float
        Half-width (Hz) of the first and second harmonic bands (HT).
    correlation_tolerance : float
        A component is discarded when its correlation with the noise surrogate
        exceeds its correlation with the PPG signal by at least this value (CT).
    ppg_band : (float, float)
        Pass band (Hz) of the pre-processing filter.
    second_harmonic_min_db : float
        The second harmonic is added to the reconstruction only if its power
        is not lower than that of the first harmonic by more than this value.
    ibi_outlier_test : {"grubbs", None}
        Test used to flag anomalous inter-beat intervals.
    visual_inspection : bool
        If True and `fpeak` is not given, the estimated spectra are shown and
        the user can confirm the estimated frequency (space bar) or pick
        another one (left click).

    Returns
    -------
    HSSRResult
    """
    ppg = np.asarray(ppg, dtype=float).ravel()
    acc = np.asarray(acc, dtype=float)
    if ppg.size == 0:
        raise ValueError("ppg is empty.")
    if acc.ndim != 2 or 3 not in acc.shape:
        raise ValueError("acc must have shape (m, 3) or (3, m).")
    if acc.shape[1] != 3:
        acc = acc.T

    n = len(ppg)
    order = int(reconstruction_order or 2 * fs_ppg)

    # --- PPG pre-processing: z-score, RMS envelope normalisation, band-pass
    ppg = _dsp.zscore(ppg)
    env_up, env_low = _dsp.rms_envelope(ppg, int(2 * fs_ppg))
    ppg_norm = (ppg - env_low) / (env_up - env_low)
    ppg_norm = _dsp.bandpass(ppg_norm, ppg_band, fs_ppg)

    # --- Noise surrogate: z-scored magnitude of the acceleration
    if fs_acc != fs_ppg:
        acc = _dsp.resample_to(acc, fs_acc, fs_ppg)
    noise = _dsp.zscore(np.sqrt((acc ** 2).sum(axis=1)))
    if len(noise) < n:
        noise = np.concatenate((np.zeros(n - len(noise)), noise))
    noise = noise[:n]

    # --- Characteristic frequency from the Wiener-filtered spectrum
    if fpeak is None:
        f, psd_ppg = _dsp.average_spectrum(ppg_norm, fs_ppg)
        _, psd_noise = _dsp.average_spectrum(noise, fs_ppg)
        p, q = _dsp.rescale01(psd_ppg), _dsp.rescale01(psd_noise)
        with np.errstate(invalid="ignore", divide="ignore"):
            wiener = np.where(p + q > 0, p / (p + q), 0.0)
        psd_clean = psd_ppg * wiener
        fpeak = float(f[np.argmax(psd_clean)])
        if visual_inspection:
            fpeak = _inspect_fpeak(f, psd_ppg, psd_noise, psd_clean, fpeak)

    ht = harmonics_tolerance
    band1 = (fpeak - ht, fpeak + ht)
    band2 = (2 * fpeak - ht, 2 * fpeak + ht)

    # --- SSA decomposition and characteristic frequency of each component
    rc = ssa(ppg_norm, order)
    freqs = np.arange(0.0, fs_ppg / 2 + 0.005, 0.01)
    f_rc = np.array([freqs[np.argmax(_dsp.burg_psd(rc[:, k], 4, freqs, fs_ppg))]
                     for k in range(order)])

    # --- Component selection: noise correlation, then harmonic bands
    corr_ppg = np.array([np.corrcoef(ppg, rc[:, k])[0, 1] for k in range(order)])
    corr_noise = np.array([np.corrcoef(noise, rc[:, k])[0, 1] for k in range(order)])
    clean = (corr_noise - corr_ppg) < correlation_tolerance

    in1 = clean & (f_rc >= band1[0]) & (f_rc <= band1[1])
    in2 = clean & (f_rc >= band2[0]) & (f_rc <= band2[1])
    harm1 = rc[:, in1].sum(axis=1)
    harm2 = rc[:, in2].sum(axis=1)

    delta_db = _dsp.welch_power_db(harm2, fs_ppg) - _dsp.welch_power_db(harm1, fs_ppg)
    ppg_rec = harm1 + harm2 if delta_db >= second_harmonic_min_db else harm1

    # --- Beat detection
    fallback = False
    try:
        peaks = slope_beat_detection(ppg_rec, fs_ppg)
        if len(peaks) < 3:
            raise ValueError("too few beats")
    except (ValueError, IndexError):
        warnings.warn("Beat detection failed on the reconstructed signal; "
                      "the pre-processed PPG signal was used instead.")
        peaks = slope_beat_detection(ppg_norm, fs_ppg)
        fallback = True

    t_beats = (peaks + 1) / fs_ppg
    ibi = np.column_stack((t_beats[1:], np.diff(t_beats) * 1000.0))
    if ibi_outlier_test == "grubbs":
        outliers = _dsp.grubbs_outliers(ibi[:, 1])
    elif ibi_outlier_test is None:
        outliers = np.zeros(len(ibi), dtype=bool)
    else:
        raise ValueError(f"Unknown outlier test: {ibi_outlier_test!r}")

    return HSSRResult(peaks, ibi, outliers, fpeak, ppg_rec, fallback)


def _inspect_fpeak(f, psd_ppg, psd_noise, psd_clean, fpeak):
    """Show the spectra and let the user confirm or change the estimate."""
    import matplotlib
    import matplotlib.pyplot as plt

    if matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "ps", "cairo"):
        warnings.warn("Non-interactive matplotlib backend: visual inspection "
                      "skipped, automatic estimate kept.")
        return fpeak

    fig, ax = plt.subplots()
    ax.plot(f, psd_ppg, label="PSD PPG raw")
    ax.plot(f, psd_noise, label="PSD noise")
    ax.plot(f, psd_clean, label="PSD PPG clean")
    ax.axvline(fpeak, color="k", label="fpeak")
    ax.set_xlim(0, 3)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_title("Confirm (space bar) or change (left click) the PPG fpeak")
    ax.grid(True)
    ax.legend()

    choice = {"f": fpeak}

    def on_click(event):
        if event.inaxes is ax and event.button == 1:
            choice["f"] = float(event.xdata)
            fig.canvas.stop_event_loop()

    def on_key(event):
        if event.key == " ":
            fig.canvas.stop_event_loop()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", lambda e: fig.canvas.stop_event_loop())
    plt.show(block=False)
    fig.canvas.start_event_loop(timeout=-1)
    plt.close(fig)
    return choice["f"]
