"""Synthetic wrist recording used by the example and the tests."""

import numpy as np


def synthetic_recording(duration=120.0, fs_ppg=64, fs_acc=32, heart_rate=75.0,
                        motion_frequency=1.9, motion_amplitude=0.8, seed=0):
    """Simulate a PPG signal corrupted by periodic motion, with its acceleration.

    Returns
    -------
    ppg : array of shape (duration * fs_ppg,)
    acc : array of shape (duration * fs_acc, 3)
    t_beats : array
        Times (s) of the simulated heart beats.
    """
    rng = np.random.default_rng(seed)

    # Beat times: mean interval modulated at 0.1 Hz and 0.25 Hz, plus jitter
    t_beats = [0.0]
    while t_beats[-1] < duration + 2.0:
        t = t_beats[-1]
        ibi = 60.0 / heart_rate * (1.0 + 0.04 * np.sin(2 * np.pi * 0.1 * t)
                                   + 0.03 * np.sin(2 * np.pi * 0.25 * t))
        t_beats.append(t + ibi + 0.005 * rng.standard_normal())
    t_beats = np.array(t_beats)

    # Cardiac component: first and second harmonic of the beat phase
    t = np.arange(int(duration * fs_ppg)) / fs_ppg
    phase = 2 * np.pi * np.interp(t, t_beats, np.arange(len(t_beats)))
    cardiac = np.cos(phase) + 0.35 * np.cos(2 * phase - 0.8)

    # Motion: periodic arm swing with slowly varying amplitude
    def swing(time):
        amplitude = 1.0 + 0.3 * np.sin(2 * np.pi * 0.03 * time)
        return amplitude * np.sin(2 * np.pi * motion_frequency * time)

    ppg = (cardiac + motion_amplitude * swing(t)
           + 0.5 * np.sin(2 * np.pi * 0.05 * t)
           + 0.1 * rng.standard_normal(len(t)))

    ta = np.arange(int(duration * fs_acc)) / fs_acc
    acc = np.column_stack((0.6 * swing(ta), 0.3 * swing(ta + 0.05),
                           1.0 + 0.2 * swing(ta)))
    acc += 0.02 * rng.standard_normal(acc.shape)

    return ppg, acc, t_beats[t_beats <= duration]
