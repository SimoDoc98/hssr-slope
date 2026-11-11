import numpy as np
import pytest

from hssr_slope import hssr_slope, optimize_parameters, slope_beat_detection, ssa
from hssr_slope.synthetic import synthetic_recording


def test_ssa_components_sum_to_signal():
    rng = np.random.default_rng(1)
    x = np.sin(np.arange(500) / 7.0) + 0.1 * rng.standard_normal(500)
    rc = ssa(x, 20)
    assert rc.shape == (500, 20)
    assert np.allclose(rc.sum(axis=1), (x - x.mean()) / x.std())


def test_slope_finds_every_beat_of_a_clean_signal():
    fs = 64
    t = np.arange(60 * fs) / fs
    peaks = slope_beat_detection(np.cos(2 * np.pi * 1.2 * t), fs)
    assert abs(len(peaks) - 72) <= 2
    assert np.allclose(np.diff(peaks) / fs, 1 / 1.2, atol=2 / fs)


def test_hssr_slope_recovers_intervals_under_motion():
    ppg, acc, t_true = synthetic_recording(duration=120, seed=3)
    out = hssr_slope(ppg, acc, 64, 32, visual_inspection=False)
    ibi_true = np.diff(t_true) * 1000
    assert not out.fallback
    assert out.fpeak == pytest.approx(1.25, abs=0.15)
    assert abs(len(out.peaks) - len(t_true)) <= 3
    assert out.ibi[:, 1].mean() == pytest.approx(ibi_true.mean(), rel=0.01)


def test_optimizer_finds_the_maximum_of_a_known_surface():
    a, b = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 1, 6), indexing="ij")
    x = np.column_stack((a.ravel(), b.ravel()))
    rng = np.random.default_rng(0)
    y = 50 - (x[:, 0] - 6) ** 2 - 40 * (x[:, 1] - 0.4) ** 2
    y = (y + 0.05 * rng.standard_normal(len(y)))[:, None]
    res = optimize_parameters(x, y, ["max"], [1],
                              [np.linspace(0, 10, 101), np.linspace(0, 1, 101)],
                              family="normal")
    assert res.best[0] == pytest.approx(6, abs=0.3)
    assert res.best[1] == pytest.approx(0.4, abs=0.05)
