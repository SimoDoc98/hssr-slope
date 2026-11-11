"""Run HSSR-SLOPE on a simulated wrist recording.

Usage:  python example_hssr_slope.py
"""

import matplotlib.pyplot as plt
import numpy as np

from hssr_slope import hssr_slope
from hssr_slope.synthetic import synthetic_recording

FS_PPG = 64
FS_ACC = 32

ppg, acc, t_true = synthetic_recording(duration=120, fs_ppg=FS_PPG, fs_acc=FS_ACC)

# With visual_inspection=True a window shows the spectra of the PPG signal and
# of the noise surrogate: press the space bar to accept the estimated
# characteristic frequency, or click on the correct peak.
result = hssr_slope(ppg, acc, FS_PPG, FS_ACC, visual_inspection=True)

ibi_true = np.diff(t_true) * 1000
print(f"Characteristic frequency: {result.fpeak:.2f} Hz")
print(f"Detected beats: {len(result.peaks)} (simulated: {len(t_true)})")
print(f"Mean IBI: {result.ibi[:, 1].mean():.1f} ms (simulated: {ibi_true.mean():.1f} ms)")
print(f"SDNN:     {result.ibi[:, 1].std(ddof=1):.1f} ms (simulated: {ibi_true.std(ddof=1):.1f} ms)")

t = np.arange(len(ppg)) / FS_PPG
fig, axes = plt.subplots(3, 1, figsize=(10, 7))
axes[0].plot(t, ppg)
axes[0].set(title="Raw PPG", xlim=(20, 40))
axes[1].plot(t, result.ppg_reconstructed)
axes[1].plot(t[result.peaks], result.ppg_reconstructed[result.peaks], "o")
axes[1].set(title="Reconstructed PPG and detected beats", xlim=(20, 40))
axes[2].plot(t_true[1:], ibi_true, label="Simulated")
axes[2].plot(result.ibi[:, 0], result.ibi[:, 1], label="HSSR-SLOPE")
axes[2].set(title="Inter-beat intervals", xlabel="Time [s]", ylabel="IBI [ms]")
axes[2].legend()
fig.tight_layout()
plt.show()
