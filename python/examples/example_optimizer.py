"""Optimise the HSSR-SLOPE parameters from a sensitivity analysis table.

The table lists, for each combination of Reconstruction Order (RO),
Correlation Tolerance (CT) and Harmonics Tolerance (HT), the response
variables measured against a reference ECG: quartiles of the beat detection
F1-score and limits of agreement (LOA) of five HRV features.

Usage:  python example_optimizer.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hssr_slope import optimize_parameters, plot_desirability

DATA = Path(__file__).resolve().parents[2] / "data" / "sensitivity_walking.csv"

table = pd.read_csv(DATA)
factors = ["RO", "CT", "HT"]
responses = [c for c in table.columns if c not in factors]

# F1-score quartiles are maximised, LOA widths are minimised
directions = ["max", "max", "max", "min", "min", "min", "min", "min"]

# Importance of each response. Change these values to give priority to the
# features that matter for the application, e.g. raise the last two (LOA of
# LF and HF) for a frequency-domain analysis.
importance = [1, 2, 1, 3, 3, 3, 3, 3]

dense_grid = [np.arange(32, 321, 4),
              np.round(np.arange(0, 0.0501, 0.005), 3),
              np.round(np.arange(0.10, 0.501, 0.01), 2)]

result = optimize_parameters(table[factors].to_numpy(),
                             table[responses].to_numpy(),
                             directions, importance, dense_grid)

print("Optimal parameters")
for name, value in zip(factors, result.best):
    print(f"  {name}: {value:g}")
print("Predicted responses")
for name, value in zip(responses, result.best_responses):
    print(f"  {name}: {value:.3f}")

plot_desirability(result, x_factor=0, y_factor=2, fixed={1: 0.025},
                  labels=["Reconstruction Order", "Correlation Tolerance",
                          "Harmonics Tolerance"])
plt.show()
