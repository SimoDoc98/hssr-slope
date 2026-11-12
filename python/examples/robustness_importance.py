"""Robustness of the selected HSSR-SLOPE parameters to the importance values.

For each condition of the sensitivity analysis, the Global Desirability (GD)
is recomputed for all the 3^8 = 6561 combinations of importance values
between 1 and 3 assigned to the eight response variables. The GD of the
selected parameters is compared with the highest GD among the tested
combinations of Reconstruction Order (RO) and Harmonics Tolerance (HT), with
Correlation Tolerance fixed at 0.025.

Usage:  python robustness_importance.py
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from hssr_slope import optimize_parameters

DATA = Path(__file__).resolve().parents[2] / "data"

FILES = {
    "Sensitivity_Analysis_PPGDalia.xlsx": ["Rest", "Cycling", "Walking", "Working"],
    "Sensitivity_Analysis_WESAD.xlsx": ["Baseline", "Meditation", "Amusement", "Stress"],
}

# Parameters selected in the paper (RO, HT)
SELECTED = {"Rest": (256, 0.25), "Cycling": (256, 0.15), "Walking": (288, 0.10),
            "Working": (320, 0.10), "Baseline": (224, 0.10), "Meditation": (192, 0.25),
            "Amusement": (256, 0.10), "Stress": (288, 0.25)}

DIRECTIONS = ["max", "max", "max", "min", "min", "min", "min", "min"]
IMPORTANCE = np.array([1, 2, 1, 3, 3, 3, 3, 3])
GRID = [np.arange(32, 321, 4),
        np.round(np.arange(0, 0.0501, 0.005), 3),
        np.round(np.arange(0.10, 0.501, 0.01), 2)]

FIXED_CT = 0.025
TESTED_RO = np.arange(32, 321, 32)
TESTED_HT = np.round(np.arange(0.10, 0.501, 0.05), 2)

weights = np.array(list(product([1, 2, 3], repeat=8)))  # 6561 x 8

rows = []
for file, conditions in FILES.items():
    for condition in conditions:
        table = pd.read_excel(DATA / file, sheet_name=condition, header=None,
                              skiprows=4, nrows=270, usecols="B:L").to_numpy(float)
        res = optimize_parameters(table[:, :3], table[:, 3:], DIRECTIONS,
                                  IMPORTANCE, GRID)

        # Desirability of each response (independent of the importance values)
        p = res.predicted
        low, high = p.min(axis=0) - 0.01, p.max(axis=0) + 0.01
        d = np.where(np.array(DIRECTIONS) == "max",
                     (p - low) / (high - low), (high - p) / (high - low))

        g = res.grid
        tested = (np.isclose(g[:, 1], FIXED_CT)
                  & np.isin(g[:, 0], TESTED_RO)
                  & np.isclose(g[:, 2][:, None], TESTED_HT).any(axis=1))
        g, log_d = g[tested], np.log(d[tested])

        ro, ht = SELECTED[condition]
        i_sel = np.flatnonzero((g[:, 0] == ro) & np.isclose(g[:, 2], ht))[0]

        gd_paper = np.exp(log_d @ IMPORTANCE / IMPORTANCE.sum())
        i_best = int(np.argmax(gd_paper))

        gd_all = np.exp(log_d @ weights.T / weights.sum(axis=1))
        ratio = gd_all[i_sel] / gd_all.max(axis=0)

        rows.append({"Condition": condition,
                     "RO_selected": ro, "HT_selected": ht,
                     "RO_best": int(g[i_best, 0]), "HT_best": g[i_best, 2],
                     "GD_selected": gd_paper[i_sel], "GD_best": gd_paper[i_best],
                     "Within5pct": 100 * np.mean(ratio >= 0.95),
                     "MedianRatio": np.median(ratio)})

results = pd.DataFrame(rows)
pd.set_option("display.precision", 3)
print(results.to_string(index=False))
results.to_csv("robustness_importance.csv", index=False)
