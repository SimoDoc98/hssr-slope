"""Multi-response optimisation of the HSSR-SLOPE parameters.

A second-order regression model is fitted to each response variable measured
on a full-factorial grid of parameter values. The models are evaluated on a
denser grid, each predicted response is turned into a desirability between 0
and 1, and the weighted geometric mean of the desirabilities (Global
Desirability) is maximised (Derringer and Suich, 1980).
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class OptimizationResult:
    """Output of :func:`optimize_parameters`.

    Attributes
    ----------
    best : array of shape (n_factors,)
        Parameter values with the highest Global Desirability.
    best_responses : array of shape (n_responses,)
        Responses predicted at `best`, on the scale of the measurements.
    global_desirability : array of shape (n_points,)
        Global Desirability on the dense grid.
    grid : array of shape (n_points, n_factors)
        Dense grid of parameter values.
    predicted : array of shape (n_points, n_responses)
        Predicted responses on the dense grid, on the scale of the measurements.
    models : list
        Fitted regression model of each response (statsmodels results).
    families : list of str
        Distribution selected for each response.
    """

    best: np.ndarray
    best_responses: np.ndarray
    global_desirability: np.ndarray
    grid: np.ndarray
    predicted: np.ndarray
    models: list
    families: list


def _quadratic_design(x):
    """Intercept, linear, interaction and squared terms of the factors."""
    n, p = x.shape
    cols = [np.ones(n)] + [x[:, i] for i in range(p)]
    cols += [x[:, i] * x[:, j] for i, j in combinations(range(p), 2)]
    cols += [x[:, i] ** 2 for i in range(p)]
    return np.column_stack(cols)


def _fit_response(design, y, family):
    import statsmodels.api as sm

    candidates = {
        "normal": lambda: sm.GLM(y, design, family=sm.families.Gaussian()),
        "gamma": lambda: sm.GLM(
            y, design, family=sm.families.Gamma(sm.families.links.Log())),
    }
    names = list(candidates) if family == "auto" else [family]
    fits = [candidates[name]().fit(use_t=True) for name in names]
    if len(fits) == 1:
        return fits[0], names[0]
    # Same selection criterion as the MATLAB implementation
    k = design.shape[1]
    score = [2 * k - 2 * m.llf for m in fits]  # Akaike information criterion
    best = int(np.argmin(score))
    return fits[best], names[best]


def optimize_parameters(x, y, directions, importance, dense_grid, *,
                        alpha=0.05, family="auto"):
    """Find the parameter values that maximise the Global Desirability.

    Parameters
    ----------
    x : array of shape (n_runs, n_factors)
        Parameter values of each run of the full-factorial design.
    y : array of shape (n_runs, n_responses)
        Response variables measured in each run.
    directions : sequence of {"max", "min"}
        Whether each response should be maximised or minimised.
    importance : sequence of float
        Importance of each response in the Global Desirability.
    dense_grid : sequence of arrays
        Values of each factor on which the fitted models are evaluated.
    alpha : float
        Significance level; regression coefficients with a larger p-value are
        set to zero before prediction.
    family : {"auto", "normal", "gamma"}
        Distribution of the regression model. "normal" uses an identity link,
        "gamma" a log link; "auto" fits both and keeps one for each response.

    Returns
    -------
    OptimizationResult
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.asarray(importance, dtype=float)
    if y.shape[1] != len(directions) or y.shape[1] != len(r):
        raise ValueError("directions and importance need one entry per response.")

    design = _quadratic_design(x)
    mesh = np.meshgrid(*dense_grid, indexing="ij")
    grid = np.column_stack([m.ravel() for m in mesh])
    grid_design = _quadratic_design(grid)

    models, families = [], []
    predicted = np.empty((len(grid), y.shape[1]))
    for i in range(y.shape[1]):
        model, name = _fit_response(design, y[:, i], family)
        beta = np.where(model.pvalues < alpha, model.params, 0.0)
        eta = grid_design @ beta
        predicted[:, i] = np.exp(eta) if name == "gamma" else eta
        models.append(model)
        families.append(name)

    desirability = np.empty_like(predicted)
    for i, direction in enumerate(directions):
        low = predicted[:, i].min() - 0.01
        high = predicted[:, i].max() + 0.01
        if direction == "max":
            desirability[:, i] = (predicted[:, i] - low) / (high - low)
        elif direction == "min":
            desirability[:, i] = (high - predicted[:, i]) / (high - low)
        else:
            raise ValueError("directions must contain 'max' or 'min'.")

    gd = np.prod(desirability ** r, axis=1) ** (1.0 / r.sum())
    k = int(np.argmax(gd))
    best_responses = predicted[k].copy()
    return OptimizationResult(grid[k], best_responses, gd, grid, predicted,
                              models, families)


def plot_desirability(result, x_factor, y_factor, fixed, labels=None, ax=None):
    """Contour plot of the Global Desirability over two factors.

    Parameters
    ----------
    result : OptimizationResult
    x_factor, y_factor : int
        Column indices of the factors on the horizontal and vertical axes.
    fixed : dict
        Value of every other factor, as {column index: value}. The value must
        belong to the dense grid.
    labels : sequence of str, optional
        Factor names, used for the axis labels.
    ax : matplotlib axes, optional
    """
    import matplotlib.pyplot as plt

    keep = np.ones(len(result.grid), dtype=bool)
    for col, value in fixed.items():
        keep &= np.isclose(result.grid[:, col], value)
    xs = np.unique(result.grid[keep, x_factor])
    ys = np.unique(result.grid[keep, y_factor])
    z = np.full((len(ys), len(xs)), np.nan)
    ix = np.searchsorted(xs, result.grid[keep, x_factor])
    iy = np.searchsorted(ys, result.grid[keep, y_factor])
    z[iy, ix] = result.global_desirability[keep]

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    filled = ax.contourf(xs, ys, z, levels=np.arange(0, 1.001, 0.01))
    lines = ax.contour(xs, ys, z, levels=np.arange(0, 1.001, 0.1),
                       colors="k", linewidths=0.6)
    ax.clabel(lines, fmt="%0.1f")
    if labels is not None:
        ax.set_xlabel(labels[x_factor])
        ax.set_ylabel(labels[y_factor])
    ax.figure.colorbar(filled, ax=ax, label="Global Desirability")
    return ax
