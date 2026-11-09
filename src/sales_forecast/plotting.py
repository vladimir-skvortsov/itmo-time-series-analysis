"""Plot helpers used by :class:`sales_forecast.forecaster.SalesForecaster`."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_series_grid(
    sales: pd.DataFrame, n_cols: int = 3, title: str | None = None
) -> Figure:
    """One small line chart per column of ``sales``."""
    n_items = sales.shape[1]
    n_rows = math.ceil(n_items / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 2.6 * n_rows), sharex=True
    )
    for ax, item in zip(axes.flat, sales.columns):
        ax.plot(sales.index, sales[item], linewidth=0.7)
        ax.set_title(item, fontsize=10)
    for ax in axes.flat[n_items:]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_forecast(
    history: pd.Series,
    actual: pd.Series,
    forecasts: dict[str, pd.Series],
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """History + actual test values + one line per model forecast."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    ax.plot(history.index, history, color="black", linewidth=1, label="history")
    ax.plot(actual.index, actual, color="black", linewidth=1, linestyle="--",
            label="actual")
    for name, forecast in forecasts.items():
        ax.plot(forecast.index, forecast, linewidth=1.3, label=name)
    ax.axvline(actual.index[0], color="grey", linestyle=":", linewidth=1)
    ax.set_title(title or "")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    return ax


def plot_metric_bars(
    table: pd.DataFrame, metric: str, ax: Axes | None = None
) -> Axes:
    """Grouped bars: x = horizon, hue = model, y = metric.

    ``table`` is a long frame with columns ``model``, ``horizon``, ``metric``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=table, x="horizon", y=metric, hue="model", ax=ax)
    ax.set_title(metric.upper())
    ax.set_xlabel("")
    ax.legend(fontsize=8)
    return ax


def plot_metric_heatmap(
    table: pd.DataFrame,
    metric: str,
    ax: Axes | None = None,
    vmax_quantile: float = 0.9,
) -> Axes:
    """Heatmap item x model for one metric and one horizon.

    ``table`` is a long frame with columns ``model``, ``item``, ``metric``.
    The colour scale is capped at the ``vmax_quantile`` quantile of the
    values so that a few items with extreme errors (e.g. stock-outs) do not
    flatten the colours of all other cells; annotations show exact values.
    """
    pivot = table.pivot(index="item", columns="model", values=metric)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.4 * len(pivot) + 1))
    vmax = np.nanquantile(pivot.to_numpy(dtype=float), vmax_quantile)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax,
                cbar=False, vmin=0, vmax=vmax)
    ax.set_title(metric.upper())
    ax.set_xlabel("")
    ax.set_ylabel("")
    return ax
