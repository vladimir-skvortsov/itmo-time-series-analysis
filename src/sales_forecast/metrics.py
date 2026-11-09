"""Forecast accuracy metrics suitable for intermittent count data.

MAPE is deliberately absent: it is undefined when the actual value is
zero, which happens on a large share of days for slow-moving items.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_NAMES = ("mae", "rmse", "wape", "mase")


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {pred.shape}")
    if true.size == 0:
        raise ValueError("empty input")
    if np.isnan(true).any() or np.isnan(pred).any():
        raise ValueError("inputs must not contain NaN")
    return true, pred


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    true, pred = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    true, pred = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: sum|e| / sum|y|.

    Returns NaN when the actuals sum to zero (relative error undefined).
    """
    true, pred = _as_arrays(y_true, y_pred)
    denominator = np.sum(np.abs(true))
    if denominator == 0:
        return float("nan")
    return float(np.sum(np.abs(true - pred)) / denominator)


def mase(y_true, y_pred, y_train, season_length: int = 1) -> float:
    """Mean absolute scaled error.

    The scale is the in-sample MAE of the seasonal naive forecast on
    ``y_train``. Returns NaN when the scale is zero.
    """
    true, pred = _as_arrays(y_true, y_pred)
    train = np.asarray(y_train, dtype=float)
    if season_length < 1:
        raise ValueError("season_length must be >= 1")
    if len(train) <= season_length:
        raise ValueError("y_train must be longer than season_length")
    scale = np.mean(np.abs(train[season_length:] - train[:-season_length]))
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(true - pred)) / scale)


def compute_metrics(
    y_true, y_pred, y_train, season_length: int = 1
) -> dict[str, float]:
    """All metrics as a ``{name: value}`` dictionary."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "mase": mase(y_true, y_pred, y_train, season_length),
    }


def aggregate_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    y_train: pd.DataFrame,
    season_length: int = 1,
) -> dict[str, float]:
    """Metrics over several items treated as one pooled sample.

    MAE/RMSE are averaged over all (day, item) cells; WAPE and MASE use
    pooled sums so that high-volume items weigh more -- which matches the
    business meaning of "total absolute error relative to total sales".
    """
    columns = list(y_true.columns)
    true = y_true[columns].to_numpy(dtype=float)
    pred = y_pred[columns].to_numpy(dtype=float)
    train = y_train[columns].to_numpy(dtype=float)
    abs_err = np.abs(true - pred)
    denominator = np.sum(np.abs(true))
    scale = np.mean(np.abs(train[season_length:] - train[:-season_length]))
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean((true - pred) ** 2))),
        "wape": float(abs_err.sum() / denominator) if denominator else np.nan,
        "mase": float(abs_err.mean() / scale) if scale else np.nan,
    }
