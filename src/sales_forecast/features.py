"""Supervised feature matrix for global LightGBM forecasting.

All lag and rolling features at date ``t`` use only values strictly before
``t``, so the matrix has no target leakage. Rows with incomplete history
(before the longest lag) are dropped when ``dropna=True``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .data import PRICE_COLUMN

DEFAULT_LAGS: tuple[int, ...] = (1, 2, 3, 7, 14, 28)
DEFAULT_WINDOWS: tuple[int, ...] = (7, 28)
ITEM_COLUMN = "item_id"
TARGET_COLUMN = "y"


def max_history(lags: Sequence[int], windows: Sequence[int]) -> int:
    """Minimum history length needed before the first valid training row."""
    values = list(lags) + list(windows)
    if not values:
        raise ValueError("lags and windows must not both be empty")
    return int(max(values))


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic calendar features aligned to ``index``."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    dow = index.dayofweek.astype(float)
    month = index.month.astype(float)
    week = index.isocalendar().week.astype(float).to_numpy()
    dayofyear = index.dayofyear.astype(float)
    return pd.DataFrame(
        {
            "dow": dow,
            "month": month,
            "weekofyear": week,
            "sin_doy": np.sin(2 * np.pi * dayofyear / 365.25),
            "cos_doy": np.cos(2 * np.pi * dayofyear / 365.25),
        },
        index=index,
    )


def lag_features(series: pd.Series, lags: Sequence[int]) -> pd.DataFrame:
    """Lagged values of ``series``. Column ``lag_k`` is ``series.shift(k)``."""
    return pd.DataFrame(
        {f"lag_{lag}": series.shift(lag) for lag in lags}, index=series.index
    )


def rolling_features(series: pd.Series, windows: Sequence[int]) -> pd.DataFrame:
    """Past-only rolling mean and std.

    Window statistics at ``t`` are computed on ``series[t-window:t]``
    (i.e. ``shift(1).rolling(window)``), never including the current target.
    """
    shifted = series.shift(1)
    frames = {}
    for window in windows:
        rolled = shifted.rolling(window, min_periods=window)
        frames[f"roll_mean_{window}"] = rolled.mean()
        frames[f"roll_std_{window}"] = rolled.std(ddof=0)
    return pd.DataFrame(frames, index=series.index)


def build_item_features(
    sales: pd.Series,
    exog: pd.DataFrame,
    price: pd.Series,
    lags: Sequence[int] = DEFAULT_LAGS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Feature matrix for one item, including the target column ``y``.

    ``exog`` must share the index of ``sales``. NaNs appear in the first
    ``max(lags, windows)`` rows and must be dropped before training.
    """
    if not sales.index.equals(exog.index):
        raise ValueError("exog index must match sales index")
    if not sales.index.equals(price.index):
        raise ValueError("price index must match sales index")
    parts = [
        sales.rename(TARGET_COLUMN).to_frame(),
        lag_features(sales, lags),
        rolling_features(sales, windows),
        calendar_features(sales.index),
        exog.astype(float),
        price.rename(PRICE_COLUMN).astype(float).to_frame(),
    ]
    return pd.concat(parts, axis=1)


def build_panel(
    sales: pd.DataFrame,
    exog: pd.DataFrame,
    prices: pd.DataFrame,
    lags: Sequence[int] = DEFAULT_LAGS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    dropna: bool = True,
    first_sale: dict[str, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Long panel of (date, item) rows ready for LightGBM.

    Parameters
    ----------
    first_sale:
        Optional mapping ``item -> first non-zero sale date``. Rows before
        that date are dropped (assortment introduction, not demand).
    """
    if not sales.columns.equals(prices.columns):
        raise ValueError("sales and prices must have the same columns")
    if not sales.index.equals(exog.index) or not sales.index.equals(prices.index):
        raise ValueError("sales, exog and prices must share the same index")

    frames = []
    for item in sales.columns:
        frame = build_item_features(
            sales[item], exog, prices[item], lags=lags, windows=windows
        )
        frame[ITEM_COLUMN] = item
        if first_sale is not None and item in first_sale and pd.notna(first_sale[item]):
            frame = frame.loc[frame.index >= first_sale[item]]
        frames.append(frame)
    panel = pd.concat(frames, axis=0)
    panel.index.name = "date"
    panel = panel.reset_index()
    if dropna:
        panel = panel.dropna()
    return panel.reset_index(drop=True)


def feature_columns(
    lags: Sequence[int] = DEFAULT_LAGS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    exog_columns: Sequence[str] = (),
) -> list[str]:
    """Ordered list of model feature names (excluding the target)."""
    columns = [f"lag_{lag}" for lag in lags]
    for window in windows:
        columns.extend([f"roll_mean_{window}", f"roll_std_{window}"])
    columns.extend(["dow", "month", "weekofyear", "sin_doy", "cos_doy"])
    columns.extend(list(exog_columns))
    columns.append(PRICE_COLUMN)
    columns.append(ITEM_COLUMN)
    return columns


def row_features_from_history(
    history: pd.Series,
    date: pd.Timestamp,
    exog_row: pd.Series,
    price: float,
    item: str,
    lags: Sequence[int] = DEFAULT_LAGS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> dict[str, float | str]:
    """One inference feature row for ``item`` at ``date`` from past ``history``.

    ``history`` must contain at least ``max_history(lags, windows)`` values
    ending the day before ``date``.
    """
    needed = max_history(lags, windows)
    if len(history) < needed:
        raise ValueError(f"need at least {needed} history points, got {len(history)}")
    values = history.astype(float).to_numpy()
    features: dict[str, float | str] = {}
    for lag in lags:
        features[f"lag_{lag}"] = float(values[-lag])
    for window in windows:
        window_vals = values[-window:]
        features[f"roll_mean_{window}"] = float(window_vals.mean())
        features[f"roll_std_{window}"] = float(window_vals.std(ddof=0))
    cal = calendar_features(pd.DatetimeIndex([date])).iloc[0]
    features.update({key: float(cal[key]) for key in cal.index})
    features.update({key: float(exog_row[key]) for key in exog_row.index})
    features[PRICE_COLUMN] = float(price)
    features[ITEM_COLUMN] = item
    return features
