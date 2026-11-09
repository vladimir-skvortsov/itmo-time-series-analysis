"""Loading and preprocessing of the retail sales dataset.

The raw data consists of three tables:

* ``shop_sales.csv`` -- daily sales per (store, item) in long format;
* ``shop_sales_dates.csv`` -- calendar with holidays and cashback flags;
* ``shop_sales_prices.csv`` -- weekly prices per (store, item).

This module turns them into a :class:`StoreDataset`: a wide daily sales
matrix with a regular ``DatetimeIndex`` plus aligned exogenous features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SALES_FILE = "shop_sales.csv"
CALENDAR_FILE = "shop_sales_dates.csv"
PRICES_FILE = "shop_sales_prices.csv"

EVENT_TYPES = ("Cultural", "National", "Religious", "Sporting")
CLOSED_DAY_EVENT = "Christmas"
PRICE_COLUMN = "sell_price"


@dataclass(frozen=True)
class RawData:
    """Raw tables as read from disk."""

    sales: pd.DataFrame
    calendar: pd.DataFrame
    prices: pd.DataFrame


@dataclass(frozen=True)
class StoreDataset:
    """Preprocessed data of a single store.

    Attributes
    ----------
    store_id:
        Identifier of the store.
    sales:
        Daily sales, ``DatetimeIndex`` with ``freq='D'``, one column per
        item. No gaps, no NaN.
    exog:
        Calendar-based exogenous features shared by all items (holiday
        type indicators, closed-day flag, cashback flag). Same index as
        ``sales``.
    prices:
        Daily selling price per item (weekly price broadcast to days),
        same index and columns as ``sales``.
    """

    store_id: str
    sales: pd.DataFrame
    exog: pd.DataFrame
    prices: pd.DataFrame

    @property
    def items(self) -> list[str]:
        return list(self.sales.columns)

    def item_exog(self, item: str) -> pd.DataFrame:
        """Exogenous matrix for one item: shared calendar features + price."""
        price = self.prices[item].rename(PRICE_COLUMN)
        return pd.concat([self.exog, price], axis=1)


def load_raw(data_dir: str | Path) -> RawData:
    """Read the three CSV files from ``data_dir``."""
    data_dir = Path(data_dir)
    sales = pd.read_csv(data_dir / SALES_FILE)
    calendar = pd.read_csv(data_dir / CALENDAR_FILE, parse_dates=["date"])
    prices = pd.read_csv(data_dir / PRICES_FILE)
    _validate_raw(sales, calendar, prices)
    return RawData(sales=sales, calendar=calendar, prices=prices)


def _validate_raw(
    sales: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame
) -> None:
    required = {
        "sales": ({"item_id", "store_id", "date_id", "cnt"}, sales),
        "calendar": ({"date", "wm_yr_wk", "date_id"}, calendar),
        "prices": ({"store_id", "item_id", "wm_yr_wk", PRICE_COLUMN}, prices),
    }
    for name, (columns, frame) in required.items():
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
    if calendar["date_id"].duplicated().any():
        raise ValueError("calendar: duplicated date_id")
    if calendar["date"].duplicated().any():
        raise ValueError("calendar: duplicated date")


def build_store_dataset(raw: RawData, store_id: str) -> StoreDataset:
    """Build the preprocessed :class:`StoreDataset` for ``store_id``."""
    store_sales = raw.sales.loc[raw.sales["store_id"] == store_id]
    if store_sales.empty:
        known = sorted(raw.sales["store_id"].unique())
        raise ValueError(f"unknown store_id {store_id!r}; known: {known}")

    calendar = raw.calendar.sort_values("date").reset_index(drop=True)
    date_index = _full_date_index(calendar)
    date_of_id = calendar.set_index("date_id")["date"]

    sales = _sales_to_wide(store_sales, date_of_id, date_index)
    exog = _build_calendar_exog(calendar, store_id, date_index)
    store_prices = raw.prices.loc[raw.prices["store_id"] == store_id]
    prices = _prices_to_daily(store_prices, calendar, date_index, sales.columns)
    return StoreDataset(store_id=store_id, sales=sales, exog=exog, prices=prices)


def _full_date_index(calendar: pd.DataFrame) -> pd.DatetimeIndex:
    start, end = calendar["date"].min(), calendar["date"].max()
    index = pd.date_range(start, end, freq="D", name="date")
    if len(index) != len(calendar):
        raise ValueError("calendar has gaps: not every day between min and max")
    return index


def _sales_to_wide(
    store_sales: pd.DataFrame,
    date_of_id: pd.Series,
    date_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    if store_sales.duplicated(["item_id", "date_id"]).any():
        raise ValueError("sales: duplicated (item_id, date_id) pairs")
    unknown = set(store_sales["date_id"]) - set(date_of_id.index)
    if unknown:
        raise ValueError(f"sales: date_id values missing in calendar: {unknown}")
    frame = store_sales.assign(date=store_sales["date_id"].map(date_of_id))
    wide = frame.pivot(index="date", columns="item_id", values="cnt")
    wide = wide.reindex(date_index).fillna(0.0).astype(float)
    wide.columns.name = None
    wide = wide.sort_index(axis=1)
    wide.index.freq = "D"
    return wide


def _build_calendar_exog(
    calendar: pd.DataFrame, store_id: str, date_index: pd.DatetimeIndex
) -> pd.DataFrame:
    cal = calendar.set_index("date").reindex(date_index)
    type_columns = [c for c in ("event_type_1", "event_type_2") if c in cal.columns]
    name_columns = [c for c in ("event_name_1", "event_name_2") if c in cal.columns]
    exog = pd.DataFrame(index=date_index)
    for event_type in EVENT_TYPES:
        is_type = cal[type_columns].eq(event_type).any(axis=1)
        exog[f"event_{event_type.lower()}"] = is_type.astype(float)
    is_closed = cal[name_columns].eq(CLOSED_DAY_EVENT).any(axis=1)
    exog["is_closed"] = is_closed.astype(float)
    cashback_col = f"CASHBACK_{store_id}"
    if cashback_col in cal.columns:
        exog["cashback"] = cal[cashback_col].fillna(0).astype(float)
    exog.index.freq = "D"
    return exog


def _prices_to_daily(
    store_prices: pd.DataFrame,
    calendar: pd.DataFrame,
    date_index: pd.DatetimeIndex,
    items: pd.Index,
) -> pd.DataFrame:
    if store_prices.duplicated(["item_id", "wm_yr_wk"]).any():
        raise ValueError("prices: duplicated (item_id, wm_yr_wk) pairs")
    weekly = store_prices.pivot(
        index="wm_yr_wk", columns="item_id", values=PRICE_COLUMN
    )
    week_of_date = calendar.set_index("date")["wm_yr_wk"].reindex(date_index)
    daily = weekly.reindex(week_of_date.values)
    daily.index = date_index
    daily = daily.reindex(columns=items)
    # Weeks before an item was introduced have no price: back-fill with the
    # first known price; forward-fill covers any later gaps.
    daily = daily.ffill().bfill()
    if daily.isna().any().any():
        missing = daily.columns[daily.isna().any()].tolist()
        raise ValueError(f"prices: no price at all for items {missing}")
    daily.columns.name = None
    daily.index.freq = "D"
    return daily


def trim_leading_zeros(series: pd.Series) -> pd.Series:
    """Drop the leading period before the first non-zero observation.

    Items introduced after the start of the dataset have long runs of
    zeros that do not reflect demand and would bias trend estimates.
    """
    nonzero = series.to_numpy() != 0
    if not nonzero.any():
        return series
    first = int(np.argmax(nonzero))
    return series.iloc[first:]


def impute_closed_days(series: pd.Series, is_closed: pd.Series) -> pd.Series:
    """Replace values on closed days by linear interpolation of neighbours.

    Closed days (store does not operate, sales are zero) are known in
    advance and should be treated as a known effect, not as demand.
    """
    mask = is_closed.reindex(series.index).fillna(0).astype(bool)
    if not mask.any():
        return series
    result = series.astype(float).copy()
    result[mask] = np.nan
    return result.interpolate(limit_direction="both")
