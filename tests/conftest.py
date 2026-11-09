"""Shared fixtures: a small synthetic dataset mirroring the real schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sales_forecast.data import RawData

N_DAYS = 420
START = pd.Timestamp("2019-01-01")
STORE = "S1"
OTHER_STORE = "S2"
ITEMS = ("S1_A", "S1_B", "S1_C")
LATE_ITEM_START = 100  # S1_B is introduced on this day (0-based)


def make_calendar(n_days: int = N_DAYS) -> pd.DataFrame:
    dates = pd.date_range(START, periods=n_days, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "wm_yr_wk": 10000 + np.arange(n_days) // 7,
            "weekday": dates.day_name(),
            "wday": (np.arange(n_days) % 7) + 1,
            "month": dates.month,
            "year": dates.year,
            "event_name_1": np.nan,
            "event_type_1": np.nan,
            "event_name_2": np.nan,
            "event_type_2": np.nan,
            "date_id": np.arange(1, n_days + 1),
            f"CASHBACK_{STORE}": (np.arange(n_days) % 10 == 0).astype(int),
            f"CASHBACK_{OTHER_STORE}": 0,
        }
    )
    christmas = frame["date"].dt.strftime("%m-%d") == "12-25"
    frame.loc[christmas, ["event_name_1", "event_type_1"]] = ["Christmas", "National"]
    sporting = frame["date_id"] == 50
    frame.loc[sporting, ["event_name_1", "event_type_1"]] = ["SuperBowl", "Sporting"]
    return frame


def make_sales(calendar: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_days = len(calendar)
    t = np.arange(n_days)
    weekly = 10 + 4 * np.sin(2 * np.pi * t / 7)
    series = {
        "S1_A": rng.poisson(weekly + 0.01 * t),
        "S1_B": np.where(t >= LATE_ITEM_START, rng.poisson(3.0, n_days), 0),
        "S1_C": np.zeros(n_days, dtype=int),
    }
    closed = (calendar["date"].dt.strftime("%m-%d") == "12-25").to_numpy()
    rows = []
    for item, values in series.items():
        values = np.where(closed, 0, values)
        rows.append(
            pd.DataFrame(
                {
                    "item_id": item,
                    "store_id": STORE,
                    "date_id": calendar["date_id"],
                    "cnt": values,
                }
            )
        )
    rows.append(
        pd.DataFrame(
            {
                "item_id": "S2_X",
                "store_id": OTHER_STORE,
                "date_id": calendar["date_id"],
                "cnt": rng.poisson(5, n_days),
            }
        )
    )
    return pd.concat(rows, ignore_index=True)


def make_prices(calendar: pd.DataFrame) -> pd.DataFrame:
    weeks = calendar["wm_yr_wk"].unique()
    late_week = calendar.loc[LATE_ITEM_START, "wm_yr_wk"]
    rows = []
    for item in ITEMS:
        item_weeks = weeks[weeks >= late_week] if item == "S1_B" else weeks
        base = {"S1_A": 2.0, "S1_B": 5.0, "S1_C": 1.0}[item]
        rows.append(
            pd.DataFrame(
                {
                    "store_id": STORE,
                    "item_id": item,
                    "wm_yr_wk": item_weeks,
                    "sell_price": base + (item_weeks % 3) * 0.1,
                }
            )
        )
    rows.append(
        pd.DataFrame(
            {"store_id": OTHER_STORE, "item_id": "S2_X", "wm_yr_wk": weeks,
             "sell_price": 3.0}
        )
    )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="session")
def raw_data() -> RawData:
    calendar = make_calendar()
    return RawData(
        sales=make_sales(calendar), calendar=calendar, prices=make_prices(calendar)
    )


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory, raw_data: RawData):
    path = tmp_path_factory.mktemp("data")
    raw_data.sales.to_csv(path / "shop_sales.csv", index=False)
    raw_data.calendar.to_csv(path / "shop_sales_dates.csv", index=False)
    raw_data.prices.to_csv(path / "shop_sales_prices.csv", index=False)
    return path


@pytest.fixture
def weekly_series() -> pd.Series:
    """Deterministic series with a clean weekly pattern and a mild trend."""
    index = pd.date_range("2020-01-01", periods=140, freq="D")
    t = np.arange(len(index))
    values = 20 + 5 * np.sin(2 * np.pi * t / 7) + 0.05 * t
    return pd.Series(values, index=index, name="y")


@pytest.fixture
def weekly_exog(weekly_series: pd.Series) -> pd.DataFrame:
    index = weekly_series.index
    return pd.DataFrame(
        {"price": 2.0 + (np.arange(len(index)) // 7 % 2) * 0.5,
         "flag": (np.arange(len(index)) % 10 == 0).astype(float)},
        index=index,
    )
