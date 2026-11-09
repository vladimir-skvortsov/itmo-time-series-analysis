import numpy as np
import pandas as pd
import pytest

from sales_forecast.features import (
    DEFAULT_LAGS,
    DEFAULT_WINDOWS,
    ITEM_COLUMN,
    build_item_features,
    build_panel,
    calendar_features,
    lag_features,
    max_history,
    rolling_features,
    row_features_from_history,
)


@pytest.fixture
def daily_index() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=60, freq="D")


@pytest.fixture
def sales_series(daily_index) -> pd.Series:
    values = np.arange(len(daily_index), dtype=float)
    return pd.Series(values, index=daily_index, name="A")


def test_max_history():
    assert max_history([1, 7], [14]) == 14
    with pytest.raises(ValueError):
        max_history([], [])


def test_lag_features_are_pure_shifts(sales_series):
    lags = lag_features(sales_series, [1, 7])
    pd.testing.assert_series_equal(
        lags["lag_1"], sales_series.shift(1), check_names=False
    )
    pd.testing.assert_series_equal(
        lags["lag_7"], sales_series.shift(7), check_names=False
    )
    assert lags.isna().sum().to_dict() == {"lag_1": 1, "lag_7": 7}


def test_rolling_features_exclude_current_value(sales_series):
    rolled = rolling_features(sales_series, [3])
    # at t=5 (0-based), uses values [2,3,4] after shift -> mean 3
    assert rolled["roll_mean_3"].iloc[5] == pytest.approx(3.0)
    assert rolled["roll_std_3"].iloc[5] == pytest.approx(
        np.std([2.0, 3.0, 4.0], ddof=0)
    )
    assert rolled.isna().sum().iloc[0] == 3


def test_calendar_features_shape(daily_index):
    cal = calendar_features(daily_index)
    assert list(cal.columns) == ["dow", "month", "weekofyear", "sin_doy", "cos_doy"]
    assert len(cal) == len(daily_index)
    assert cal["dow"].between(0, 6).all()


def test_build_item_features_no_target_leakage(sales_series, daily_index):
    exog = pd.DataFrame({"flag": np.zeros(len(daily_index))}, index=daily_index)
    price = pd.Series(2.0, index=daily_index)
    frame = build_item_features(sales_series, exog, price, lags=[1, 2], windows=[3])
    complete = frame.dropna()
    for date in complete.index:
        assert frame.loc[date, "lag_1"] == sales_series.loc[date - pd.Timedelta(days=1)]
        assert frame.loc[date, "lag_2"] == sales_series.loc[date - pd.Timedelta(days=2)]
        # rolling window never includes y[date]
        start = date - pd.Timedelta(days=3)
        end = date - pd.Timedelta(days=1)
        expected = sales_series.loc[start:end]
        assert frame.loc[date, "roll_mean_3"] == pytest.approx(expected.mean())


def test_build_panel_and_first_sale_filter(daily_index):
    sales = pd.DataFrame(
        {
            "A": np.arange(len(daily_index), dtype=float),
            "B": np.concatenate(
                [np.zeros(20), np.arange(len(daily_index) - 20, dtype=float)]
            ),
        },
        index=daily_index,
    )
    exog = pd.DataFrame({"flag": 0.0}, index=daily_index)
    prices = pd.DataFrame({"A": 1.0, "B": 2.0}, index=daily_index)
    first_sale = {"A": daily_index[0], "B": daily_index[20]}
    panel = build_panel(
        sales, exog, prices, lags=[1, 2], windows=[3], first_sale=first_sale
    )
    assert set(panel[ITEM_COLUMN]) == {"A", "B"}
    assert panel.loc[panel[ITEM_COLUMN] == "B", "date"].min() >= daily_index[20]
    assert not panel.isna().any().any()
    needed = max_history([1, 2], [3])
    assert panel.loc[panel[ITEM_COLUMN] == "A", "date"].min() == daily_index[needed]


def test_row_features_match_panel_row(daily_index):
    sales = pd.Series(np.arange(40, dtype=float), index=daily_index[:40])
    exog = pd.DataFrame({"flag": 1.0}, index=daily_index[:40])
    price = pd.Series(3.5, index=daily_index[:40])
    frame = build_item_features(sales, exog, price, lags=[1, 7], windows=[7]).dropna()
    date = frame.index[5]
    history = sales.loc[: date - pd.Timedelta(days=1)]
    row = row_features_from_history(
        history, date, exog.loc[date], 3.5, "A", lags=[1, 7], windows=[7]
    )
    for key in ["lag_1", "lag_7", "roll_mean_7", "roll_std_7", "flag", "dow"]:
        assert row[key] == pytest.approx(float(frame.loc[date, key]))
    assert row[ITEM_COLUMN] == "A"


def test_row_features_require_enough_history(sales_series):
    with pytest.raises(ValueError):
        row_features_from_history(
            sales_series.iloc[:5],
            sales_series.index[5],
            pd.Series({"flag": 0.0}),
            1.0,
            "A",
            lags=DEFAULT_LAGS,
            windows=DEFAULT_WINDOWS,
        )
