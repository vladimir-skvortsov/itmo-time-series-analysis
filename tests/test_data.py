import numpy as np
import pandas as pd
import pytest

from conftest import ITEMS, LATE_ITEM_START, N_DAYS, START, STORE
from sales_forecast.data import (
    RawData,
    build_store_dataset,
    impute_closed_days,
    load_raw,
    trim_leading_zeros,
)


def test_load_raw_reads_three_tables(data_dir, raw_data):
    raw = load_raw(data_dir)
    assert len(raw.sales) == len(raw_data.sales)
    assert pd.api.types.is_datetime64_any_dtype(raw.calendar["date"])
    assert len(raw.prices) == len(raw_data.prices)


def test_load_raw_missing_column_raises(data_dir, tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    for name in ("shop_sales.csv", "shop_sales_dates.csv", "shop_sales_prices.csv"):
        (broken / name).write_bytes((data_dir / name).read_bytes())
    pd.read_csv(broken / "shop_sales.csv").drop(columns="cnt").to_csv(
        broken / "shop_sales.csv", index=False
    )
    with pytest.raises(ValueError, match="cnt"):
        load_raw(broken)


def test_build_dataset_shapes_and_index(raw_data):
    dataset = build_store_dataset(raw_data, STORE)
    assert dataset.store_id == STORE
    assert dataset.items == sorted(ITEMS)
    assert dataset.sales.shape == (N_DAYS, len(ITEMS))
    assert dataset.sales.index.freq == "D"
    assert dataset.sales.index[0] == START
    assert not dataset.sales.isna().any().any()
    assert dataset.exog.index.equals(dataset.sales.index)
    assert dataset.prices.index.equals(dataset.sales.index)
    assert list(dataset.prices.columns) == dataset.items


def test_build_dataset_values_match_long_table(raw_data):
    dataset = build_store_dataset(raw_data, STORE)
    long = raw_data.sales.query("store_id == @STORE and item_id == 'S1_A'")
    expected = long.set_index("date_id")["cnt"].to_numpy(dtype=float)
    np.testing.assert_array_equal(dataset.sales["S1_A"].to_numpy(), expected)


def test_unknown_store_raises(raw_data):
    with pytest.raises(ValueError, match="unknown store_id"):
        build_store_dataset(raw_data, "NOPE")


def test_missing_days_are_filled_with_zero(raw_data):
    sales = raw_data.sales
    dropped = sales.loc[~((sales["item_id"] == "S1_A") & (sales["date_id"] == 10))]
    raw = RawData(dropped, raw_data.calendar, raw_data.prices)
    dataset = build_store_dataset(raw, STORE)
    assert dataset.sales.shape[0] == N_DAYS
    assert dataset.sales.loc[START + pd.Timedelta(days=9), "S1_A"] == 0


def test_duplicate_sales_rows_raise(raw_data):
    duplicated = pd.concat([raw_data.sales, raw_data.sales.iloc[:1]])
    raw = RawData(duplicated, raw_data.calendar, raw_data.prices)
    with pytest.raises(ValueError, match="duplicated"):
        build_store_dataset(raw, STORE)


def test_calendar_gap_raises(raw_data):
    calendar = raw_data.calendar.drop(index=5)
    with pytest.raises(ValueError, match="gaps"):
        build_store_dataset(RawData(raw_data.sales, calendar, raw_data.prices), STORE)


def test_exog_columns_and_flags(raw_data):
    dataset = build_store_dataset(raw_data, STORE)
    exog = dataset.exog
    expected = {
        "event_cultural", "event_national", "event_religious", "event_sporting",
        "is_closed", "cashback",
    }
    assert set(exog.columns) == expected
    assert set(np.unique(exog.to_numpy())) <= {0.0, 1.0}
    christmas = exog.index.strftime("%m-%d") == "12-25"
    assert exog.loc[christmas, "is_closed"].eq(1).all()
    assert exog.loc[~christmas, "is_closed"].eq(0).all()
    assert exog.loc[christmas, "event_national"].eq(1).all()
    assert exog["event_sporting"].sum() == 1
    # cashback of the chosen store, not of the other one
    assert exog["cashback"].sum() == (np.arange(N_DAYS) % 10 == 0).sum()


def test_exog_without_secondary_event_columns(raw_data):
    calendar = raw_data.calendar.drop(columns=["event_name_2", "event_type_2"])
    raw = RawData(raw_data.sales, calendar, raw_data.prices)
    exog = build_store_dataset(raw, STORE).exog
    assert exog["is_closed"].sum() == (exog.index.strftime("%m-%d") == "12-25").sum()
    assert exog["event_sporting"].sum() == 1


def test_prices_are_broadcast_to_days_and_backfilled(raw_data):
    dataset = build_store_dataset(raw_data, STORE)
    prices = dataset.prices
    assert not prices.isna().any().any()
    # weekly price constant within a calendar week
    week_of = raw_data.calendar.set_index("date")["wm_yr_wk"]
    week_keys = week_of.reindex(prices.index).to_numpy()
    per_week = prices["S1_A"].groupby(week_keys).nunique()
    assert (per_week == 1).all()
    # late item: days before introduction get the first known price
    first_price = raw_data.prices.query("item_id == 'S1_B'").sort_values("wm_yr_wk")
    first_value = first_price["sell_price"].iloc[0]
    assert prices["S1_B"].iloc[:LATE_ITEM_START].eq(first_value).all()


def test_item_exog_appends_price(raw_data):
    dataset = build_store_dataset(raw_data, STORE)
    exog = dataset.item_exog("S1_B")
    assert "sell_price" in exog.columns
    assert exog.shape == (N_DAYS, dataset.exog.shape[1] + 1)
    pd.testing.assert_series_equal(
        exog["sell_price"], dataset.prices["S1_B"], check_names=False
    )


def test_trim_leading_zeros():
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    series = pd.Series([0, 0, 3, 0, 2, 0], index=index, dtype=float)
    trimmed = trim_leading_zeros(series)
    assert trimmed.index[0] == index[2]
    assert len(trimmed) == 4
    assert trimmed.index.freq == "D"


def test_trim_leading_zeros_keeps_all_zero_series():
    series = pd.Series(np.zeros(5), index=pd.date_range("2020", periods=5, freq="D"))
    pd.testing.assert_series_equal(trim_leading_zeros(series), series)


def test_trim_leading_zeros_no_leading_zeros_is_identity():
    index = pd.date_range("2020", periods=3, freq="D")
    series = pd.Series([1.0, 0.0, 2.0], index=index)
    pd.testing.assert_series_equal(trim_leading_zeros(series), series)


def test_impute_closed_days_interpolates():
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    series = pd.Series([10.0, 12.0, 0.0, 14.0, 16.0], index=index)
    closed = pd.Series([0, 0, 1, 0, 0], index=index)
    result = impute_closed_days(series, closed)
    assert result.iloc[2] == pytest.approx(13.0)
    assert result.drop(index[2]).equals(series.drop(index[2]))


def test_impute_closed_days_handles_edges_and_no_closed_days():
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    series = pd.Series([0.0, 5.0, 7.0, 0.0], index=index)
    closed = pd.Series([1, 0, 0, 1], index=index)
    result = impute_closed_days(series, closed)
    assert result.tolist() == [5.0, 5.0, 7.0, 7.0]
    pd.testing.assert_series_equal(
        impute_closed_days(series, closed * 0), series
    )
