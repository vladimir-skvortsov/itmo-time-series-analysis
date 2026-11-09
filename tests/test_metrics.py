import math

import numpy as np
import pandas as pd
import pytest

from sales_forecast.metrics import (
    METRIC_NAMES,
    aggregate_metrics,
    compute_metrics,
    mae,
    mase,
    rmse,
    wape,
)

Y_TRUE = np.array([10.0, 0.0, 5.0, 5.0])
Y_PRED = np.array([8.0, 1.0, 5.0, 9.0])  # abs errors: 2, 1, 0, 4


def test_mae_known_value():
    assert mae(Y_TRUE, Y_PRED) == pytest.approx(7 / 4)


def test_rmse_known_value():
    assert rmse(Y_TRUE, Y_PRED) == pytest.approx(math.sqrt((4 + 1 + 0 + 16) / 4))


def test_wape_known_value():
    assert wape(Y_TRUE, Y_PRED) == pytest.approx(7 / 20)


def test_wape_is_nan_when_actuals_are_all_zero():
    assert math.isnan(wape([0.0, 0.0], [1.0, 2.0]))


def test_perfect_forecast_gives_zero_errors():
    assert mae(Y_TRUE, Y_TRUE) == 0
    assert rmse(Y_TRUE, Y_TRUE) == 0
    assert wape(Y_TRUE, Y_TRUE) == 0


def test_mase_scales_by_seasonal_naive_in_sample_error():
    train = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])  # season 2 -> scale 0
    assert math.isnan(mase(Y_TRUE, Y_PRED, train, season_length=2))
    # season 1: naive diffs are all 1 -> scale 1 -> MASE == MAE
    assert mase(Y_TRUE, Y_PRED, train, season_length=1) == pytest.approx(7 / 4)


def test_mase_equal_to_one_for_seasonal_naive_on_training_like_data():
    train = np.array([3.0, 7.0, 3.0, 7.0, 3.0, 7.0, 3.0, 7.0])
    # the naive forecast of the test block makes the same error as in-sample
    assert mase([3.0, 7.0], [7.0, 3.0], train, season_length=1) == pytest.approx(1.0)


def test_mase_rejects_bad_inputs():
    with pytest.raises(ValueError):
        mase(Y_TRUE, Y_PRED, [1.0, 2.0], season_length=2)
    with pytest.raises(ValueError):
        mase(Y_TRUE, Y_PRED, [1.0, 2.0, 3.0], season_length=0)


@pytest.mark.parametrize("func", [mae, rmse, wape])
def test_shape_mismatch_raises(func):
    with pytest.raises(ValueError):
        func([1.0, 2.0], [1.0])


@pytest.mark.parametrize("func", [mae, rmse, wape])
def test_nan_input_raises(func):
    with pytest.raises(ValueError):
        func([1.0, np.nan], [1.0, 2.0])


@pytest.mark.parametrize("func", [mae, rmse, wape])
def test_empty_input_raises(func):
    with pytest.raises(ValueError):
        func([], [])


def test_compute_metrics_returns_all_names():
    train = np.arange(20, dtype=float)
    result = compute_metrics(Y_TRUE, Y_PRED, train, season_length=7)
    assert set(result) == set(METRIC_NAMES)
    assert result["mae"] == pytest.approx(mae(Y_TRUE, Y_PRED))
    assert result["mase"] == pytest.approx(mase(Y_TRUE, Y_PRED, train, 7))


def test_aggregate_metrics_matches_pooled_manual_computation():
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    y_true = pd.DataFrame({"a": Y_TRUE, "b": Y_TRUE * 2}, index=index)
    y_pred = pd.DataFrame({"b": Y_PRED * 2, "a": Y_PRED}, index=index)  # shuffled
    train_index = pd.date_range("2019-12-01", periods=20, freq="D")
    y_train = pd.DataFrame(
        {"a": np.arange(20, dtype=float), "b": np.arange(20, dtype=float) * 2},
        index=train_index,
    )
    result = aggregate_metrics(y_true, y_pred, y_train, season_length=7)

    all_true = np.concatenate([Y_TRUE, Y_TRUE * 2])
    all_pred = np.concatenate([Y_PRED, Y_PRED * 2])
    assert result["mae"] == pytest.approx(mae(all_true, all_pred))
    assert result["rmse"] == pytest.approx(rmse(all_true, all_pred))
    assert result["wape"] == pytest.approx(wape(all_true, all_pred))
    # season 7 on arange(20) and 2*arange(20): 13 diffs of 7 and 13 of 14
    expected_scale = np.mean(np.concatenate([np.full(13, 7.0), np.full(13, 14.0)]))
    assert result["mase"] == pytest.approx(mae(all_true, all_pred) / expected_scale)


def test_aggregate_metrics_nan_when_no_sales_and_no_scale():
    index = pd.date_range("2020-01-01", periods=2, freq="D")
    zeros = pd.DataFrame({"a": [0.0, 0.0]}, index=index)
    train = pd.DataFrame({"a": np.zeros(10)}, index=pd.date_range("2019", periods=10))
    result = aggregate_metrics(zeros, zeros + 1, train, season_length=1)
    assert math.isnan(result["wape"]) and math.isnan(result["mase"])
    assert result["mae"] == 1.0
