import numpy as np
import pandas as pd
import pytest

from sales_forecast.models import (
    DEFAULT_MODELS,
    HoltWintersModel,
    Log1pModel,
    RegressionModel,
    SarimaModel,
    SeasonalNaiveModel,
    future_index,
    make_deterministic_features,
)

HORIZON = 10


def fast_sarima() -> SarimaModel:
    return SarimaModel(order=(1, 0, 0), seasonal_order=(0, 1, 0))


def all_models():
    return [
        SeasonalNaiveModel(),
        HoltWintersModel(),
        fast_sarima(),
        RegressionModel(),
        Log1pModel(HoltWintersModel()),
        Log1pModel(RegressionModel()),
    ]


def exog_for(model, index, weekly_exog):
    if not model.uses_exog:
        return None
    t = np.arange(len(index))
    return pd.DataFrame(
        {"price": 2.0 + (t // 7 % 2) * 0.5, "flag": (t % 10 == 0).astype(float)},
        index=index,
    )


def test_future_index_continues_daily_index(weekly_series):
    index = future_index(weekly_series.index, 3)
    assert index[0] == weekly_series.index[-1] + pd.Timedelta(days=1)
    assert len(index) == 3 and index.freq == "D"


def test_future_index_without_frequency_raises():
    irregular = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-05"])
    with pytest.raises(ValueError):
        future_index(irregular, 2)


@pytest.mark.parametrize("model", all_models(), ids=lambda m: type(m).__name__)
def test_predict_shape_index_and_name(model, weekly_series, weekly_exog):
    exog = weekly_exog if model.uses_exog else None
    model.fit(weekly_series, exog)
    index = future_index(weekly_series.index, HORIZON)
    forecast = model.predict(HORIZON, exog_for(model, index, weekly_exog))
    assert isinstance(forecast, pd.Series)
    assert len(forecast) == HORIZON
    assert forecast.index.equals(index)
    assert forecast.name == model.name
    assert np.isfinite(forecast).all()


@pytest.mark.parametrize("model", all_models(), ids=lambda m: type(m).__name__)
def test_models_track_clean_weekly_pattern(model, weekly_series, weekly_exog):
    """On a noise-free series every model must beat a flat mean forecast."""
    exog = weekly_exog if model.uses_exog else None
    model.fit(weekly_series, exog)
    index = future_index(weekly_series.index, 14)
    t = np.arange(len(weekly_series), len(weekly_series) + 14)
    truth = 20 + 5 * np.sin(2 * np.pi * t / 7) + 0.05 * t
    forecast = model.predict(14, exog_for(model, index, weekly_exog)).to_numpy()
    flat_error = np.mean(np.abs(truth - weekly_series.mean()))
    assert np.mean(np.abs(truth - forecast)) < flat_error / 2


@pytest.mark.parametrize("model", all_models(), ids=lambda m: type(m).__name__)
def test_constant_series_is_predicted_as_constant(model, weekly_exog):
    index = weekly_exog.index
    constant = pd.Series(3.0, index=index)
    exog = weekly_exog if model.uses_exog else None
    model.fit(constant, exog)
    future = future_index(index, HORIZON)
    forecast = model.predict(HORIZON, exog_for(model, future, weekly_exog))
    assert (forecast == 3.0).all()


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        SeasonalNaiveModel().predict(3)


def test_invalid_horizon_raises(weekly_series):
    model = SeasonalNaiveModel().fit(weekly_series)
    with pytest.raises(ValueError):
        model.predict(0)


def test_series_with_nan_raises(weekly_series):
    broken = weekly_series.copy()
    broken.iloc[3] = np.nan
    with pytest.raises(ValueError):
        SeasonalNaiveModel().fit(broken)


def test_non_datetime_index_raises():
    with pytest.raises(TypeError):
        SeasonalNaiveModel().fit(pd.Series([1.0, 2.0, 3.0]))


def test_seasonal_naive_repeats_last_season(weekly_series):
    model = SeasonalNaiveModel(season_length=7).fit(weekly_series)
    forecast = model.predict(16).to_numpy()
    last = weekly_series.to_numpy()[-7:]
    np.testing.assert_allclose(forecast[:7], last)
    np.testing.assert_allclose(forecast[7:14], last)
    np.testing.assert_allclose(forecast[14:], last[:2])


def test_seasonal_naive_too_short_series_raises():
    short = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020", periods=3, freq="D"))
    with pytest.raises(ValueError):
        SeasonalNaiveModel(season_length=7).fit(short)


def test_holt_winters_too_short_series_raises():
    index = pd.date_range("2020", periods=10, freq="D")
    short = pd.Series(np.arange(10.0), index=index)
    with pytest.raises(ValueError):
        HoltWintersModel(seasonal_periods=7).fit(short)


def test_sarima_fixed_order_skips_search(weekly_series):
    model = fast_sarima().fit(weekly_series)
    assert model.selected_order_ == (1, 0, 0)
    assert model.selected_seasonal_order_ == (0, 1, 0)
    assert model.aic_table_ == {}


def test_sarima_auto_selection_stays_within_grid(weekly_series):
    model = SarimaModel(max_p=1, max_q=1, max_seasonal_p=1, max_seasonal_q=0,
                        maxiter=20)
    model.fit(weekly_series)
    p, d, q = model.selected_order_
    big_p, big_d, big_q = model.selected_seasonal_order_
    assert p <= 1 and q <= 1 and d in (0, 1)
    assert big_p <= 1 and big_q == 0 and big_d == 1
    # stage 1: 2 seasonal candidates, stage 2: 4 non-seasonal, (1,d,1) shared
    assert len(model.aic_table_) == 2 + 4 - 1
    assert all(np.isfinite(v) for v in model.aic_table_.values())


def test_sarima_train_window_limits_history(weekly_series):
    model = SarimaModel(order=(0, 0, 0), seasonal_order=(0, 1, 0), train_window=35)
    model.fit(weekly_series)
    assert model._result.nobs == 35


def test_sarima_too_short_series_raises():
    index = pd.date_range("2020", periods=14, freq="D")
    with pytest.raises(ValueError):
        fast_sarima().fit(pd.Series(np.arange(14.0), index=index))


def test_deterministic_features_are_consistent_across_calls():
    origin = pd.Timestamp("2020-01-01")
    full = pd.date_range(origin, periods=30, freq="D")
    features_full = make_deterministic_features(full, origin, ((7, 2),))
    features_tail = make_deterministic_features(full[20:], origin, ((7, 2),))
    pd.testing.assert_frame_equal(features_full.iloc[20:], features_tail)
    assert list(features_full.columns) == [
        "trend", "sin_7_1", "cos_7_1", "sin_7_2", "cos_7_2"
    ]
    assert features_full["trend"].iloc[0] == 0


def test_regression_requires_exog(weekly_series, weekly_exog):
    with pytest.raises(ValueError):
        RegressionModel().fit(weekly_series)
    model = RegressionModel().fit(weekly_series, weekly_exog)
    with pytest.raises(ValueError):
        model.predict(HORIZON)


def test_regression_exog_index_mismatch_raises(weekly_series, weekly_exog):
    with pytest.raises(ValueError, match="index"):
        RegressionModel().fit(weekly_series, weekly_exog.iloc[1:])
    model = RegressionModel().fit(weekly_series, weekly_exog)
    wrong_index = weekly_exog.iloc[:HORIZON]  # past dates, not future ones
    with pytest.raises(ValueError, match="index"):
        model.predict(HORIZON, wrong_index)


def test_regression_missing_exog_column_raises(weekly_series, weekly_exog):
    model = RegressionModel().fit(weekly_series, weekly_exog)
    future = future_index(weekly_series.index, HORIZON)
    with pytest.raises(ValueError, match="missing columns"):
        model.predict(HORIZON, pd.DataFrame({"price": 1.0}, index=future))


def test_regression_exog_with_nan_raises(weekly_series, weekly_exog):
    broken = weekly_exog.copy()
    broken.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        RegressionModel().fit(weekly_series, broken)


def test_regression_uses_exogenous_effect(weekly_exog):
    """A series driven by an exogenous flag is predicted from that flag."""
    index = weekly_exog.index
    y = pd.Series(10 + 20 * weekly_exog["flag"], index=index)
    model = RegressionModel(fourier_periods=(), alpha=1e-6).fit(y, weekly_exog)
    future = future_index(index, HORIZON)
    t = np.arange(len(index), len(index) + HORIZON)
    exog_future = pd.DataFrame(
        {"price": 2.0, "flag": (t % 10 == 0).astype(float)}, index=future
    )
    forecast = model.predict(HORIZON, exog_future)
    expected = 10 + 20 * exog_future["flag"]
    np.testing.assert_allclose(forecast, expected, atol=0.5)


def test_log1p_wrapper_inverts_transform(weekly_series):
    inner = SeasonalNaiveModel()
    wrapped = Log1pModel(inner).fit(weekly_series)
    plain = SeasonalNaiveModel().fit(weekly_series)
    np.testing.assert_allclose(wrapped.predict(7), plain.predict(7), rtol=1e-10)
    assert wrapped.name == inner.name
    assert wrapped.uses_exog == inner.uses_exog


def test_log1p_wrapper_rejects_negative_values():
    index = pd.date_range("2020", periods=20, freq="D")
    negative = pd.Series(np.arange(-5.0, 15.0), index=index)
    with pytest.raises(ValueError):
        Log1pModel(SeasonalNaiveModel()).fit(negative)


def test_log1p_wrapper_propagates_exog_flag(weekly_series, weekly_exog):
    wrapped = Log1pModel(RegressionModel())
    assert wrapped.uses_exog
    with pytest.raises(ValueError):
        wrapped.fit(weekly_series)


def test_default_models_registry_builds_fresh_instances():
    first = {name: factory() for name, factory in DEFAULT_MODELS.items()}
    second = {name: factory() for name, factory in DEFAULT_MODELS.items()}
    for name in DEFAULT_MODELS:
        assert first[name] is not second[name]
        assert first[name].name == name
