import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from conftest import ITEMS, STORE  # noqa: E402
from sales_forecast import HORIZONS, SalesForecaster, SarimaModel  # noqa: E402
from sales_forecast.metrics import METRIC_NAMES, compute_metrics  # noqa: E402

FAST_MODELS = ["seasonal_naive", "holt_winters", "regression"]
SHORT_HORIZONS = {"week": 7, "month": 30}


def make_forecaster(data_dir, **kwargs) -> SalesForecaster:
    defaults = dict(models=FAST_MODELS, horizons=SHORT_HORIZONS)
    defaults.update(kwargs)
    forecaster = SalesForecaster(STORE, data_dir=data_dir, **defaults)
    forecaster.load_data()
    return forecaster


@pytest.fixture(scope="module")
def fitted(data_dir) -> SalesForecaster:
    return make_forecaster(data_dir).fit()


def test_defaults():
    forecaster = SalesForecaster(STORE)
    assert forecaster.horizons == HORIZONS
    assert forecaster.test_size == 90
    assert forecaster.model_names == ["seasonal_naive", "holt_winters", "sarima",
                                      "regression"]


def test_invalid_configuration_raises():
    with pytest.raises(ValueError, match="unknown model"):
        SalesForecaster(STORE, models=["nope"])
    with pytest.raises(ValueError, match="at least one"):
        SalesForecaster(STORE, models=[])
    with pytest.raises(ValueError, match="test_size"):
        SalesForecaster(STORE, test_size=10)
    with pytest.raises(ValueError, match="horizons"):
        SalesForecaster(STORE, horizons={})


def test_methods_require_data_and_fit(data_dir):
    forecaster = SalesForecaster(STORE, data_dir=data_dir, models=FAST_MODELS)
    with pytest.raises(RuntimeError, match="load_data"):
        forecaster.fit()
    forecaster.load_data()
    with pytest.raises(RuntimeError, match="fit"):
        forecaster.evaluate()
    with pytest.raises(RuntimeError, match="fit"):
        forecaster.predict("week")


def test_describe_items(fitted):
    summary = fitted.describe_items()
    assert set(summary.index) == set(ITEMS)
    assert summary["mean"].is_monotonic_decreasing
    assert summary.loc["S1_C", "zero_share"] == 1.0
    assert pd.isna(summary.loc["S1_C", "first_sale"])
    assert summary.loc["S1_B", "first_sale"] > summary.loc["S1_A", "first_sale"]


def test_split_default_and_explicit(fitted):
    train, test = fitted.split()
    sales = fitted.dataset.sales
    assert len(test) == fitted.test_size
    assert len(train) + len(test) == len(sales)
    assert train.index[-1] + pd.Timedelta(days=1) == test.index[0]

    train_end = sales.index[200]
    train2, test2 = fitted.split(train_end)
    assert train2.index[-1] == train_end
    assert len(test2) == fitted.test_size
    with pytest.raises(ValueError):
        fitted.split(sales.index[-5])
    with pytest.raises(ValueError):
        fitted.split(sales.index[3])


def test_fit_produces_forecasts_for_every_model_and_item(fitted):
    assert set(fitted.forecasts_) == set(FAST_MODELS)
    for frame in fitted.forecasts_.values():
        assert frame.shape == (fitted.test_size, len(ITEMS))
        assert frame.index.equals(fitted.test_.index)
        assert (frame >= 0).all().all()
        assert not frame.isna().any().any()


def test_predict_returns_prefix_of_full_forecast(fitted):
    week = fitted.predict("week")
    month = fitted.predict(30)
    for name in FAST_MODELS:
        assert len(week[name]) == 7
        pd.testing.assert_frame_equal(week[name], month[name].iloc[:7])
    with pytest.raises(ValueError):
        fitted.predict(fitted.test_size + 1)
    with pytest.raises(KeyError):
        fitted.predict("year")


def test_all_zero_item_is_forecast_as_zero(fitted):
    for frame in fitted.forecasts_.values():
        assert (frame["S1_C"] == 0).all()


def test_closed_days_are_forced_to_zero(data_dir):
    forecaster = make_forecaster(data_dir, models=["seasonal_naive"])
    sales = forecaster.dataset.sales
    christmas = sales.index[sales.index.strftime("%m-%d") == "12-25"][0]
    train_end = christmas - pd.Timedelta(days=3)
    forecaster.fit(train_end)
    forecast = forecaster.forecasts_["seasonal_naive"]
    assert christmas in forecast.index
    assert (forecast.loc[christmas] == 0).all()
    # training target had the closed day imputed, not left as zero
    window = sales.loc[:christmas + pd.Timedelta(days=5)]
    target = forecaster._prepare_target("S1_A", window)
    assert target.loc[christmas] > 0


def test_closed_day_handling_can_be_disabled(data_dir):
    forecaster = make_forecaster(
        data_dir, models=["seasonal_naive"], impute_closed_days=False
    )
    sales = forecaster.dataset.sales
    christmas = sales.index[sales.index.strftime("%m-%d") == "12-25"][0]
    target = forecaster._prepare_target("S1_A", sales)
    assert target.loc[christmas] == 0


def test_leading_zeros_trimmed_for_late_item(fitted):
    target = fitted._prepare_target("S1_B", fitted.train_)
    assert target.iloc[0] > 0
    assert len(target) < len(fitted.train_)
    untrimmed = SalesForecaster(STORE, trim_leading_zeros=False)
    untrimmed.dataset = fitted.dataset
    assert len(untrimmed._prepare_target("S1_B", fitted.train_)) == len(fitted.train_)


def test_train_window_limits_training_history(data_dir):
    forecaster = make_forecaster(data_dir, models=["seasonal_naive"], train_window=100)
    target = forecaster._prepare_target("S1_A", forecaster.dataset.sales)
    assert len(target) == 100
    assert target.index[-1] == forecaster.dataset.sales.index[-1]
    # window applies after trimming: late item has fewer non-zero days than window
    late = SalesForecaster(STORE, train_window=10_000)
    late.dataset = forecaster.dataset
    assert late._prepare_target("S1_B", forecaster.dataset.sales).iloc[0] > 0
    with pytest.raises(ValueError, match="train_window"):
        SalesForecaster(STORE, train_window=5)


def test_evaluate_layout_and_consistency(fitted):
    results = fitted.evaluate()
    assert set(results.columns) == {"model", "item", "horizon", *METRIC_NAMES}
    assert len(results) == len(FAST_MODELS) * len(ITEMS) * len(SHORT_HORIZONS)
    assert set(results["horizon"]) == set(SHORT_HORIZONS)
    # a row must equal a direct metric computation on the forecast prefix
    row = results.query(
        "model == 'regression' and item == 'S1_A' and horizon == 'week'"
    )
    expected = compute_metrics(
        fitted.test_["S1_A"].iloc[:7],
        fitted.forecasts_["regression"]["S1_A"].iloc[:7],
        fitted.train_["S1_A"],
        fitted.season_length,
    )
    for name in METRIC_NAMES:
        assert row[name].iloc[0] == pytest.approx(expected[name])


def test_evaluate_total_and_summary(fitted):
    total = fitted.evaluate_total()
    assert len(total) == len(FAST_MODELS) * len(SHORT_HORIZONS)
    assert np.isfinite(total[list(METRIC_NAMES)].to_numpy()).all()
    summary = fitted.summary("mae")
    assert list(summary.columns) == list(SHORT_HORIZONS)
    assert set(summary.index) == set(FAST_MODELS)
    with pytest.raises(ValueError):
        fitted.summary("mape")
    best = fitted.best_model("mae", "week")
    assert best == summary["week"].idxmin()


def test_seasonal_naive_mase_pooled_is_reasonable(fitted):
    """Seasonal naive on stationary-ish synthetic data has MASE around 1."""
    total = fitted.evaluate_total().set_index(["model", "horizon"])
    assert 0.5 < total.loc[("seasonal_naive", "month"), "mase"] < 2.0


def test_forecast_table(fitted):
    table = fitted.forecast_table("S1_A", "week")
    assert list(table.columns) == ["actual", *FAST_MODELS]
    assert len(table) == 7
    pd.testing.assert_series_equal(
        table["actual"], fitted.test_["S1_A"].iloc[:7], check_names=False
    )


def test_backtest_folds_and_fold_zero_equals_holdout(data_dir):
    forecaster = make_forecaster(data_dir, models=["seasonal_naive", "regression"])
    forecaster.fit()
    backtest = forecaster.backtest(n_folds=2)
    assert set(backtest["fold"]) == {0, 1}
    assert len(backtest) == 2 * 2 * len(SHORT_HORIZONS)
    fold0 = backtest.query("fold == 0").drop(columns=["fold", "train_end"])
    pd.testing.assert_frame_equal(
        fold0.reset_index(drop=True), forecaster.evaluate_total(), check_like=True
    )
    ends = backtest.groupby("fold")["train_end"].first()
    assert ends[0] - ends[1] == pd.Timedelta(days=forecaster.test_size)
    summary = forecaster.backtest_summary(backtest, "mae")
    assert summary.shape == (2, len(SHORT_HORIZONS))
    assert list(summary.columns) == list(SHORT_HORIZONS)
    fold_means = backtest.groupby(["model", "horizon"])["mae"].mean()
    assert summary.loc["regression", "week"] == pytest.approx(
        fold_means[("regression", "week")]
    )
    with pytest.raises(ValueError):
        forecaster.backtest_summary(backtest, "mape")
    with pytest.raises(ValueError):
        forecaster.backtest(n_folds=50)


def test_log_transform_option(data_dir):
    forecaster = make_forecaster(data_dir, models=["holt_winters"], log_transform=True)
    forecaster.fit()
    frame = forecaster.forecasts_["holt_winters"]
    assert (frame >= 0).all().all() and np.isfinite(frame.to_numpy()).all()


def test_custom_model_factories_and_selected_orders(data_dir):
    factories = {
        "sarima": lambda: SarimaModel(order=(1, 0, 0), seasonal_order=(0, 1, 0)),
        "seasonal_naive": SalesForecaster._resolve_models(["seasonal_naive"])[
            "seasonal_naive"
        ],
    }
    forecaster = make_forecaster(data_dir, models=factories)
    forecaster.fit()
    orders = forecaster.selected_orders()
    # S1_C is constant, so SARIMA was never estimated for it
    assert set(orders["item"]) == {"S1_A", "S1_B"}
    assert (orders["order"] == (1, 0, 0)).all()
    assert set(orders["model"]) == {"sarima"}


def test_selected_orders_none_without_sarima(fitted):
    assert fitted.selected_orders() is None


def test_decompose(fitted):
    result = fitted.decompose("S1_A")
    assert len(result.seasonal) == len(fitted.dataset.sales)
    np.testing.assert_allclose(
        result.trend + result.seasonal + result.resid,
        fitted._prepare_target("S1_A", fitted.dataset.sales),
    )


def test_plots_return_matplotlib_objects(fitted):
    assert isinstance(fitted.plot_items(), Figure)
    assert isinstance(fitted.plot_forecast("S1_A", "week"), Axes)
    assert isinstance(fitted.plot_forecast("S1_A", 30, models=["regression"]), Axes)
    assert isinstance(fitted.plot_total_metrics("wape"), Axes)
    assert isinstance(fitted.plot_item_metrics("mae", "week"), Axes)
    matplotlib.pyplot.close("all")
