import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from conftest import ITEMS, STORE  # noqa: E402
from sales_forecast import ModernSalesForecaster  # noqa: E402
from sales_forecast.metrics import METRIC_NAMES  # noqa: E402

SHORT_HORIZONS = {"week": 7, "month": 30}


def make_modern(data_dir, **kwargs) -> ModernSalesForecaster:
    defaults = dict(
        horizons=SHORT_HORIZONS,
        lgbm_num_boost_round=40,
        lgbm_early_stopping_rounds=10,
        lgbm_val_size=40,
    )
    defaults.update(kwargs)
    forecaster = ModernSalesForecaster(STORE, data_dir=data_dir, **defaults)
    forecaster.load_data()
    return forecaster


@pytest.fixture(scope="module")
def fitted(data_dir) -> ModernSalesForecaster:
    return make_modern(data_dir).fit()


def test_model_names_include_lightgbm_first(fitted):
    assert fitted.model_names[0] == "lightgbm"
    assert "seasonal_naive" in fitted.model_names


def test_fit_forecasts_cover_items_and_horizon(fitted):
    assert set(fitted.forecasts_) == {"lightgbm", "seasonal_naive"}
    frame = fitted.forecasts_["lightgbm"]
    assert frame.shape == (fitted.test_size, len(ITEMS))
    assert (frame >= 0).all().all()
    assert frame.index.equals(fitted.test_.index)


def test_evaluate_contains_lightgbm(fitted):
    total = fitted.evaluate_total()
    assert set(total["model"]) == {"lightgbm", "seasonal_naive"}
    assert set(total["horizon"]) == set(SHORT_HORIZONS)
    assert set(METRIC_NAMES) <= set(total.columns)
    summary = fitted.summary("mae")
    assert "lightgbm" in summary.index
    assert list(summary.columns) == list(SHORT_HORIZONS)


def test_predict_prefix(fitted):
    week = fitted.predict("week")
    assert len(week["lightgbm"]) == 7
    pd.testing.assert_frame_equal(
        week["lightgbm"], fitted.forecasts_["lightgbm"].iloc[:7]
    )


def test_feature_importance(fitted):
    importance = fitted.feature_importance(top_n=5)
    assert len(importance) == 5
    assert importance.is_monotonic_decreasing
    with pytest.raises(RuntimeError):
        ModernSalesForecaster(STORE).feature_importance()


def test_closed_days_forced_to_zero(data_dir):
    forecaster = make_modern(data_dir)
    sales = forecaster.dataset.sales
    christmas = sales.index[sales.index.strftime("%m-%d") == "12-25"][0]
    train_end = christmas - pd.Timedelta(days=3)
    forecaster.fit(train_end)
    assert christmas in forecaster.forecasts_["lightgbm"].index
    assert (forecaster.forecasts_["lightgbm"].loc[christmas] == 0).all()


def test_backtest_includes_lightgbm(data_dir):
    forecaster = make_modern(data_dir)
    backtest = forecaster.backtest(n_folds=2)
    assert set(backtest["model"]) == {"lightgbm", "seasonal_naive"}
    assert set(backtest["fold"]) == {0, 1}
    fold0 = backtest.query("fold == 0").drop(columns=["fold", "train_end"])
    # fold 0 must match a fresh holdout fit
    forecaster.fit()
    pd.testing.assert_frame_equal(
        fold0.sort_values(["model", "horizon"]).reset_index(drop=True),
        forecaster.evaluate_total()
        .sort_values(["model", "horizon"])
        .reset_index(drop=True),
        check_like=True,
        atol=1e-8,
    )


def test_plots(fitted):
    assert isinstance(fitted.plot_forecast("S1_A", "week"), Axes)
    assert isinstance(fitted.plot_total_metrics("wape"), Axes)
    matplotlib.pyplot.close("all")
