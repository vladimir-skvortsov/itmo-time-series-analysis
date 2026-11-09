import numpy as np
import pandas as pd
import pytest

from sales_forecast.boosting import LightGBMGlobalModel
from sales_forecast.models import future_index


def _synthetic_store(n_days: int = 400, n_items: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2018-01-01", periods=n_days, freq="D")
    sales = {}
    prices = {}
    for i in range(n_items):
        t = np.arange(n_days)
        weekly = 8 + 3 * np.sin(2 * np.pi * (t + i) / 7)
        trend = 0.01 * t
        noise = rng.normal(0, 0.7, n_days)
        sales[f"I{i}"] = np.clip(weekly + trend + noise, 0, None)
        prices[f"I{i}"] = 2.0 + 0.1 * i
    sales = pd.DataFrame(sales, index=index)
    prices = pd.DataFrame(prices, index=index)
    exog = pd.DataFrame(
        {
            "event_national": (index.dayofweek == 6).astype(float),
            "is_closed": 0.0,
            "cashback": (np.arange(n_days) % 11 == 0).astype(float),
        },
        index=index,
    )
    return sales, exog, prices


def test_fit_predict_shapes_and_non_negative():
    sales, exog, prices = _synthetic_store()
    train = sales.iloc[:-90]
    model = LightGBMGlobalModel(
        num_boost_round=60, early_stopping_rounds=10, val_size=30
    )
    model.fit(train, exog.loc[train.index], prices.loc[train.index])
    future_index_ = future_index(train.index, 90)
    forecast = model.predict(
        90, exog.reindex(future_index_), prices.reindex(future_index_)
    )
    assert forecast.shape == (90, train.shape[1])
    assert forecast.index.equals(future_index_)
    assert (forecast >= 0).all().all()
    assert model.best_iteration_ is not None
    assert model.feature_importance_ is not None
    assert model.feature_importance_.sum() > 0


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        LightGBMGlobalModel().predict(
            3, pd.DataFrame(), pd.DataFrame()
        )


def test_too_short_series_raises():
    sales, exog, prices = _synthetic_store(n_days=40)
    model = LightGBMGlobalModel(val_size=20)
    with pytest.raises(ValueError, match="too short"):
        model.fit(sales, exog, prices)


def test_beats_flat_mean_on_clean_weekly_pattern():
    sales, exog, prices = _synthetic_store(n_days=500, seed=1)
    train, test = sales.iloc[:-60], sales.iloc[-60:]
    model = LightGBMGlobalModel(
        num_boost_round=120, early_stopping_rounds=20, val_size=40
    )
    model.fit(train, exog.loc[train.index], prices.loc[train.index])
    forecast = model.predict(
        60, exog.reindex(test.index), prices.reindex(test.index)
    )
    mae_model = np.mean(np.abs(test.to_numpy() - forecast.to_numpy()))
    mae_flat = np.mean(np.abs(test.to_numpy() - train.mean().to_numpy()))
    assert mae_model < mae_flat


def test_first_sale_filter_is_respected():
    sales, exog, prices = _synthetic_store(n_days=300, n_items=2)
    sales["I1"].iloc[:80] = 0.0
    first = {"I0": sales.index[0], "I1": sales.index[80]}
    model = LightGBMGlobalModel(
        num_boost_round=40, early_stopping_rounds=10, val_size=30
    )
    model.fit(
        sales.iloc[:-30],
        exog.iloc[:-30],
        prices.iloc[:-30],
        first_sale=first,
    )
    assert model.is_fitted
    # history still keeps full training matrix for recursive lags
    assert len(model.history_) == len(sales) - 30
