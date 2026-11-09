"""Global LightGBM model for multi-item daily sales forecasting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import lightgbm as lgb
import pandas as pd

from .features import (
    DEFAULT_LAGS,
    DEFAULT_WINDOWS,
    ITEM_COLUMN,
    TARGET_COLUMN,
    build_panel,
    feature_columns,
    max_history,
    row_features_from_history,
)
from .models import future_index


DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "regression_l1",
    "metric": "l1",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "n_jobs": -1,
}


class LightGBMGlobalModel:
    """One LightGBM model for all items of a store (panel / global model).

    Training rows are (date, item) pairs with lag, rolling, calendar and
    exogenous features. Multi-step forecasts are produced recursively:
    each predicted day is appended to the history and used as a lag for
    the next step.
    """

    name = "lightgbm"

    def __init__(
        self,
        lags: Sequence[int] = DEFAULT_LAGS,
        windows: Sequence[int] = DEFAULT_WINDOWS,
        params: Mapping[str, Any] | None = None,
        num_boost_round: int = 800,
        early_stopping_rounds: int = 50,
        val_size: int = 90,
        random_state: int = 42,
    ) -> None:
        if val_size < 1:
            raise ValueError("val_size must be >= 1")
        if num_boost_round < 1:
            raise ValueError("num_boost_round must be >= 1")
        self.lags = tuple(lags)
        self.windows = tuple(windows)
        self.params = {**DEFAULT_LGBM_PARAMS, **dict(params or {})}
        self.params["seed"] = random_state
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.val_size = val_size
        self.random_state = random_state

        self.model_: lgb.Booster | None = None
        self.feature_names_: list[str] = []
        self.items_: list[str] = []
        self.history_: pd.DataFrame | None = None
        self.exog_columns_: list[str] = []
        self.best_iteration_: int | None = None
        self.feature_importance_: pd.Series | None = None

    @property
    def is_fitted(self) -> bool:
        return self.model_ is not None

    def fit(
        self,
        sales: pd.DataFrame,
        exog: pd.DataFrame,
        prices: pd.DataFrame,
        first_sale: Mapping[str, pd.Timestamp] | None = None,
    ) -> "LightGBMGlobalModel":
        """Fit on a wide sales matrix ending at the last training day."""
        self._validate_frames(sales, exog, prices)
        if len(sales) <= self.val_size + max_history(self.lags, self.windows):
            raise ValueError("training series is too short for the chosen val_size")

        self.items_ = list(sales.columns)
        self.exog_columns_ = list(exog.columns)
        self.feature_names_ = feature_columns(
            self.lags, self.windows, self.exog_columns_
        )
        self.history_ = sales.astype(float).copy()
        self.history_.index.freq = sales.index.freq or "D"

        panel = build_panel(
            sales,
            exog,
            prices,
            lags=self.lags,
            windows=self.windows,
            dropna=True,
            first_sale=dict(first_sale) if first_sale is not None else None,
        )
        if panel.empty:
            raise ValueError("feature panel is empty after dropna")

        split_date = sales.index[-self.val_size]
        train_panel = panel.loc[panel["date"] < split_date]
        valid_panel = panel.loc[panel["date"] >= split_date]
        if train_panel.empty or valid_panel.empty:
            raise ValueError("train/validation split of the panel is empty")

        categorical = [ITEM_COLUMN]
        train_x = self._design(train_panel)
        valid_x = self._design(valid_panel)
        train_set = lgb.Dataset(
            train_x,
            label=train_panel[TARGET_COLUMN].to_numpy(),
            categorical_feature=categorical,
            free_raw_data=False,
        )
        valid_set = lgb.Dataset(
            valid_x,
            label=valid_panel[TARGET_COLUMN].to_numpy(),
            categorical_feature=categorical,
            reference=train_set,
            free_raw_data=False,
        )
        callbacks = [
            lgb.early_stopping(self.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        self.model_ = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=callbacks,
        )
        self.best_iteration_ = self.model_.best_iteration
        importance = self.model_.feature_importance(importance_type="gain")
        self.feature_importance_ = pd.Series(
            importance, index=self.feature_names_, name="gain"
        ).sort_values(ascending=False)
        return self

    def predict(
        self,
        horizon: int,
        future_exog: pd.DataFrame,
        future_prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Recursive multi-step forecast for all items."""
        if not self.is_fitted:
            raise RuntimeError("model is not fitted")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if len(future_exog) < horizon or len(future_prices) < horizon:
            raise ValueError("future exog/prices must cover the horizon")
        if list(future_prices.columns) != self.items_:
            raise ValueError("future_prices columns must match training items")
        if list(future_exog.columns) != self.exog_columns_:
            raise ValueError("future_exog columns must match training exog")

        index = future_index(self.history_.index, horizon)
        if not future_exog.index[:horizon].equals(index):
            # allow any index as long as length matches; reindex to expected dates
            future_exog = future_exog.reindex(index)
            future_prices = future_prices.reindex(index)
            if future_exog.isna().any().any() or future_prices.isna().any().any():
                raise ValueError("future exog/prices are missing for forecast dates")

        history = {
            item: self.history_[item].astype(float).tolist() for item in self.items_
        }
        rows = []
        for date in index:
            day_pred: dict[str, float] = {}
            feature_rows = []
            for item in self.items_:
                hist = pd.Series(history[item])
                features = row_features_from_history(
                    hist,
                    date,
                    future_exog.loc[date],
                    float(future_prices.loc[date, item]),
                    item,
                    lags=self.lags,
                    windows=self.windows,
                )
                feature_rows.append(features)
            design = pd.DataFrame(feature_rows)[self.feature_names_]
            design[ITEM_COLUMN] = pd.Categorical(
                design[ITEM_COLUMN], categories=self.items_
            )
            preds = self.model_.predict(
                design, num_iteration=self.best_iteration_
            )
            for item, value in zip(self.items_, preds):
                value = float(max(value, 0.0))
                day_pred[item] = value
                history[item].append(value)
            rows.append(day_pred)
        forecast = pd.DataFrame(rows, index=index)[self.items_]
        forecast.index.freq = "D"
        return forecast

    def _design(self, panel: pd.DataFrame) -> pd.DataFrame:
        design = panel[self.feature_names_].copy()
        design[ITEM_COLUMN] = pd.Categorical(
            design[ITEM_COLUMN], categories=self.items_
        )
        return design

    @staticmethod
    def _validate_frames(
        sales: pd.DataFrame, exog: pd.DataFrame, prices: pd.DataFrame
    ) -> None:
        if sales.isna().any().any():
            raise ValueError("sales must not contain NaN")
        if exog.isna().any().any() or prices.isna().any().any():
            raise ValueError("exog and prices must not contain NaN")
        if not sales.columns.equals(prices.columns):
            raise ValueError("sales and prices columns differ")
        if not sales.index.equals(exog.index) or not sales.index.equals(prices.index):
            raise ValueError("sales, exog and prices indexes differ")
        if not isinstance(sales.index, pd.DatetimeIndex):
            raise TypeError("sales must have a DatetimeIndex")
