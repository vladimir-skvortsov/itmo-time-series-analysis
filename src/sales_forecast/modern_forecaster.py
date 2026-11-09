"""Forecasting pipeline with a global LightGBM model (lab 2)."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .boosting import DEFAULT_LGBM_PARAMS, LightGBMGlobalModel
from .data import impute_closed_days
from .forecaster import SalesForecaster
from .models import ModelFactory

logger = logging.getLogger(__name__)


class ModernSalesForecaster(SalesForecaster):
    """Same API as :class:`SalesForecaster`, plus a global LightGBM model.

    Classical baseline models (default: seasonal naive) are still trained
    per item. LightGBM is trained once on the pooled panel of all items
    and forecasts every item recursively for the full test horizon.
    """

    def __init__(
        self,
        store_id: str,
        data_dir: str | Path = "data",
        baseline_models: Mapping[str, ModelFactory] | Iterable[str] | None = (
            "seasonal_naive",
        ),
        horizons: Mapping[str, int] | None = None,
        test_size: int | None = None,
        season_length: int = 7,
        trim_leading_zeros: bool = True,
        impute_closed_days: bool = True,
        train_window: int | None = None,
        lgbm_params: Mapping[str, Any] | None = None,
        lgbm_num_boost_round: int = 800,
        lgbm_early_stopping_rounds: int = 50,
        lgbm_val_size: int = 90,
        random_state: int = 42,
    ) -> None:
        baselines = (
            list(baseline_models)
            if baseline_models is not None
            else ["seasonal_naive"]
        )
        if not baselines:
            # parent requires at least one classical factory; seasonal naive
            # is cheap and useful as a MASE reference in the report
            baselines = ["seasonal_naive"]
        super().__init__(
            store_id=store_id,
            data_dir=data_dir,
            models=baselines,
            horizons=horizons,
            test_size=test_size,
            season_length=season_length,
            log_transform=False,
            trim_leading_zeros=trim_leading_zeros,
            impute_closed_days=impute_closed_days,
            train_window=train_window,
        )
        self.lgbm_params = {**DEFAULT_LGBM_PARAMS, **dict(lgbm_params or {})}
        self.lgbm_num_boost_round = lgbm_num_boost_round
        self.lgbm_early_stopping_rounds = lgbm_early_stopping_rounds
        self.lgbm_val_size = lgbm_val_size
        self.random_state = random_state
        self.lgbm_: LightGBMGlobalModel | None = None

    @property
    def model_names(self) -> list[str]:
        return ["lightgbm", *self.model_factories]

    def _fit_all(self, train: pd.DataFrame):
        """Fit only classical baselines; LightGBM is trained separately."""
        from .forecaster import FittedModels

        names = list(self.model_factories)
        fitted: FittedModels = {name: {} for name in names}
        for item in train.columns:
            y = self._prepare_target(item, train)
            exog = self._exog_for(item, y.index)
            for name in names:
                logger.info("fit %s on %s", name, item)
                model = self._make_model(name)
                fitted[name][item] = model.fit(
                    y, exog if model.uses_exog else None
                )
        return fitted

    def _prepare_train_matrix(self, train: pd.DataFrame) -> pd.DataFrame:
        """Wide training matrix with closed-day imputation and optional window."""
        dataset = self._require_dataset()
        prepared = train.astype(float).copy()
        if self.impute_closed_days:
            closed = dataset.exog["is_closed"]
            for item in prepared.columns:
                prepared[item] = impute_closed_days(prepared[item], closed)
        if self.train_window is not None:
            prepared = prepared.iloc[-self.train_window:]
        prepared.index.freq = "D"
        return prepared

    def _first_sale_map(self, sales: pd.DataFrame) -> dict[str, pd.Timestamp]:
        if not self.trim_leading_zeros:
            return {}
        first = {}
        for item in sales.columns:
            nonzero = sales[item].to_numpy() != 0
            if nonzero.any():
                first[item] = sales.index[int(nonzero.argmax())]
        return first

    def _fit_lightgbm(self, train: pd.DataFrame) -> LightGBMGlobalModel:
        dataset = self._require_dataset()
        prepared = self._prepare_train_matrix(train)
        exog = dataset.exog.reindex(prepared.index)
        prices = dataset.prices.reindex(prepared.index)
        model = LightGBMGlobalModel(
            params=self.lgbm_params,
            num_boost_round=self.lgbm_num_boost_round,
            early_stopping_rounds=self.lgbm_early_stopping_rounds,
            val_size=min(self.lgbm_val_size, max(14, len(prepared) // 5)),
            random_state=self.random_state,
        )
        logger.info("fit lightgbm on %d days x %d items", *prepared.shape)
        return model.fit(
            prepared,
            exog,
            prices,
            first_sale=self._first_sale_map(prepared),
        )

    def _predict_lightgbm(
        self, model: LightGBMGlobalModel, horizon: int, index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        dataset = self._require_dataset()
        future_exog = dataset.exog.reindex(index)
        future_prices = dataset.prices.reindex(index)
        if future_exog.isna().any().any() or future_prices.isna().any().any():
            raise ValueError("exogenous features are not available for forecast dates")
        forecast = model.predict(horizon, future_exog, future_prices)
        return self._postprocess(forecast)

    def fit(self, train_end: pd.Timestamp | None = None) -> "ModernSalesForecaster":
        """Fit LightGBM (global) and baseline classical models."""
        self.train_, self.test_ = self.split(train_end)
        self.lgbm_ = self._fit_lightgbm(self.train_)
        self.fitted_models_ = self._fit_all(self.train_)
        classical = self._predict_all(
            self.fitted_models_, self.test_size, self.test_.index
        )
        lightgbm_forecast = self._predict_lightgbm(
            self.lgbm_, self.test_size, self.test_.index
        )
        self.forecasts_ = {"lightgbm": lightgbm_forecast, **classical}
        return self

    def backtest(self, n_folds: int = 3, step: int | None = None) -> pd.DataFrame:
        """Rolling-origin evaluation including LightGBM."""
        if n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        sales = self._require_dataset().sales
        step = step or self.test_size
        rows = []
        for fold in range(n_folds):
            end_position = len(sales) - fold * step - self.test_size
            if end_position < 2 * self.season_length:
                raise ValueError("not enough history for the requested folds")
            train_end = sales.index[end_position - 1]
            train, test = self.split(train_end)
            logger.info("backtest fold %d: train end %s", fold, train_end.date())
            lgbm = self._fit_lightgbm(train)
            fitted = self._fit_all(train)
            forecasts = self._predict_all(fitted, self.test_size, test.index)
            forecasts = {
                "lightgbm": self._predict_lightgbm(lgbm, self.test_size, test.index),
                **forecasts,
            }
            _, total = self._evaluate_split(train, test, forecasts)
            rows.append(total.assign(fold=fold, train_end=train_end))
        return pd.concat(rows, ignore_index=True)

    def feature_importance(self, top_n: int | None = 20) -> pd.Series:
        """Gain-based LightGBM feature importance from the last ``fit``."""
        if self.lgbm_ is None or self.lgbm_.feature_importance_ is None:
            raise RuntimeError("call fit() first")
        importance = self.lgbm_.feature_importance_
        return importance if top_n is None else importance.head(top_n)
