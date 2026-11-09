"""Classical forecasting models behind a common ``fit`` / ``predict`` API.

Every model works on a single univariate ``pd.Series`` with a regular
``DatetimeIndex``. Models that support exogenous regressors declare
``uses_exog = True`` and expect an aligned ``pd.DataFrame`` both at fit
and at predict time.
"""

from __future__ import annotations

import itertools
import warnings
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import kpss

ModelFactory = Callable[[], "ForecastModel"]


def future_index(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """``horizon`` consecutive timestamps following ``index``."""
    freq = index.freq or pd.infer_freq(index)
    if freq is None:
        raise ValueError("cannot infer frequency of the index")
    start = index[-1] + pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(start, periods=horizon, freq=freq, name=index.name)


def _validate_series(y: pd.Series) -> pd.Series:
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series")
    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError("y must have a DatetimeIndex")
    if y.isna().any():
        raise ValueError("y must not contain NaN")
    if len(y) < 2:
        raise ValueError("y must contain at least two observations")
    y = y.astype(float)
    if y.index.freq is None:
        freq = pd.infer_freq(y.index)
        if freq is None:
            raise ValueError("y index must have a regular frequency")
        y = y.copy()
        y.index.freq = freq
    return y


def _validate_exog(
    exog: pd.DataFrame | None, index: pd.DatetimeIndex, required: bool
) -> pd.DataFrame | None:
    if exog is None:
        if required:
            raise ValueError("this model requires exogenous features")
        return None
    if not exog.index.equals(index):
        raise ValueError("exog index must match the target index exactly")
    if exog.isna().any().any():
        raise ValueError("exog must not contain NaN")
    return exog.astype(float)


class ForecastModel(ABC):
    """Base class: validation, constant-series guard, output formatting."""

    name: str = "base"
    uses_exog: bool = False

    def __init__(self) -> None:
        self._y_train: pd.Series | None = None
        self._constant: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self._y_train is not None

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "ForecastModel":
        y = _validate_series(y)
        exog = _validate_exog(exog, y.index, required=self.uses_exog)
        self._y_train = y
        # Degenerate series (all equal) break most estimators; predict the
        # constant instead of failing.
        if y.nunique() <= 1:
            self._constant = float(y.iloc[0])
        else:
            self._constant = None
            self._fit(y, exog if self.uses_exog else None)
        return self

    def predict(
        self, horizon: int, exog: pd.DataFrame | None = None
    ) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("model is not fitted")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        index = future_index(self._y_train.index, horizon)
        exog = _validate_exog(exog, index, required=self.uses_exog)
        if self._constant is not None:
            values = np.full(horizon, self._constant)
        else:
            values = np.asarray(
                self._predict(horizon, exog if self.uses_exog else None),
                dtype=float,
            )
        return pd.Series(values, index=index, name=self.name)

    @abstractmethod
    def _fit(self, y: pd.Series, exog: pd.DataFrame | None) -> None:
        """Estimate parameters on a non-constant series."""

    @abstractmethod
    def _predict(self, horizon: int, exog: pd.DataFrame | None) -> np.ndarray:
        """Return ``horizon`` point forecasts."""


class SeasonalNaiveModel(ForecastModel):
    """Repeat the last observed season. Baseline for everything else."""

    name = "seasonal_naive"

    def __init__(self, season_length: int = 7) -> None:
        super().__init__()
        if season_length < 1:
            raise ValueError("season_length must be >= 1")
        self.season_length = season_length

    def _fit(self, y: pd.Series, exog: pd.DataFrame | None) -> None:
        if len(y) < self.season_length:
            raise ValueError("series shorter than one season")
        self._last_season = y.to_numpy()[-self.season_length:]

    def _predict(self, horizon: int, exog: pd.DataFrame | None) -> np.ndarray:
        repeats = int(np.ceil(horizon / self.season_length))
        return np.tile(self._last_season, repeats)[:horizon]


class HoltWintersModel(ForecastModel):
    """Holt-Winters exponential smoothing (additive, damped trend)."""

    name = "holt_winters"

    def __init__(
        self,
        seasonal_periods: int = 7,
        trend: str | None = "add",
        damped_trend: bool = True,
        seasonal: str | None = "add",
    ) -> None:
        super().__init__()
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.damped_trend = damped_trend and trend is not None
        self.seasonal = seasonal

    def _fit(self, y: pd.Series, exog: pd.DataFrame | None) -> None:
        if self.seasonal and len(y) < 2 * self.seasonal_periods:
            raise ValueError("need at least two seasons to fit Holt-Winters")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                y,
                trend=self.trend,
                damped_trend=self.damped_trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods if self.seasonal else None,
                initialization_method="estimated",
            )
            self._result = model.fit(optimized=True)

    def _predict(self, horizon: int, exog: pd.DataFrame | None) -> np.ndarray:
        return self._result.forecast(horizon).to_numpy()


class SarimaModel(ForecastModel):
    """SARIMA(p, d, q)(P, D, Q)_s with stepwise AIC order selection.

    Selection is two-stage to keep the number of fits small: first the
    seasonal (P, Q) part is chosen with (p, q) = (1, 1), then (p, q) is
    chosen with the best seasonal part fixed. ``d`` is chosen by a KPSS
    test on the seasonally differenced series; ``D`` is fixed.
    If ``order`` and ``seasonal_order`` are given, no search is done.
    """

    name = "sarima"

    def __init__(
        self,
        season_length: int = 7,
        order: tuple[int, int, int] | None = None,
        seasonal_order: tuple[int, int, int] | None = None,
        max_p: int = 2,
        max_q: int = 2,
        max_seasonal_p: int = 1,
        max_seasonal_q: int = 1,
        seasonal_diff: int = 1,
        train_window: int | None = None,
        maxiter: int = 50,
    ) -> None:
        super().__init__()
        self.season_length = season_length
        self.order = order
        self.seasonal_order = seasonal_order
        self.max_p = max_p
        self.max_q = max_q
        self.max_seasonal_p = max_seasonal_p
        self.max_seasonal_q = max_seasonal_q
        self.seasonal_diff = seasonal_diff
        self.train_window = train_window
        self.maxiter = maxiter
        self.selected_order_: tuple | None = None
        self.selected_seasonal_order_: tuple | None = None
        self.aic_table_: dict[tuple, float] = {}

    def _fit(self, y: pd.Series, exog: pd.DataFrame | None) -> None:
        if self.train_window is not None:
            y = y.iloc[-self.train_window:]
        if len(y) < 3 * self.season_length:
            raise ValueError("need at least three seasons to fit SARIMA")
        if self.order is not None and self.seasonal_order is not None:
            order, seasonal = self.order, self.seasonal_order
        else:
            order, seasonal = self._select_order(y)
        self.selected_order_ = order
        self.selected_seasonal_order_ = seasonal
        self._result = self._fit_one(y, order, seasonal)

    def _predict(self, horizon: int, exog: pd.DataFrame | None) -> np.ndarray:
        return self._result.get_forecast(horizon).predicted_mean.to_numpy()

    def _fit_one(self, y: pd.Series, order: tuple, seasonal: tuple):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y,
                order=order,
                seasonal_order=(*seasonal, self.season_length),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            return model.fit(disp=False, maxiter=self.maxiter)

    def _aic(self, y: pd.Series, order: tuple, seasonal: tuple) -> float:
        key = (order, seasonal)
        if key not in self.aic_table_:
            try:
                aic = float(self._fit_one(y, order, seasonal).aic)
            except (np.linalg.LinAlgError, ValueError):
                aic = float("inf")
            self.aic_table_[key] = aic if np.isfinite(aic) else float("inf")
        return self.aic_table_[key]

    def _select_order(self, y: pd.Series) -> tuple[tuple, tuple]:
        d = self._choose_d(y)
        big_d = self.seasonal_diff
        self.aic_table_ = {}

        seasonal_grid = itertools.product(
            range(self.max_seasonal_p + 1), range(self.max_seasonal_q + 1)
        )
        best_seasonal = min(
            ((big_p, big_d, big_q) for big_p, big_q in seasonal_grid),
            key=lambda s: self._aic(y, (1, d, 1), s),
        )
        nonseasonal_grid = itertools.product(
            range(self.max_p + 1), range(self.max_q + 1)
        )
        best_order = min(
            ((p, d, q) for p, q in nonseasonal_grid),
            key=lambda o: self._aic(y, o, best_seasonal),
        )
        if not np.isfinite(self._aic(y, best_order, best_seasonal)):
            raise RuntimeError("all SARIMA candidates failed to fit")
        return best_order, best_seasonal

    def _choose_d(self, y: pd.Series) -> int:
        """0 if the seasonally differenced series is KPSS-stationary else 1."""
        values = y.to_numpy()
        if self.seasonal_diff > 0:
            lag = self.season_length * self.seasonal_diff
            values = values[lag:] - values[:-lag]
        if np.ptp(values) == 0:
            return 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, p_value, *_ = kpss(values, regression="c", nlags="auto")
        return 0 if p_value >= 0.05 else 1


def make_deterministic_features(
    index: pd.DatetimeIndex,
    origin: pd.Timestamp,
    fourier_periods: tuple[tuple[float, int], ...],
) -> pd.DataFrame:
    """Linear trend plus Fourier harmonics for each (period, n_harmonics).

    Time is measured in days from ``origin`` so that features computed at
    fit and at predict time lie on the same axis.
    """
    t = (index - origin) / pd.Timedelta(days=1)
    t = np.asarray(t, dtype=float)
    features = {"trend": t / 365.25}
    for period, n_harmonics in fourier_periods:
        for k in range(1, n_harmonics + 1):
            angle = 2 * np.pi * k * t / period
            features[f"sin_{period:g}_{k}"] = np.sin(angle)
            features[f"cos_{period:g}_{k}"] = np.cos(angle)
    return pd.DataFrame(features, index=index)


class RegressionModel(ForecastModel):
    """Ridge regression on trend + Fourier seasonality + exogenous features.

    The deterministic part follows the "trend + seasonal dummies" approach
    from the seminar, with Fourier terms instead of dummies. By default
    only the weekly cycle is modelled (3 harmonics span the 6 degrees of
    freedom of day-of-week dummies); a yearly cycle can be added through
    ``fourier_periods``, e.g. ``((7, 3), (365.25, 3))``. Exogenous
    regressors (price, holidays, cashback) are optional.
    """

    name = "regression"
    uses_exog = True

    def __init__(
        self,
        fourier_periods: tuple[tuple[float, int], ...] = ((7, 3),),
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.fourier_periods = fourier_periods
        self.alpha = alpha
        self._pipeline: Pipeline | None = None

    def _design(self, index: pd.DatetimeIndex, exog: pd.DataFrame) -> pd.DataFrame:
        deterministic = make_deterministic_features(
            index, self._origin, self.fourier_periods
        )
        return pd.concat([deterministic, exog], axis=1)

    def _fit(self, y: pd.Series, exog: pd.DataFrame | None) -> None:
        self._origin = y.index[0]
        design = self._design(y.index, exog)
        self._columns = list(design.columns)
        self._pipeline = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self._pipeline.fit(design.to_numpy(), y.to_numpy())

    def _predict(self, horizon: int, exog: pd.DataFrame | None) -> np.ndarray:
        index = future_index(self._y_train.index, horizon)
        design = self._design(index, exog)
        missing = set(self._columns) - set(design.columns)
        if missing:
            raise ValueError(f"exog is missing columns {sorted(missing)}")
        return self._pipeline.predict(design[self._columns].to_numpy())


class Log1pModel(ForecastModel):
    """Fit the wrapped model on ``log1p(y)`` and back-transform forecasts.

    Additive models on the log scale behave multiplicatively on the
    original scale, which suits sales series whose variance grows with
    the level.
    """

    def __init__(self, inner: ForecastModel) -> None:
        super().__init__()
        self.inner = inner
        self.name = inner.name
        self.uses_exog = inner.uses_exog

    def _fit(self, y: pd.Series, exog: pd.DataFrame | None) -> None:
        if (y < 0).any():
            raise ValueError("log1p transform requires non-negative data")
        self.inner.fit(np.log1p(y), exog)

    def _predict(self, horizon: int, exog: pd.DataFrame | None) -> np.ndarray:
        return np.expm1(self.inner.predict(horizon, exog).to_numpy())


DEFAULT_MODELS: dict[str, ModelFactory] = {
    SeasonalNaiveModel.name: SeasonalNaiveModel,
    HoltWintersModel.name: HoltWintersModel,
    SarimaModel.name: SarimaModel,
    RegressionModel.name: RegressionModel,
}
