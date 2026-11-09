"""End-to-end pipeline: preprocessing, training, forecasting, evaluation."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from statsmodels.tsa.seasonal import STL, DecomposeResult

from . import plotting
from .data import (
    StoreDataset,
    build_store_dataset,
    impute_closed_days,
    load_raw,
    trim_leading_zeros,
)
from .metrics import METRIC_NAMES, aggregate_metrics, compute_metrics
from .models import DEFAULT_MODELS, ForecastModel, Log1pModel, ModelFactory

logger = logging.getLogger(__name__)

HORIZONS: dict[str, int] = {"week": 7, "month": 30, "quarter": 90}

FittedModels = dict[str, dict[str, ForecastModel]]
Forecasts = dict[str, pd.DataFrame]


class SalesForecaster:
    """Forecast daily item sales of one store with classical models.

    Workflow::

        forecaster = SalesForecaster("STORE_2", data_dir="data")
        forecaster.load_data()
        forecaster.fit()
        results = forecaster.evaluate()        # per item
        totals = forecaster.evaluate_total()   # pooled over items

    Parameters
    ----------
    store_id:
        Store to model.
    data_dir:
        Directory with the three CSV files.
    models:
        Either a mapping ``{name: factory}`` or an iterable of names from
        :data:`sales_forecast.models.DEFAULT_MODELS`. ``None`` uses all
        default models.
    horizons:
        Mapping ``{label: n_days}``; defaults to week / month / quarter.
    test_size:
        Length of the hold-out period in days. Defaults to the longest
        horizon.
    season_length:
        Seasonal period in days used by models and by MASE (7 = weekly).
    log_transform:
        Fit all models on ``log1p`` of the target.
    trim_leading_zeros:
        Drop the all-zero period before an item was first sold.
    impute_closed_days:
        Interpolate sales on known closed days for training and force the
        forecast to zero on such days.
    train_window:
        If set, only the last ``train_window`` days of each training series
        are used for fitting. Useful when the distant past is not
        representative (assortment changes, long stock-outs).
    """

    def __init__(
        self,
        store_id: str,
        data_dir: str | Path = "data",
        models: Mapping[str, ModelFactory] | Iterable[str] | None = None,
        horizons: Mapping[str, int] | None = None,
        test_size: int | None = None,
        season_length: int = 7,
        log_transform: bool = False,
        trim_leading_zeros: bool = True,
        impute_closed_days: bool = True,
        train_window: int | None = None,
    ) -> None:
        self.store_id = store_id
        self.data_dir = Path(data_dir)
        self.horizons = dict(horizons) if horizons is not None else dict(HORIZONS)
        if not self.horizons or min(self.horizons.values()) < 1:
            raise ValueError("horizons must be a non-empty mapping of positive ints")
        self.test_size = test_size or max(self.horizons.values())
        if self.test_size < max(self.horizons.values()):
            raise ValueError("test_size must cover the longest horizon")
        self.season_length = season_length
        self.log_transform = log_transform
        self.trim_leading_zeros = trim_leading_zeros
        self.impute_closed_days = impute_closed_days
        if train_window is not None and train_window < 2 * season_length:
            raise ValueError("train_window must cover at least two seasons")
        self.train_window = train_window
        self.model_factories = self._resolve_models(models)

        self.dataset: StoreDataset | None = None
        self.train_: pd.DataFrame | None = None
        self.test_: pd.DataFrame | None = None
        self.fitted_models_: FittedModels | None = None
        self.forecasts_: Forecasts | None = None

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _resolve_models(
        models: Mapping[str, ModelFactory] | Iterable[str] | None,
    ) -> dict[str, ModelFactory]:
        if models is None:
            return dict(DEFAULT_MODELS)
        if isinstance(models, Mapping):
            return dict(models)
        resolved = {}
        for name in models:
            if name not in DEFAULT_MODELS:
                raise ValueError(
                    f"unknown model {name!r}; known: {sorted(DEFAULT_MODELS)}"
                )
            resolved[name] = DEFAULT_MODELS[name]
        if not resolved:
            raise ValueError("at least one model is required")
        return resolved

    def _make_model(self, name: str) -> ForecastModel:
        model = self.model_factories[name]()
        return Log1pModel(model) if self.log_transform else model

    @property
    def model_names(self) -> list[str]:
        return list(self.model_factories)

    # ------------------------------------------------------------------- data
    def load_data(self) -> StoreDataset:
        """Read the CSV files and build the preprocessed store dataset."""
        raw = load_raw(self.data_dir)
        self.dataset = build_store_dataset(raw, self.store_id)
        logger.info(
            "loaded %s: %d items, %d days (%s .. %s)",
            self.store_id,
            len(self.dataset.items),
            len(self.dataset.sales),
            self.dataset.sales.index[0].date(),
            self.dataset.sales.index[-1].date(),
        )
        return self.dataset

    def _require_dataset(self) -> StoreDataset:
        if self.dataset is None:
            raise RuntimeError("call load_data() first")
        return self.dataset

    @property
    def items(self) -> list[str]:
        return self._require_dataset().items

    def describe_items(self) -> pd.DataFrame:
        """Per-item summary used to characterise the series before modelling."""
        dataset = self._require_dataset()
        sales = dataset.sales
        first_sale = sales.apply(lambda s: s[s > 0].index.min())
        summary = pd.DataFrame(
            {
                "mean": sales.mean(),
                "std": sales.std(),
                "zero_share": (sales == 0).mean(),
                "max": sales.max(),
                "first_sale": first_sale,
                "mean_price": dataset.prices.mean(),
            }
        )
        return summary.sort_values("mean", ascending=False)

    def decompose(self, item: str, period: int | None = None) -> DecomposeResult:
        """STL decomposition of one item (EDA helper)."""
        dataset = self._require_dataset()
        series = self._prepare_target(item, dataset.sales)
        return STL(series, period=period or self.season_length, robust=True).fit()

    def split(self, train_end: pd.Timestamp | None = None):
        """Return ``(train, test)`` sales frames.

        ``test`` is the ``test_size`` days following ``train_end``
        (default: the last ``test_size`` days of the dataset).
        """
        sales = self._require_dataset().sales
        if train_end is None:
            split_at = len(sales) - self.test_size
        else:
            split_at = int(sales.index.get_indexer([pd.Timestamp(train_end)])[0]) + 1
            if split_at == 0:
                raise ValueError("train_end is not in the index")
        if split_at < 2 * self.season_length:
            raise ValueError("training part is too short")
        if split_at + self.test_size > len(sales):
            raise ValueError("not enough data after train_end for the test period")
        return sales.iloc[:split_at], sales.iloc[split_at:split_at + self.test_size]

    # -------------------------------------------------------------- training
    def _prepare_target(self, item: str, sales: pd.DataFrame) -> pd.Series:
        """Preprocess one item's training series."""
        dataset = self._require_dataset()
        series = sales[item]
        if self.impute_closed_days:
            series = impute_closed_days(series, dataset.exog["is_closed"])
        if self.trim_leading_zeros:
            series = trim_leading_zeros(series)
        if self.train_window is not None:
            series = series.iloc[-self.train_window:]
        series = series.copy()
        series.index.freq = "D"
        return series

    def _exog_for(self, item: str, index: pd.DatetimeIndex) -> pd.DataFrame:
        exog = self._require_dataset().item_exog(item).reindex(index)
        if exog.isna().any().any():
            raise ValueError("exogenous features are not available for all dates")
        return exog

    def _fit_all(self, train: pd.DataFrame) -> FittedModels:
        fitted: FittedModels = {name: {} for name in self.model_names}
        for item in train.columns:
            y = self._prepare_target(item, train)
            exog = self._exog_for(item, y.index)
            for name in self.model_names:
                logger.info("fit %s on %s", name, item)
                model = self._make_model(name)
                fitted[name][item] = model.fit(y, exog if model.uses_exog else None)
        return fitted

    def _postprocess(self, forecast: pd.DataFrame) -> pd.DataFrame:
        forecast = forecast.clip(lower=0)
        if self.impute_closed_days:
            closed = self._require_dataset().exog["is_closed"].reindex(forecast.index)
            forecast.loc[closed.fillna(0).astype(bool)] = 0.0
        return forecast

    def _predict_all(
        self, fitted: FittedModels, horizon: int, index: pd.DatetimeIndex
    ) -> Forecasts:
        forecasts: Forecasts = {}
        for name, per_item in fitted.items():
            columns = {}
            for item, model in per_item.items():
                exog = self._exog_for(item, index) if model.uses_exog else None
                columns[item] = model.predict(horizon, exog)
            frame = pd.DataFrame(columns, index=index)
            forecasts[name] = self._postprocess(frame)
        return forecasts

    def fit(self, train_end: pd.Timestamp | None = None) -> "SalesForecaster":
        """Fit every model on every item using data up to ``train_end``."""
        self.train_, self.test_ = self.split(train_end)
        self.fitted_models_ = self._fit_all(self.train_)
        self.forecasts_ = self._predict_all(
            self.fitted_models_, self.test_size, self.test_.index
        )
        return self

    def _require_fitted(self) -> tuple[pd.DataFrame, pd.DataFrame, Forecasts]:
        if self.forecasts_ is None or self.train_ is None or self.test_ is None:
            raise RuntimeError("call fit() first")
        return self.train_, self.test_, self.forecasts_

    def _horizon_days(self, horizon: int | str) -> int:
        days = self.horizons[horizon] if isinstance(horizon, str) else int(horizon)
        if days < 1 or days > self.test_size:
            raise ValueError(f"horizon must be within 1..{self.test_size} days")
        return days

    def predict(self, horizon: int | str = "quarter") -> Forecasts:
        """Forecasts ``{model: DataFrame(dates x items)}`` for ``horizon``."""
        _, _, forecasts = self._require_fitted()
        days = self._horizon_days(horizon)
        return {name: frame.iloc[:days] for name, frame in forecasts.items()}

    # ------------------------------------------------------------ evaluation
    def _evaluate_split(
        self, train: pd.DataFrame, test: pd.DataFrame, forecasts: Forecasts
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        per_item_rows, total_rows = [], []
        for label, days in self.horizons.items():
            actual = test.iloc[:days]
            for name, frame in forecasts.items():
                predicted = frame.iloc[:days]
                for item in test.columns:
                    scores = compute_metrics(
                        actual[item], predicted[item], train[item], self.season_length
                    )
                    per_item_rows.append(
                        {"model": name, "item": item, "horizon": label, **scores}
                    )
                pooled = aggregate_metrics(actual, predicted, train, self.season_length)
                total_rows.append({"model": name, "horizon": label, **pooled})
        return pd.DataFrame(per_item_rows), pd.DataFrame(total_rows)

    def evaluate(self) -> pd.DataFrame:
        """Per-item metrics on the hold-out set, long format.

        Columns: ``model, item, horizon, mae, rmse, wape, mase``.
        """
        per_item, _ = self._evaluate_split(*self._require_fitted())
        return per_item

    def evaluate_total(self) -> pd.DataFrame:
        """Metrics pooled over all items, one row per (model, horizon)."""
        _, total = self._evaluate_split(*self._require_fitted())
        return total

    def summary(self, metric: str = "wape") -> pd.DataFrame:
        """Pivot of pooled ``metric``: rows = model, columns = horizon."""
        if metric not in METRIC_NAMES:
            raise ValueError(f"unknown metric {metric!r}; known: {METRIC_NAMES}")
        total = self.evaluate_total()
        return total.pivot(index="model", columns="horizon", values=metric)[
            list(self.horizons)
        ]

    def best_model(self, metric: str = "wape", horizon: str = "quarter") -> str:
        """Model name with the lowest pooled ``metric`` at ``horizon``."""
        return str(self.summary(metric)[horizon].idxmin())

    def backtest(self, n_folds: int = 3, step: int | None = None) -> pd.DataFrame:
        """Rolling-origin evaluation with non-overlapping test windows.

        Fold 0 is the last ``test_size`` days, fold 1 the ``step`` days
        before it, and so on. Returns pooled metrics per (fold, model,
        horizon). Does not change the state set by :meth:`fit`.
        """
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
            fitted = self._fit_all(train)
            forecasts = self._predict_all(fitted, self.test_size, test.index)
            _, total = self._evaluate_split(train, test, forecasts)
            rows.append(total.assign(fold=fold, train_end=train_end))
        return pd.concat(rows, ignore_index=True)

    def backtest_summary(
        self, backtest: pd.DataFrame, metric: str = "wape"
    ) -> pd.DataFrame:
        """Mean of ``metric`` over folds: rows = model, columns = horizon."""
        if metric not in METRIC_NAMES:
            raise ValueError(f"unknown metric {metric!r}; known: {METRIC_NAMES}")
        table = backtest.pivot_table(
            index="model", columns="horizon", values=metric, aggfunc="mean"
        )
        return table[list(self.horizons)]

    # ----------------------------------------------------------------- plots
    def plot_items(self, title: str | None = None) -> Figure:
        """Grid of all item series."""
        sales = self._require_dataset().sales
        return plotting.plot_series_grid(sales, title=title or self.store_id)

    def plot_forecast(
        self,
        item: str,
        horizon: int | str = "quarter",
        models: Iterable[str] | None = None,
        history_days: int = 60,
        ax: Axes | None = None,
    ) -> Axes:
        """History, actuals and model forecasts for one item."""
        train, test, _ = self._require_fitted()
        days = self._horizon_days(horizon)
        forecasts = self.predict(days)
        names = list(models) if models is not None else self.model_names
        return plotting.plot_forecast(
            history=train[item].iloc[-history_days:],
            actual=test[item].iloc[:days],
            forecasts={name: forecasts[name][item] for name in names},
            title=f"{item}, horizon {days} days",
            ax=ax,
        )

    def plot_total_metrics(self, metric: str = "wape", ax: Axes | None = None) -> Axes:
        """Bars of pooled ``metric`` per horizon and model."""
        return plotting.plot_metric_bars(self.evaluate_total(), metric, ax=ax)

    def plot_item_metrics(
        self, metric: str = "wape", horizon: str = "quarter", ax: Axes | None = None
    ) -> Axes:
        """Heatmap item x model of ``metric`` at ``horizon``."""
        table = self.evaluate()
        table = table.loc[table["horizon"] == horizon]
        ax = plotting.plot_metric_heatmap(table, metric, ax=ax)
        ax.set_title(f"{metric.upper()}, horizon = {horizon}")
        return ax

    def selected_orders(self) -> pd.DataFrame | None:
        """SARIMA orders chosen per item, if a SARIMA model was fitted."""
        if self.fitted_models_ is None:
            return None
        rows = []
        for name, per_item in self.fitted_models_.items():
            for item, model in per_item.items():
                inner = model.inner if isinstance(model, Log1pModel) else model
                if hasattr(inner, "selected_order_") and inner.selected_order_:
                    rows.append(
                        {
                            "model": name,
                            "item": item,
                            "order": inner.selected_order_,
                            "seasonal_order": inner.selected_seasonal_order_,
                        }
                    )
        return pd.DataFrame(rows) if rows else None

    def forecast_table(self, item: str, horizon: int | str = "week") -> pd.DataFrame:
        """Actual values and forecasts of all models for one item side by side."""
        _, test, _ = self._require_fitted()
        days = self._horizon_days(horizon)
        forecasts = self.predict(days)
        table = {"actual": test[item].iloc[:days]}
        table.update({name: frame[item] for name, frame in forecasts.items()})
        return pd.DataFrame(table)
