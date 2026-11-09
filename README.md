# Прогнозирование продаж (классика + LightGBM)

Практические задания №1 и №2 по курсу "Прикладной анализ временных рядов" (ИТМО).
Прогноз дневных продаж 15 товаров магазина `STORE_2` на неделю, месяц и квартал.

- Задание №1: Seasonal Naive, Holt-Winters, SARIMA, регрессия (тренд + Фурье + экзогенные).
- Задание №2: глобальный LightGBM с лагами и календарными признаками.

## Структура

```
data/                         исходные CSV (продажи, календарь, цены)
src/sales_forecast/
  data.py                     загрузка, валидация, предобработка -> StoreDataset
  models.py                   классические модели (fit / predict)
  features.py                 признаки для бустинга
  boosting.py                 LightGBMGlobalModel
  metrics.py                  MAE, RMSE, WAPE, MASE
  plotting.py                 графики
  forecaster.py               SalesForecaster (лаба 1)
  modern_forecaster.py        ModernSalesForecaster (лаба 2)
tests/                        pytest
notebooks/report.ipynb        отчет по заданию №1
notebooks/report_modern.ipynb отчет по заданию №2
```

## Запуск

```bash
pip install -r requirements.txt
pytest
flake8 src tests
jupyter nbconvert --to notebook --execute --inplace notebooks/report.ipynb          # ~10 мин
jupyter nbconvert --to notebook --execute --inplace notebooks/report_modern.ipynb   # ~3 мин
```

Пример (лаба 2):

```python
from sales_forecast import ModernSalesForecaster

forecaster = ModernSalesForecaster("STORE_2", data_dir="data")
forecaster.load_data()
forecaster.fit()
forecaster.summary("wape")
forecaster.feature_importance(10)
```
