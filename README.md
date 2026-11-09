# Прогнозирование продаж классическими методами

Практическое задание №1 по курсу «Прикладной анализ временных рядов» (ИТМО).
Прогноз дневных продаж 15 товаров магазина `STORE_2` на неделю, месяц и квартал
с помощью Seasonal Naive, Holt–Winters, SARIMA и регрессии на тренд + Фурье + экзогенные признаки.

## Структура

```
data/                      исходные CSV (продажи, календарь, цены)
src/sales_forecast/
  data.py                  загрузка, валидация, предобработка -> StoreDataset
  models.py                модели с единым интерфейсом fit / predict
  metrics.py               MAE, RMSE, WAPE, MASE
  plotting.py              графики
  forecaster.py            класс SalesForecaster — весь пайплайн
tests/                     pytest (синтетические данные)
notebooks/report.ipynb     отчёт: вызовы методов класса, графики, ответы на вопросы
```

## Запуск

```bash
pip install -r requirements.txt
pytest                     # тесты
flake8 src tests           # PEP-8
jupyter nbconvert --to notebook --execute --inplace notebooks/report.ipynb   # ~10 мин
```

Минимальный пример:

```python
from sales_forecast import SalesForecaster

forecaster = SalesForecaster("STORE_2", data_dir="data")
forecaster.load_data()
forecaster.fit()                      # обучение всех моделей на всех товарах
forecaster.summary("wape")            # модель x горизонт (week / month / quarter)
forecaster.predict("month")           # {модель: DataFrame(даты x товары)}
```
