import numpy as np
import pandas as pd

import accuracy
import forecasting


def test_identical_fits_reuse_result_without_sharing_mutable_arrays(monkeypatch):
    calls = []

    def fit(series, periods, seasonal_period):
        calls.append((series.tolist(), periods, seasonal_period))
        values = np.full(periods, float(series.iloc[-1]))
        return values, values - 1, values + 1, 'test model'

    forecasting._cached_fit.cache_clear()
    monkeypatch.setattr(forecasting, '_fit_and_forecast_uncached', fit)
    history = pd.Series([10.0, 20.0, 30.0, 40.0])
    first = forecasting._fit_and_forecast(history, 3, 1)
    first[0][0] = -999.0
    second = forecasting._fit_and_forecast(history, 3, 1)
    assert len(calls) == 1
    assert second[0].tolist() == [40.0, 40.0, 40.0]
    forecasting._fit_and_forecast(history, 4, 1)
    forecasting._fit_and_forecast(pd.Series([10.0, 20.0, 30.0, 41.0]), 3, 1)
    assert len(calls) == 3
    forecasting._cached_fit.cache_clear()


def test_repeated_backtest_reuses_identical_training_windows(monkeypatch):
    calls = []

    def fit(series, periods, seasonal_period):
        calls.append(len(series))
        point = np.full(periods, float(series.iloc[-1]))
        return point, point - 1, point + 1, 'test model'

    forecasting._cached_fit.cache_clear()
    monkeypatch.setattr(forecasting, '_fit_and_forecast_uncached', fit)
    history = pd.DataFrame({
        'date': pd.date_range('2014-01-01', periods=36, freq='MS'),
        'value': np.arange(36, dtype=float) + 100,
    })
    first = accuracy.quick_accuracy(history, horizon=3)
    count = len(calls)
    second = accuracy.quick_accuracy(history.copy(), horizon=3)
    assert first == second
    assert count > 0 and len(calls) == count
    forecasting._cached_fit.cache_clear()
