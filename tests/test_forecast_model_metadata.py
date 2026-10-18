import numpy as np
import pandas as pd

import forecasting


def test_flat_series_can_select_no_trend_ets(monkeypatch):
    from statsmodels.tsa import holtwinters

    class FakeETS:
        def __init__(self, values, **options):
            self.values = values
            self.options = options
            self.fittedvalues = values

        def fit(self, optimized):
            self.aic = 0 if self.options['trend'] is None else 10
            return self

        def forecast(self, periods):
            return np.full(periods, self.values[-1])

    monkeypatch.setattr(holtwinters, 'ExponentialSmoothing', FakeETS)
    _, _, _, method = forecasting._fit_and_forecast(
        pd.Series([10.0] * 8), periods=2, seasonal_period=1,
    )
    assert 'no trend' in method


def test_nonseasonal_arima_fallback_is_disclosed(monkeypatch):
    monkeypatch.setattr(forecasting, '_fit_and_forecast',
                        lambda series, periods, seasonal_period: (
                            np.full(periods, 10.0), np.full(periods, 8.0),
                            np.full(periods, 12.0), 'ARIMA(1,1,1)'))
    rows = pd.DataFrame({
        'period': pd.date_range('2015-01-01', periods=24, freq='MS'),
        'value': [10.0] * 24,
    })
    result = forecasting.forecast_series(rows, periods=2)
    assert result.meta['seasonal_period'] > 1
    assert 'nonseasonal ARIMA' in result.meta['warnings'][0]
