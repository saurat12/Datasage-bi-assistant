import numpy as np
import pandas as pd
import pytest

import forecasting
from query_planning import forecast_history_sql


@pytest.fixture(autouse=True)
def fast_model(monkeypatch):
    # Test calendar/routing behavior independently from statistical fitting.
    def fit(values, periods, seasonal_period):
        values = np.full(periods, 10.0)
        return values, values - 1, values + 1, 'test model'
    monkeypatch.setattr(forecasting, '_fit_and_forecast', fit)


def history(freq='MS', periods=48):
    return pd.DataFrame({'period': pd.date_range('2020-01-01', periods=periods, freq=freq),
                         'revenue': np.arange(periods) + 100.0})


@pytest.mark.parametrize('amount,unit,expected', [(2, 'years', 24), (3, 'quarters', 9),
                                                  (6, 'months', 6), (2, 'periods', 2)])
def test_monthly_calendar_horizons(amount, unit, expected):
    result = forecasting.forecast_series(history(), periods=amount, horizon_unit=unit)
    assert len(result.forecast) == expected
    assert result.meta['periods'] == expected
    assert result.forecast.date.iloc[0] == pd.Timestamp('2024-01-01')


def test_two_year_monthly_endpoint():
    result = forecasting.forecast_series(history(), periods=2, horizon_unit='years')
    assert result.forecast.date.iloc[-1] == pd.Timestamp('2025-12-01')


def test_next_year_without_number_preserves_calendar_meaning():
    amount, unit = forecasting.extract_horizon('Forecast annual revenue next year')
    assert (amount, unit) == (1, 'year')
    assert len(forecasting.forecast_series(history(), periods=amount, horizon_unit=unit).forecast) == 12
    assert len(forecasting.forecast_series(history('YS', 4), periods=amount, horizon_unit=unit).forecast) == 1


def test_annual_grain_and_horizon_agree():
    schema = {'orders': {'order_date': 'DATE', 'total_sales': 'REAL'}}
    sql = forecast_history_sql('Forecast annual revenue for the next 2 years', schema)
    assert "strftime('%Y-01-01'" in sql and '%Y-%m-01' not in sql
    result = forecasting.forecast_series(history('YS', 4), periods=2, horizon_unit='years')
    assert list(result.forecast.date) == [pd.Timestamp('2024-01-01'), pd.Timestamp('2025-01-01')]


def test_monthly_grain_with_year_horizon_is_not_annual():
    schema = {'orders': {'order_date': 'DATE', 'order_id': 'TEXT'}}
    assert '%Y-%m-01' in forecast_history_sql('Forecast orders for the next 2 years', schema)
    assert '%Y-01-01' in forecast_history_sql('Forecast orders for each year', schema)


def test_absolute_year_still_clips_correctly():
    target = forecasting.extract_target_window('Forecast annual revenue in 2024')
    result = forecasting.forecast_series(history('YS', 4), target_window=target)
    assert list(result.forecast.date) == [pd.Timestamp('2024-01-01')]


def test_grouped_horizons_convert_for_every_group():
    df = pd.concat([history().assign(region='East'), history().assign(region='West')])
    result = forecasting.forecast_series(df, periods=2, horizon_unit='years', group_col='region')
    assert result.forecast.groupby('group').size().to_dict() == {'East': 24, 'West': 24}
    assert result.meta['periods_by_group'] == {'East': 24, 'West': 24}


def test_daily_calendar_year_includes_leap_day():
    assert forecasting._relative_periods(pd.Timestamp('2023-12-31'), 'D', 1, 'year') == 366


def test_native_period_callers_are_unchanged():
    assert len(forecasting.forecast_series(history(), periods=3).forecast) == 3


@pytest.mark.parametrize('amount,unit', [(0, 'years'), (-1, 'months'), (1.5, 'months'), (1, 'invalid')])
def test_invalid_relative_horizons_fail_clearly(amount, unit):
    with pytest.raises(ValueError):
        forecasting.forecast_series(history(), periods=amount, horizon_unit=unit)


def test_finer_horizon_than_history_requires_finer_data():
    with pytest.raises(ValueError, match='finer-grained'):
        forecasting.forecast_series(history('YS', 4), periods=2, horizon_unit='months')
