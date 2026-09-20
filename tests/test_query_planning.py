import pytest

from query_planning import forecast_history_sql
from forecasting import detect_forecast_intent, extract_target_window


SCHEMA = {'orders': {'order_date': 'DATE', 'order_id': 'TEXT', 'total_sales': 'REAL'}}


def test_forecast_target_is_not_a_historical_filter():
    question = 'What will be the forecasted order in each month in 2018?'
    assert detect_forecast_intent(question)
    assert extract_target_window(question)[2] == '2018'
    sql = forecast_history_sql(question, SCHEMA)
    assert 'COUNT(DISTINCT order_id)' in sql
    assert '2018' not in sql and 'WHERE' not in sql and 'LIMIT' not in sql


@pytest.mark.parametrize('question', [
    'Forecast orders for West in 2018',
    'Forecast orders using history since 2015',
    'Forecast sales by region next year',
    'Forecast weekly orders in 2018',
    'Forecast profit next year',
])
def test_unsupported_scopes_are_not_silently_dropped(question):
    assert forecast_history_sql(question, SCHEMA) is None


def test_revenue_and_schema_permissions():
    assert 'SUM(total_sales)' in forecast_history_sql('Forecast revenue next 12 months', SCHEMA)
    assert forecast_history_sql('Forecast orders next year', {'orders': {'order_date': 'DATE'}}) is None


def test_statistical_fallback_is_limited_to_verified_history_plans():
    from types import SimpleNamespace
    from test_database_gateway import _pipeline_functions
    nodes = _pipeline_functions({'data_only_summary'}, {'AccuracyMetrics': type('Metrics', (), {})})
    state = {'forecast_result': SimpleNamespace(meta={'warnings': ['Limited history.']}),
             'sql_plan_verified': True, 'rows': [{'period': '2017-12'}]}
    summary = nodes['data_only_summary'](state)
    assert 'statistical forecast' in summary and 'Limited history.' in summary
    assert 'not actual' in summary
    state['sql_plan_verified'] = False
    assert 'statistical forecast' not in nodes['data_only_summary'](state)
