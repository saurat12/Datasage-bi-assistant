import pytest
import json
import re
import sqlite3

from query_planning import (
    forecast_history_sql, historical_monthly_order_count_sql,
    requested_metric, metric_sql_issue,
)
from forecasting import detect_forecast_intent, extract_target_window


SCHEMA = {'orders': {'order_date': 'DATE', 'order_id': 'TEXT', 'total_sales': 'REAL'}}


@pytest.mark.parametrize(('question', 'metric'), [
    ('What is the monthly order total in 2014?', 'order_count'),
    ('How many orders were placed in East in 2017?', 'order_count'),
    ('Show total sales by region in 2017', 'revenue'),
    ('What is the quantity sold each month?', 'quantity'),
    ('Compare revenue and order counts by region', None),
    ('Show all orders in 2017', None),
    ('What is the average order value?', None),
    ('What is the total order value by region?', None),
])
def test_metric_intent_only_when_single_aggregate_is_clear(question, metric):
    assert requested_metric(question) == metric


def test_metric_contract_catches_clear_substitutions_but_allows_matching_sql():
    question = 'What is the number of orders by region in 2017?'
    metric = requested_metric(question)
    assert metric_sql_issue(metric, 'SELECT region, SUM(total_sales) FROM orders GROUP BY region')
    assert metric_sql_issue(metric, 'SELECT region, COUNT(DISTINCT o.order_id) FROM orders o GROUP BY region') is None
    assert metric_sql_issue('revenue', 'SELECT COUNT(DISTINCT order_id) FROM orders')
    assert metric_sql_issue('quantity', 'SELECT SUM(total_sales) FROM orders')
    assert metric_sql_issue(None, 'SELECT SUM(total_sales) FROM orders') is None


def test_metric_mismatch_blocks_wrong_sql_before_model_review():
    from test_database_gateway import _pipeline_functions
    from evaluation import SQL_CHECKS, failed_verdict
    from sql_safety import split_statements

    calls = []
    nodes = _pipeline_functions({'critic_node'}, {
        'requested_metric': requested_metric, 'metric_sql_issue': metric_sql_issue,
        'SQL_CHECKS': SQL_CHECKS, 'failed_verdict': failed_verdict,
        '_split_statements': split_statements, 're': re, 'json': json,
        'APPROVED_SCHEMA': json.dumps(SCHEMA),
        'invoke_evaluator': lambda *args, **kwargs: calls.append('critic'),
    })
    result = nodes['critic_node']({
        'question': 'How many orders were placed in East in 2017?',
        'forecast': False,
        'sql': "SELECT SUM(total_sales) FROM orders WHERE region = 'East'",
    })
    assert result['critic_approved'] is False
    assert result['retry_count'] == 1
    assert 'order count' in result['critic_reason']
    assert not calls


@pytest.mark.parametrize('question', [
    'What is the monthly order total in 2014?',
    'What is the total number of orders each month in 2016?',
    'How many orders per month in 2014?',
])
def test_historical_monthly_order_counts_use_distinct_ids(question):
    sql = historical_monthly_order_count_sql(question, SCHEMA)
    assert 'COUNT(DISTINCT order_id)' in sql
    assert 'total_sales' not in sql


def test_historical_monthly_order_count_query_returns_months_not_annual_total():
    sql = historical_monthly_order_count_sql('What is the monthly order total in 2014?', SCHEMA)
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE orders (order_id TEXT, order_date TEXT, total_sales REAL)')
        db.executemany('INSERT INTO orders VALUES (?, ?, ?)', [
            ('A', '2014-01-03', 100), ('A', '2014-01-03', 200),
            ('B', '2014-01-20', 300), ('C', '2014-02-01', 400),
            ('D', '2015-01-01', 500),
        ])
        cursor = db.execute(sql)
        assert [column[0] for column in cursor.description] == ['period', 'monthly_orders']
        assert cursor.fetchall() == [('2014-01', 2), ('2014-02', 1)]


@pytest.mark.parametrize('question', [
    'What is the monthly order total in East region in 2014?',
    'What is the monthly sales total in 2014?',
    'What is the monthly order total in 2014 excluding returns?',
    'Forecast monthly orders in 2014',
    'What is the monthly order total in 2014 and 2015?',
])
def test_historical_template_does_not_drop_extra_scope(question):
    assert historical_monthly_order_count_sql(question, SCHEMA) is None


def test_historical_template_requires_approved_order_columns():
    assert historical_monthly_order_count_sql(
        'What is the monthly order total in 2014?',
        {'orders': {'order_date': 'DATE'}},
    ) is None


def test_clear_monthly_order_question_bypasses_mistaken_sql_reviewer():
    from test_database_gateway import _pipeline_functions
    from evaluation import SQL_CHECKS
    from sql_safety import split_statements

    calls = []
    nodes = _pipeline_functions({'sql_agent_node', 'critic_node'}, {
        'historical_monthly_order_count_sql': historical_monthly_order_count_sql,
        'requested_metric': requested_metric, 'metric_sql_issue': metric_sql_issue,
        'APPROVED_SCHEMA': json.dumps(SCHEMA), 'json': json,
        'SQL_CHECKS': SQL_CHECKS, '_split_statements': split_statements, 're': re,
        'sql_llm': type('Model', (), {'invoke': lambda self, messages: calls.append(messages)})(),
        'invoke_evaluator': lambda *args, **kwargs: calls.append('critic'),
    })
    result = nodes['sql_agent_node']({
        'question': 'What is the monthly order total in 2014?', 'forecast': False,
    })
    assert result['sql_plan_verified'] is True
    assert 'COUNT(DISTINCT order_id)' in result['sql']
    reviewed = nodes['critic_node'](result)
    assert reviewed['critic_approved'] is True
    assert reviewed['sql_evaluation']['source'] == 'deterministic_monthly_order_count_plan'
    assert not calls


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
