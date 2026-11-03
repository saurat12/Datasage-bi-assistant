import pytest
import json
import re
import sqlite3

from query_planning import (
    forecast_history_sql, historical_monthly_order_count_sql,
    metric_sql_issue,
)
from forecasting import detect_forecast_intent, extract_target_window


SCHEMA = {'orders': {'order_date': 'DATE', 'order_id': 'TEXT', 'total_sales': 'REAL'}}


@pytest.mark.parametrize(('column', 'sql', 'mismatch'), [
    ('order_id', 'SELECT region, SUM(total_sales) FROM orders GROUP BY region', True),
    ('order_id', 'SELECT region, COUNT(DISTINCT o.order_id) FROM orders o GROUP BY region', False),
    ('total_sales', 'SELECT COUNT(DISTINCT order_id) FROM orders', True),
    ('quantity', 'SELECT SUM(quantity) FROM orders', False),
    (None, 'SELECT SUM(total_sales) FROM orders', False),
])
def test_metric_contract_catches_clear_substitutions(column, sql, mismatch):
    assert bool(metric_sql_issue(column, sql)) is mismatch


def test_metric_contract_generalizes_to_unseen_billing_schema():
    assert metric_sql_issue('mrr', 'SELECT COUNT(DISTINCT signup_id) FROM subscriptions')
    assert metric_sql_issue('mrr', 'SELECT SUM(s.mrr) FROM subscriptions s') is None
    assert metric_sql_issue('signup_id', 'SELECT COUNT(DISTINCT signup_id) FROM subscriptions') is None
    assert metric_sql_issue('mrr', "SELECT COUNT(*) FROM subscriptions WHERE note='SUM(mrr)'")


def test_metric_contract_allows_detail_projection_but_not_substituted_column():
    assert metric_sql_issue(
        'total_sales',
        "SELECT order_id, order_date, total_sales FROM orders "
        "WHERE region = 'West' ORDER BY order_date DESC LIMIT 5",
    ) is None
    assert metric_sql_issue(
        'total_sales',
        'SELECT order_id, order_date FROM orders ORDER BY order_date DESC LIMIT 5',
    )
    assert metric_sql_issue(
        'total_sales',
        'SELECT COUNT(DISTINCT order_id) FROM orders',
    )


def test_revenue_driver_dimensions_follow_question_and_orders_schema():
    from query_planning import requested_revenue_driver_dimensions, revenue_driver_sql_issue

    schema = {'orders': {'order_date': 'TEXT', 'region': 'TEXT',
                         'product_name': 'TEXT', 'total_sales': 'REAL'}}
    single = "What's driving the revenue in June 2016?"
    regional = "What's driving revenue in each region in June 2015?"
    assert requested_revenue_driver_dimensions(single, schema) == ('product_name',)
    assert requested_revenue_driver_dimensions(regional, schema) == ('region', 'product_name')
    assert requested_revenue_driver_dimensions(regional, {'billing': schema['orders']}) == ()
    assert requested_revenue_driver_dimensions('Revenue by region', schema) == ()
    assert requested_revenue_driver_dimensions("What's driving revenue growth?", schema) == ()
    assert requested_revenue_driver_dimensions("What's driving revenue by category?", schema) == ()

    product_sql = ('SELECT product_name, SUM(total_sales) FROM orders '
                   'GROUP BY product_name')
    regional_sql = ('SELECT region, product_name, SUM(total_sales) FROM orders '
                    'GROUP BY region, product_name')
    assert revenue_driver_sql_issue(product_sql, ('product_name',)) is None
    assert revenue_driver_sql_issue(regional_sql, ('region', 'product_name')) is None
    assert revenue_driver_sql_issue('SELECT region, SUM(total_sales) FROM orders '
                                    'GROUP BY region', ('region', 'product_name'))
    assert revenue_driver_sql_issue('SELECT region, product_name, SUM(total_sales) '
                                    'FROM orders GROUP BY region', ('region', 'product_name'))


def test_sql_critic_accepts_detail_projection_and_rejects_missing_metric():
    from test_database_gateway import _pipeline_functions
    from evaluation import SQL_CHECKS, SQL_EVALUATOR_PROMPT, failed_verdict, sql_review_rules
    from sql_safety import split_statements

    def approved(*args, **kwargs):
        return dict(passed=True, checks={key: True for key in SQL_CHECKS},
                    issues=[], retry_target=None, fix_instructions='',
                    requested_metric_column='total_sales')

    nodes = _pipeline_functions({'critic_node'}, {
        'metric_sql_issue': metric_sql_issue, 'SQL_EVALUATOR_PROMPT': SQL_EVALUATOR_PROMPT,
        'sql_review_rules': sql_review_rules, 'SQL_CHECKS': SQL_CHECKS,
        'failed_verdict': failed_verdict, '_split_statements': split_statements,
        're': re, 'json': json, 'APPROVED_SCHEMA': json.dumps(SCHEMA),
        'invoke_evaluator': approved, 'critic_llm': object(),
    })
    valid = nodes['critic_node']({
        'question': 'Show order ID, date and sales for the latest five orders',
        'sql': 'SELECT order_id, order_date, total_sales FROM orders '
               'ORDER BY order_date DESC LIMIT 5', 'forecast': False,
    })
    assert valid['critic_approved'] is True
    invalid = nodes['critic_node']({
        'question': 'Show order ID, date and sales for the latest five orders',
        'sql': 'SELECT order_id, order_date FROM orders ORDER BY order_date DESC LIMIT 5',
        'forecast': False,
    })
    assert invalid['critic_approved'] is False
    assert 'total_sales' in invalid['critic_reason']


def test_forecast_sql_critic_uses_separate_model_without_changing_historical_review():
    from test_database_gateway import _pipeline_functions
    from evaluation import SQL_CHECKS, SQL_EVALUATOR_PROMPT, failed_verdict, sql_review_rules
    from sql_safety import split_statements

    historical_model, forecast_model = object(), object()
    seen = []

    def reviewed(model, *args, **kwargs):
        seen.append(model)
        return dict(passed=True, checks={key: True for key in SQL_CHECKS},
                    issues=[], retry_target=None, fix_instructions='',
                    requested_metric_column='total_sales')

    nodes = _pipeline_functions({'critic_node'}, {
        'metric_sql_issue': metric_sql_issue, 'SQL_EVALUATOR_PROMPT': SQL_EVALUATOR_PROMPT,
        'sql_review_rules': sql_review_rules, 'SQL_CHECKS': SQL_CHECKS,
        'failed_verdict': failed_verdict, '_split_statements': split_statements,
        '_looks_like_future_filter': lambda sql: False,
        're': re, 'json': json, 'APPROVED_SCHEMA': json.dumps(SCHEMA),
        'invoke_evaluator': reviewed, 'critic_llm': historical_model,
        'forecast_sql_critic_llm': forecast_model,
    })
    sql = 'SELECT SUM(total_sales) FROM orders'
    assert nodes['critic_node']({'question': 'Total sales', 'sql': sql,
                                 'forecast': False})['critic_approved']
    forecast_sql = ("SELECT strftime('%Y-%m-01', order_date) AS period, "
                    'SUM(total_sales) FROM orders GROUP BY period ORDER BY period')
    assert nodes['critic_node']({'question': 'Forecast sales', 'sql': forecast_sql,
                                 'forecast': True})['critic_approved']
    assert seen == [historical_model, forecast_model]


def test_metric_contract_checks_only_simple_passthrough_ctes():
    assert metric_sql_issue(
        'total_sales',
        'WITH recent AS (SELECT * FROM orders) '
        'SELECT COUNT(DISTINCT order_id) FROM recent',
    )
    assert metric_sql_issue(
        'total_sales',
        'WITH recent AS (SELECT * FROM orders) '
        'SELECT SUM(total_sales) FROM recent',
    ) is None
    assert metric_sql_issue(
        'total_sales',
        'WITH recent AS (SELECT total_sales, order_id FROM orders) '
        'SELECT COUNT(order_id) FROM recent',
    )
    assert metric_sql_issue(
        'total_sales',
        'WITH x AS (SELECT SUM(total_sales) AS revenue FROM orders) '
        'SELECT SUM(revenue) FROM x',
    ) is None
    assert metric_sql_issue(
        'total_sales',
        'SELECT COUNT(order_id) FROM orders '
        'UNION ALL SELECT COUNT(order_id) FROM orders',
    ) is None


def test_metric_mismatch_blocks_wrong_sql_from_same_model_verdict():
    from test_database_gateway import _pipeline_functions
    from evaluation import SQL_CHECKS, failed_verdict, SQL_EVALUATOR_PROMPT, sql_review_rules
    from sql_safety import split_statements

    calls = []
    def reviewed(*args, **kwargs):
        calls.append('critic')
        return dict(passed=True, checks={key: True for key in SQL_CHECKS}, issues=[],
                    retry_target=None, fix_instructions='', requested_metric_column='mrr')
    nodes = _pipeline_functions({'critic_node'}, {
        'metric_sql_issue': metric_sql_issue, 'SQL_EVALUATOR_PROMPT': SQL_EVALUATOR_PROMPT,
        'sql_review_rules': sql_review_rules,
        'SQL_CHECKS': SQL_CHECKS, 'failed_verdict': failed_verdict,
        '_split_statements': split_statements, 're': re, 'json': json,
        'APPROVED_SCHEMA': json.dumps({'subscriptions': {'mrr': 'REAL', 'signup_id': 'TEXT'}}),
        'invoke_evaluator': reviewed, 'critic_llm': object(),
    })
    result = nodes['critic_node']({
        'question': 'What is the recurring billing total?',
        'forecast': False,
        'sql': 'SELECT COUNT(DISTINCT signup_id) FROM subscriptions',
    })
    assert result['critic_approved'] is False
    assert result['retry_count'] == 1
    assert 'mrr' in result['critic_reason']
    assert calls == ['critic']


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
    assert historical_monthly_order_count_sql(
        'What is the monthly order total in 2014?',
        {'subscriptions': {'mrr': 'REAL', 'signup_id': 'TEXT'}},
    ) is None


def test_clear_monthly_order_question_bypasses_mistaken_sql_reviewer():
    from test_database_gateway import _pipeline_functions
    from evaluation import SQL_CHECKS
    from sql_safety import split_statements

    calls = []
    nodes = _pipeline_functions({'sql_agent_node', 'critic_node'}, {
        'historical_monthly_order_count_sql': historical_monthly_order_count_sql,
        'metric_sql_issue': metric_sql_issue,
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
