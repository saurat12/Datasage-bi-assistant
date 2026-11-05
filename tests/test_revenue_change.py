"""Revenue-change diagnostics stay inside the approved SQLite gateway."""

import json
import calendar
import re
from decimal import Decimal, InvalidOperation
import sqlite3
from types import SimpleNamespace
from typing import Optional

import pandas as pd

from database_gateway import DatabaseGateway
from evaluation import build_evidence, check_statements
from query_planning import revenue_change_sql
from test_database_gateway import _pipeline_functions

_attach_period_citations = _pipeline_functions(
    {'_attach_period_citations'}, {'re': re, 'pd': pd, 'calendar': calendar},
)['_attach_period_citations']

_currency_helpers = _pipeline_functions(
    {'_is_currency_metric', '_currency_display', '_format_currency_chart'},
    {'re': re, 'Decimal': Decimal, 'InvalidOperation': InvalidOperation},
)


def test_followup_drop_summary_accepts_negative_currency_change():
    rows = [
        {'period': '2015-01', 'breakdown': 'overall', 'revenue': 200.0,
         'order_count': 2, 'revenue_per_order': 100.0},
        {'period': '2015-02', 'breakdown': 'overall', 'revenue': 123.0,
         'order_count': 2, 'revenue_per_order': 61.5,
         'revenue_change': -77.0},
    ]
    evidence = build_evidence(rows)
    describe = _pipeline_functions({'_revenue_change_statements'}, {
        'Optional': Optional, 'check_statements': check_statements,
        **_currency_helpers,
    })['_revenue_change_statements']
    statements = describe(rows, evidence)
    assert statements is not None
    assert 'a change of -$77.00' in statements[0]['text']
    assert not check_statements(statements, evidence)


def test_followup_summary_cites_named_periods_from_referenced_rows():
    rows = [{'period': '2015-01', 'revenue': 129287.53},
            {'period': '2015-02', 'revenue': 52926.44,
             'revenue_change': -76361.09}]
    evidence = build_evidence(rows)
    statements = [{'text': ('Revenue fell from $129,287.53 in January 2015 '
                            'to $52,926.44 in February 2015, a decline of $76,361.09.'),
                   'fact_ids': ['rows[0].revenue', 'rows[1].revenue',
                                'rows[1].revenue_change']}]
    _attach_period_citations(statements, evidence)
    assert 'rows[0].period' in statements[0]['fact_ids']
    assert 'rows[1].period' in statements[0]['fact_ids']
    assert not check_statements(statements, evidence)


def test_followup_summary_cites_product_names_containing_numbers():
    evidence = build_evidence([{'item': 'HON 5400 Series Task Chair',
                                'revenue': 10.0}])
    statements = [{'text': 'HON 5400 Series Task Chair contributed $10.00.',
                   'fact_ids': ['rows[0].revenue']}]
    _attach_period_citations(statements, evidence)
    assert 'rows[0].item' in statements[0]['fact_ids']
    assert not check_statements(statements, evidence)


def test_final_comparison_revision_keeps_grounded_answer():
    rows = [
        {'period': '2016-09', 'breakdown': 'overall', 'revenue': 200.0,
         'order_count': 2, 'revenue_per_order': 100.0},
        {'period': '2016-10', 'breakdown': 'overall', 'revenue': 123.0,
         'order_count': 2, 'revenue_per_order': 61.5,
         'revenue_change': -77.0},
    ]
    nodes = _pipeline_functions({'summary_node', '_revenue_change_statements'}, {
        'settings': SimpleNamespace(summary_sample_rows=15,
                                    max_evaluation_retries=2),
        'Optional': Optional, 'build_evidence': build_evidence,
        'check_statements': check_statements,
        '_safe_rows_for_llm': lambda data, limit: data[:limit],
        'summary_llm': SimpleNamespace(invoke=lambda _: (_ for _ in ()).throw(
            AssertionError('final revision should use grounded fallback'))),
        **_currency_helpers,
    })
    result = nodes['summary_node']({
        'question': 'Why did revenue drop in October 2016?',
        'rows': rows, 'driver_analysis': True,
        'evaluation_retry_count': 2,
    })
    assert result['summary_source'] == 'grounded_fallback'
    assert 'Distinct orders changed' in result['summary']
    assert not check_statements(result['summary_statements'], result['evidence'])


def test_comparison_uses_distinct_orders_and_actual_periods(tmp_path):
    path = tmp_path / 'orders.db'
    with sqlite3.connect(path) as conn:
        conn.executescript('''
            CREATE TABLE orders (order_id TEXT, order_date TEXT, product_name TEXT,
                                 total_sales REAL, quantity INTEGER, discount REAL);
            INSERT INTO orders VALUES
              ('a', '2016-05-04', 'Desk', 70, 1, 0),
              ('a', '2016-05-04', 'Chair', 30, 1, 0.1),
              ('b', '2016-05-06', 'Desk', 100, 1, 0),
              ('c', '2016-06-01', 'Desk', 40, 1, 0.2),
              ('d', '2016-06-08', 'Chair', 60, 1, 0),
              ('e', '2016-07-01', 'Desk', 900, 1, 0);
        ''')
    settings = SimpleNamespace(
        database_file=path, db_allowed_tables='orders', db_allowed_columns='',
        sqlite_timeout_seconds=1, db_query_timeout_seconds=1,
        sql_row_limit=100, db_forecast_row_limit=10000,
        db_max_result_bytes=100000, db_max_concurrent_queries=1,
    )
    gateway = DatabaseGateway(settings)
    sql = revenue_change_sql('Why did revenue drop in June 2016?',
                             json.loads(gateway.schema_description()))
    rows = gateway.execute(sql)
    overall = sorted((row for row in rows if row['breakdown'] == 'overall'),
                     key=lambda row: row['period'])
    assert [row['period'] for row in overall] == ['2016-05', '2016-06']
    assert [(row['revenue'], row['order_count']) for row in overall] == [(200, 2), (100, 2)]
    assert [row['revenue_per_order'] for row in overall] == [100, 50]
    assert overall[1]['previous_revenue'] == 200
    assert overall[1]['revenue_change'] == -100
    assert any(row['breakdown'] == 'product' and row['item'] == 'Desk' for row in rows)
    assert any(row['breakdown'] == 'discount' and row['item'] == 'Discounted' for row in rows)

    nodes = _pipeline_functions({'_revenue_change_statements', 'viz_node'}, {
        'Optional': Optional,
        'check_statements': check_statements,
        **_currency_helpers,
        'chart_result': lambda *args: None,
        'generate_chart_spec': lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('driver chart must be deterministic')),
    })
    evidence = build_evidence(rows, 15)
    statements = nodes['_revenue_change_statements'](rows, evidence)
    assert statements and not check_statements(statements, evidence)
    assert 'Distinct orders' in ' '.join(item['text'] for item in statements)
    assert 'Revenue per unit sold' in ' '.join(item['text'] for item in statements)
    assert 'discounted line items' in ' '.join(item['text'] for item in statements)
    assert any('For Desk, product revenue changed' in item['text']
               for item in statements)
    assert not any('For Chair, product revenue changed' in item['text']
                   for item in statements)
    calls = []
    summary = _pipeline_functions({'summary_node'}, {
        'json': json,
        'settings': SimpleNamespace(summary_sample_rows=15),
        '_safe_rows_for_llm': lambda data, limit: data[:limit],
        'build_evidence': build_evidence,
        'parse_statements': __import__('evaluation').parse_statements,
        '_attach_period_citations': _attach_period_citations,
        'SUMMARY_SYSTEM_PROMPT': '', 'STRUCTURED_SUMMARY_PROMPT': '',
        'summary_llm': SimpleNamespace(invoke=lambda messages: (
            calls.append(messages) or SimpleNamespace(
                content=json.dumps({'statements': statements})))),
    })['summary_node']({'question': 'Why did revenue drop in June 2016?',
                        'rows': rows, 'sql': sql, 'driver_analysis': True,
                        'forecast': False})
    assert len(calls) == 1
    assert summary['summary_source'] == 'llm'
    assert not summary['summary_error']
    assert not check_statements(summary['summary_statements'], summary['evidence'])
    chart = nodes['viz_node']({'rows': rows, 'driver_analysis': True,
                               'forecast': False, 'question': 'Why did revenue drop?'})
    assert chart['chart_spec']['data']['values'] == [
        {'period': '2016-05', 'revenue': 200},
        {'period': '2016-06', 'revenue': 100},
    ]


def test_comparison_declines_unknown_scope_or_schema():
    schema = {'orders': {column: 'TEXT' for column in
                         ('order_date', 'order_id', 'total_sales', 'product_name')}}
    assert revenue_change_sql('Why did revenue drop in June 2016?', schema)
    assert revenue_change_sql('Why did revenue increase in January 2017?', schema)
    assert revenue_change_sql("What's driving the revenue drop in Jun 2016?", schema)
    assert "2016-12-01" in revenue_change_sql(
        'What caused revenue to increase in Jan 2017?', schema)
    assert revenue_change_sql('Why did revenue drop in Northeast in June 2016?', schema) is None
    assert revenue_change_sql('Why did revenue drop in June 2016 versus June 2015?', schema) is None
    assert revenue_change_sql('Why did revenue drop in June 2016?', {'billing': {}}) is None


def test_increase_summary_names_product_contributing_to_increase():
    rows = [
        {'period': '2017-01', 'breakdown': 'overall', 'revenue': 100.0,
         'order_count': 2, 'revenue_per_order': 50.0},
        {'period': '2017-02', 'breakdown': 'overall', 'revenue': 140.0,
         'order_count': 2, 'revenue_per_order': 70.0},
        {'period': '2017-01', 'breakdown': 'product', 'item': 'Growing item',
         'revenue': 10.0},
        {'period': '2017-02', 'breakdown': 'product', 'item': 'Growing item',
         'revenue': 80.0},
        {'period': '2017-01', 'breakdown': 'product', 'item': 'Declining item',
         'revenue': 90.0},
        {'period': '2017-02', 'breakdown': 'product', 'item': 'Declining item',
         'revenue': 60.0},
    ]
    describe = _pipeline_functions({'_revenue_change_statements'}, {
        'Optional': Optional, 'check_statements': check_statements,
        **_currency_helpers,
    })['_revenue_change_statements']
    statements = describe(rows, build_evidence(rows, 15))
    assert any('For Growing item, product revenue changed' in item['text']
               for item in statements)
    assert not any('For Declining item, product revenue changed' in item['text']
                   for item in statements)


def test_followup_comparison_uses_vetted_template_through_critic_node():
    schema = {'orders': {column: 'TEXT' for column in
                         ('order_date', 'order_id', 'total_sales', 'product_name')}}
    sql_agent = _pipeline_functions({'sql_agent_node'}, {
        'APPROVED_SCHEMA': json.dumps(schema), 'json': json,
        'revenue_change_sql': revenue_change_sql,
        'forecast_history_sql': lambda *args: None,
        'historical_monthly_order_count_sql': lambda *args: None,
    })['sql_agent_node']
    state = sql_agent({'question': 'Why did revenue drop in June 2016?',
                       'original_question': 'Why did revenue drop in June 2016?',
                       'conversation_context': {'summary': 'Prior trend'},
                       'context_used': True, 'forecast': False, 'retry_count': 0})
    assert state['driver_analysis'] is True
    assert state['sql_plan_verified'] is True
    assert 'COUNT(DISTINCT order_id)' in state['sql']
    critic = _pipeline_functions({'critic_node'}, {
        'APPROVED_SCHEMA': json.dumps(schema), 'json': json, 're': re,
        'SQL_CHECKS': ('answers_question', 'schema_valid',
                       'aggregation_correct', 'rules_followed'),
        '_split_statements': lambda sql: [sql],
        'invoke_evaluator': lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('fixed template must not use model approval')),
    })['critic_node']
    reviewed = critic(state)
    assert reviewed['critic_approved'] is True
    assert reviewed['sql_evaluation']['source'] == 'deterministic_revenue_change_plan'
