import json
import calendar
import re
from decimal import Decimal, InvalidOperation
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pandas as pd

from accuracy import AccuracyMetrics
from evaluation import ANSWER_CHECKS, STRUCTURED_SUMMARY_PROMPT, build_evidence, check_statements
from test_database_gateway import _pipeline_functions

_currency_helpers = _pipeline_functions({'_is_currency_metric', '_currency_display'}, {
    're': re, 'Decimal': Decimal, 'InvalidOperation': InvalidOperation,
})
_attach_period_citations = _pipeline_functions(
    {'_attach_period_citations'}, {'re': re, 'pd': pd, 'calendar': calendar},
)['_attach_period_citations']


def test_forecast_citations_only_attach_to_matching_periods():
    evidence = build_evidence([], forecast={
        'horizon': 'Q1 2018',
        'monthly_values': [{'date': '2018-01-01', 'value': 10.0}],
    })
    statements = [
        {'text': 'The Q1 2018 total is an estimate.', 'fact_ids': []},
        {'text': 'January 2018 is forecast at $10.00.',
         'fact_ids': ['forecast.monthly_values[0].value']},
        {'text': 'April 2018 is forecast at $10.00.',
         'fact_ids': ['forecast.monthly_values[0].value']},
    ]
    result = _attach_period_citations(statements, evidence)
    assert result[0]['fact_ids'] == ['forecast.horizon']
    assert 'forecast.monthly_values[0].date' in result[1]['fact_ids']
    assert 'forecast.monthly_values[0].date' not in result[2]['fact_ids']


def _state():
    result = SimpleNamespace(
        forecast=pd.DataFrame({
            'date': pd.date_range('2018-01-01', periods=3, freq='MS'),
            'value': [10.0, 20.0, 30.0],
        }),
        method='test model',
        meta={
            'target_window': 'Q1 2018', 'target_total': 60.0,
            'warnings': ['Treat this forecast as directional.'],
            'accuracy': AccuracyMetrics(1.0, 2.0, 5.0, 4.0, 9, 3),
        },
    )
    return {
        'question': 'What will be the revenue in Q1 of 2018?',
        'forecast': True, 'sql_plan_verified': True, 'critic_approved': True,
        'sql': "SELECT strftime('%Y-%m-01', order_date) AS period, "
               'SUM(total_sales) AS total_sales FROM orders GROUP BY period',
        'rows': [{'period': '2017-12-01', 'total_sales': 100.0}],
        'forecast_result': result,
    }


def test_simple_forecast_summary_uses_model_and_still_gets_answer_review():
    calls = []
    response = json.dumps({'statements': [
        {'text': 'Estimated total for Q1 2018 is $60.00, summed from forecasted values rather than observed sales.',
         'fact_ids': ['forecast.target_total', 'forecast.target_total_scope']},
        {'text': 'Forecast caveat: Treat this forecast as directional.',
         'fact_ids': ['forecast.warnings[0]']},
        {'text': 'Historical backtest error was 5.00% MAPE across 3 validation splits (9 forecast points) on single historical series.',
         'fact_ids': ['forecast.accuracy.metric', 'forecast.accuracy.mape_percent',
                      'forecast.accuracy.validation_splits', 'forecast.accuracy.validation_points',
                      'forecast.accuracy.validation_scope']},
    ]})
    nodes = _pipeline_functions({
        '_forecast_summary_statements', 'summary_node', '_safe_rows_for_llm',
    }, {
        'Optional': Optional, 'Dict': Dict, 'Any': Any, 'List': List,
        'AccuracyMetrics': AccuracyMetrics, 'pd': pd, 'json': json,
        'settings': SimpleNamespace(summary_sample_rows=15, llm_column_allowlist=set()),
        'build_evidence': build_evidence, 'check_statements': check_statements,
        'parse_statements': __import__('evaluation').parse_statements,
        '_attach_period_citations': _attach_period_citations,
        'SUMMARY_SYSTEM_PROMPT': '', 'STRUCTURED_SUMMARY_PROMPT': '',
        **_currency_helpers,
        'summary_llm': SimpleNamespace(invoke=lambda messages: (
            calls.append(messages) or SimpleNamespace(content=response))),
    })
    state = nodes['summary_node'](_state())
    assert len(calls) == 1
    assert state['summary_error'] is None
    assert state['summary_source'] == 'llm'
    assert 'Q1 2018 is $60.00' in state['summary']
    assert 'forecast.horizon' in state['summary_statements'][0]['fact_ids']
    assert not check_statements(state['summary_statements'], state['evidence'])
    assert not any(key.startswith('forecast.forecast_stats.')
                   for key in state['evidence']['facts'])
    assert 'forecast.horizon in every statement' in STRUCTURED_SUMMARY_PROMPT

    reviewed = []
    answer_node = _pipeline_functions({'answer_evaluation_node'}, {
        'ANSWER_CHECKS': ANSWER_CHECKS, 'check_statements': check_statements,
        'invoke_evaluator': lambda *args, **kwargs: (
            reviewed.append(args[2]) or {
                'passed': True, 'checks': {key: True for key in ANSWER_CHECKS},
                'issues': [], 'retry_target': None, 'fix_instructions': '',
            }),
        'apply_answer_applicability': lambda verdict, forecast: verdict,
        'settings': SimpleNamespace(max_evaluation_retries=2),
        'critic_llm': object(), 'ANSWER_EVALUATOR_PROMPT': 'review',
    })['answer_evaluation_node']
    final = answer_node(state)
    assert len(reviewed) == 1
    assert reviewed[0]['historical_sql_reviewed_separately'] is True
    assert 'executed_sql' not in reviewed[0]
    assert any(key.startswith('forecast.') for key in reviewed[0]['evidence']['facts'])
    assert 'rows[0].total_sales' in reviewed[0]['evidence']['facts']
    assert all(ref in reviewed[0]['evidence']['facts']
               for item in state['summary_statements'] for ref in item['fact_ids'])
    assert final['evaluation']['status'] == 'passed'


def test_grouped_forecast_keeps_model_summary_path():
    state = _state()
    state['forecast_result'].forecast['group'] = 'East'
    facts = build_evidence(state['rows'], forecast={'horizon': 'Q1 2018'})
    helper = _pipeline_functions({'_forecast_summary_statements'}, {
        'Optional': Optional, 'AccuracyMetrics': AccuracyMetrics,
        'check_statements': check_statements,
        **_currency_helpers,
    })['_forecast_summary_statements']
    assert helper(state, facts) is None


def test_simple_relative_forecast_cites_each_returned_period():
    state = _state()
    state['forecast_result'].meta.pop('target_total')
    state['forecast_result'].meta.pop('target_window')
    state['forecast_result'].meta['periods'] = 3
    forecast = {
        'monthly_values': state['forecast_result'].forecast.to_dict(orient='records'),
        'horizon': '3 periods',
        'warnings': state['forecast_result'].meta['warnings'],
        'accuracy': {
            'metric': 'historical walk-forward backtest error',
            'mape_percent': 5.0, 'validation_points': 9,
            'validation_splits': 3, 'validation_scope': 'Single historical series',
        },
    }
    evidence = build_evidence(state['rows'], forecast=forecast)
    helper = _pipeline_functions({'_forecast_summary_statements'}, {
        'Optional': Optional, 'AccuracyMetrics': AccuracyMetrics,
        'check_statements': check_statements,
        **_currency_helpers,
    })['_forecast_summary_statements']
    statements = helper(state, evidence)
    assert statements is not None
    assert len([item for item in statements if item['text'].startswith('Forecast total sales')]) == 3
    assert not check_statements(statements, evidence)
