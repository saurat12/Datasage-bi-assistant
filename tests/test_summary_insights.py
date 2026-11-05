import math
import json
import re
from decimal import Decimal, InvalidOperation
from types import SimpleNamespace
from typing import Optional

from evaluation import build_evidence, check_statements
from test_database_gateway import _pipeline_functions

_currency_helpers = _pipeline_functions({'_is_currency_metric', '_currency_display'}, {
    're': re, 'Decimal': Decimal, 'InvalidOperation': InvalidOperation,
})


def test_enumerated_periods_get_a_cited_pattern_without_a_new_total():
    rows = [
        {'period': '2023-01', 'net_amount': 10.0},
        {'period': '2023-02', 'net_amount': 5.0},
        {'period': '2023-03', 'net_amount': 20.0},
    ]
    evidence = build_evidence(rows)
    statements = [
        {'text': f"Net amount was {row['net_amount']:.2f} in {row['period']}.",
         'fact_ids': [f'rows[{index}].period', f'rows[{index}].net_amount']}
        for index, row in enumerate(rows)
    ]
    helper = _pipeline_functions({'_add_period_pattern'}, {
        'settings': SimpleNamespace(summary_sample_rows=15), 'math': math,
        **_currency_helpers,
    })['_add_period_pattern']

    result = helper(statements, rows, evidence)

    assert len(result) == 4
    assert 'lowest in 2023-02 at 5.00' in result[-1]['text']
    assert 'highest in 2023-03 at 20.00' in result[-1]['text']
    assert result[-1]['fact_ids'] == [
        'rows[1].period', 'rows[1].net_amount',
        'rows[2].period', 'rows[2].net_amount',
    ]
    assert not check_statements(result, evidence)
    assert 'total' not in result[-1]['text'].lower()


def test_pattern_skips_partial_or_multi_metric_results():
    helper = _pipeline_functions({'_add_period_pattern'}, {
        'settings': SimpleNamespace(summary_sample_rows=15), 'math': math,
        **_currency_helpers,
    })['_add_period_pattern']
    rows = [
        {'period': '2023-01', 'revenue': 10, 'orders': 2},
        {'period': '2023-02', 'revenue': 15, 'orders': 3},
        {'period': '2023-03', 'revenue': 20, 'orders': 4},
    ]
    statements = [{'text': 'Revenue rose.', 'fact_ids': ['rows[0].revenue']}]
    assert helper(statements, rows, build_evidence(rows)) == statements
    assert helper(statements * 3, rows, build_evidence(rows)) == statements * 3
    grouped = [
        {'period': '2023-01', 'region': 'East', 'revenue': 10},
        {'period': '2023-01', 'region': 'West', 'revenue': 15},
        {'period': '2023-02', 'region': 'East', 'revenue': 20},
    ]
    assert helper(statements * 3, grouped, build_evidence(grouped)) == statements * 3


def test_trend_summary_is_generic_cited_and_does_not_add_totals():
    rows = [
        {'period': '2023-01', 'net_amount': 10.0},
        {'period': '2023-02', 'net_amount': 5.0},
        {'period': '2023-03', 'net_amount': 20.0},
    ]
    evidence = build_evidence(rows)
    helper = _pipeline_functions({'_trend_summary_statements'}, {
        'settings': SimpleNamespace(summary_sample_rows=15),
        're': re, 'math': math, 'Optional': Optional,
        'pd': __import__('pandas'), 'calendar': __import__('calendar'),
        'check_statements': check_statements,
        **_currency_helpers,
    })['_trend_summary_statements']

    statements = helper('Show the net amount trend in 2023', rows, evidence)

    assert len(statements) == 4
    assert 'finished higher than it began' in statements[0]['text']
    assert '10.00 in January 2023 to 20.00 in March 2023' in statements[0]['text']
    assert 'low came in February 2023 at 5.00' in statements[1]['text']
    assert 'peak came in March 2023' in statements[1]['text']
    assert 'sharpest decline' in statements[2]['text']
    assert 'January 2023 to February 2023' in statements[2]['text']
    assert 'largest increase' in statements[3]['text']
    assert 'February 2023 to March 2023' in statements[3]['text']
    assert not check_statements(statements, evidence)
    assert 'total' not in ' '.join(item['text'].lower() for item in statements)
    assert helper('Net amount in each month of 2023', rows, evidence) is None
    assert helper('Show the trend', rows + rows, evidence) is None


def test_2015_revenue_trend_uses_model_for_grounded_narrative():
    amounts = [129287.53, 52926.44, 199570.30, 143565.60,
               167432.57, 96498.49, 149309.88, 186699.67,
               334950.08, 140431.98, 376661.92, 407239.83]
    rows = [{'period': f'2015-{month:02d}', 'revenue': amount}
            for month, amount in enumerate(amounts, 1)]
    response = json.dumps({'statements': [
        {'text': 'Revenue rose from $129,287.53 in January 2015 to $407,239.83 in December 2015.',
         'fact_ids': ['rows[0].period', 'rows[0].revenue',
                      'rows[11].period', 'rows[11].revenue']},
        {'text': 'February 2015 was the low point at $52,926.44.',
         'fact_ids': ['rows[1].period', 'rows[1].revenue']},
        {'text': 'Revenue fell from $334,950.08 in September 2015 to $140,431.98 in October 2015.',
         'fact_ids': ['rows[8].period', 'rows[8].revenue',
                      'rows[9].period', 'rows[9].revenue']},
    ]})
    calls = []
    nodes = _pipeline_functions({'summary_node'}, {
        'settings': SimpleNamespace(summary_sample_rows=15),
        're': re, 'math': math, 'Optional': Optional,
        'pd': __import__('pandas'), 'calendar': __import__('calendar'),
        'Dict': dict, 'Any': object, 'List': list, 'json': json,
        'check_statements': check_statements, 'build_evidence': build_evidence,
        '_safe_rows_for_llm': lambda values, limit: values[:limit],
        'parse_statements': __import__('evaluation').parse_statements,
        '_attach_period_citations': lambda statements, evidence: statements,
        'SUMMARY_SYSTEM_PROMPT': '', 'STRUCTURED_SUMMARY_PROMPT': '',
        'summary_llm': SimpleNamespace(invoke=lambda messages: (
            calls.append(messages) or SimpleNamespace(content=response))),
        **_currency_helpers,
    })
    result = nodes['summary_node']({'question': 'Revenue trend in 2015?', 'rows': rows})
    assert len(calls) == 1
    assert result['summary_source'] == 'llm'
    assert len(result['summary_statements']) == 3
    assert '$129,287.53' in result['summary']
    assert '$407,239.83' in result['summary']
    assert 'February 2015 was the low point' in result['summary']
    assert '\\$' not in result['summary']
    assert not check_statements(result['summary_statements'], result['evidence'])


def test_2015_grounded_fallback_explains_pattern_without_repeating_amounts():
    amounts = [129287.53, 52926.44, 199570.30, 143565.60,
               167432.57, 96498.49, 149309.88, 186699.67,
               334950.08, 140431.98, 376661.92, 407239.83]
    rows = [{'period': f'2015-{month:02d}', 'revenue': amount}
            for month, amount in enumerate(amounts, 1)]
    evidence = build_evidence(rows)
    helper = _pipeline_functions({'_trend_summary_statements'}, {
        'settings': SimpleNamespace(summary_sample_rows=15),
        're': re, 'math': math, 'Optional': Optional,
        'pd': __import__('pandas'), 'calendar': __import__('calendar'),
        'check_statements': check_statements,
        **_currency_helpers,
    })['_trend_summary_statements']

    statements = helper('Revenue trend in 2015?', rows, evidence)
    summary = ' '.join(item['text'] for item in statements)

    assert len(statements) == 4
    assert 'finished higher than it began' in summary
    assert 'low came in February 2015 at $52,926.44' in summary
    assert 'peak came in December 2015' in summary
    assert 'sharpest decline between reported periods came from September 2015 to October 2015' in summary
    assert 'largest increase between reported periods came from October 2015 to November 2015' in summary
    assert summary.count('$407,239.83') == 1
    assert not check_statements(statements, evidence)


def test_final_trend_revision_uses_grounded_fallback():
    rows = [{'period': '2017-01', 'revenue': 10.0},
            {'period': '2017-02', 'revenue': 5.0},
            {'period': '2017-03', 'revenue': 20.0}]
    nodes = _pipeline_functions({'summary_node', '_trend_summary_statements'}, {
        'settings': SimpleNamespace(summary_sample_rows=15,
                                    max_evaluation_retries=2),
        're': re, 'math': math, 'Optional': Optional,
        'pd': __import__('pandas'), 'calendar': __import__('calendar'),
        'Dict': dict, 'Any': object, 'List': list,
        '_safe_rows_for_llm': lambda data, limit: data[:limit],
        'build_evidence': build_evidence, 'check_statements': check_statements,
        'summary_llm': SimpleNamespace(invoke=lambda _: (_ for _ in ()).throw(
            AssertionError('final revision should use grounded fallback'))),
        **_currency_helpers,
    })
    result = nodes['summary_node']({
        'question': 'Revenue trend in 2017?', 'rows': rows,
        'evaluation_retry_count': 2,
    })
    assert result['summary_source'] == 'grounded_fallback'
    assert len(result['summary_statements']) == 4
    assert not check_statements(result['summary_statements'], result['evidence'])


def test_short_model_trend_requests_revision_before_fallback():
    rows = [{'period': '2017-01', 'revenue': 10.0},
            {'period': '2017-02', 'revenue': 5.0},
            {'period': '2017-03', 'revenue': 20.0}]
    response = json.dumps({'statements': [
        {'text': 'Revenue was $20.00 in March 2017.',
         'fact_ids': ['rows[2].period', 'rows[2].revenue']},
    ]})
    nodes = _pipeline_functions({'summary_node', '_trend_summary_statements'}, {
        'settings': SimpleNamespace(summary_sample_rows=15),
        're': re, 'math': math, 'Optional': Optional,
        'pd': __import__('pandas'), 'calendar': __import__('calendar'),
        'Dict': dict, 'Any': object, 'List': list, 'json': json,
        '_safe_rows_for_llm': lambda data, limit: data[:limit],
        'build_evidence': build_evidence, 'check_statements': check_statements,
        'parse_statements': __import__('evaluation').parse_statements,
        '_attach_period_citations': lambda statements, evidence: statements,
        'SUMMARY_SYSTEM_PROMPT': '', 'STRUCTURED_SUMMARY_PROMPT': '',
        'summary_llm': SimpleNamespace(invoke=lambda _: SimpleNamespace(content=response)),
        **_currency_helpers,
    })
    result = nodes['summary_node']({'question': 'Revenue trend in 2017?', 'rows': rows})
    assert 'several grounded observations' in result['summary_error']
    assert result['summary'] == ''


def test_review_feedback_uses_llm_summary_revision():
    response = json.dumps({'statements': [
        {'text': 'Net amount was 20.00 in 2023-03.',
         'fact_ids': ['rows[2].period', 'rows[2].net_amount']},
    ]})
    calls = []
    nodes = _pipeline_functions({'summary_node'}, {
        'Dict': dict, 'Any': object, 'List': list, 'json': json,
        'settings': SimpleNamespace(summary_sample_rows=15),
        '_safe_rows_for_llm': lambda rows, limit: rows[:limit],
        'build_evidence': build_evidence,
        '_trend_summary_statements': lambda *args: None,
        '_add_period_pattern': lambda statements, rows, evidence: statements,
        'parse_statements': __import__('evaluation').parse_statements,
        '_attach_period_citations': lambda statements, evidence: statements,
        'SUMMARY_SYSTEM_PROMPT': '', 'STRUCTURED_SUMMARY_PROMPT': '',
        'summary_llm': SimpleNamespace(invoke=lambda messages: (
            calls.append(messages) or SimpleNamespace(content=response))),
    })
    rows = [
        {'period': '2023-01', 'net_amount': 10.0},
        {'period': '2023-02', 'net_amount': 5.0},
        {'period': '2023-03', 'net_amount': 20.0},
    ]

    result = nodes['summary_node']({
        'question': 'Show the net amount trend in 2023', 'rows': rows,
        'evaluation_feedback': 'Explain the trend more clearly.',
    })

    assert len(calls) == 1
    assert result['summary_source'] == 'llm'
    assert not check_statements(result['summary_statements'], result['evidence'])
