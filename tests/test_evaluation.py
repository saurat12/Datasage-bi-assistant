import json
import re
from types import SimpleNamespace

import pytest

from evaluation import (
    ANSWER_CHECKS, SQL_CHECKS, build_evidence, check_statements,
    failed_verdict, parse_statements, parse_verdict,
    apply_answer_applicability,
    sql_review_rules,
    invoke_evaluator,
)
from test_database_gateway import _pipeline_functions


def approved(checks=ANSWER_CHECKS):
    return dict(passed=True, checks={key: True for key in checks}, issues=[],
                retry_target=None, fix_instructions='')


def test_forecast_check_only_applies_to_forecasts():
    verdict = approved()
    verdict.update(passed=False, issues=['Forecast caveat missing'],
                   retry_target='summary_agent', fix_instructions='Add caveat')
    verdict['checks']['forecast_caveats_present'] = False
    assert apply_answer_applicability(verdict, False)['passed']
    assert not apply_answer_applicability(verdict, True)['passed']
    verdict['checks']['numbers_supported'] = False
    assert not apply_answer_applicability(verdict, False)['passed']


def test_sql_reviewer_receives_only_applicable_rules():
    historical = sql_review_rules(False)
    forecast = sql_review_rules(True)
    assert 'WHERE date/year filters explicitly requested' in historical
    assert 'LIMIT is optional' in historical
    assert 'FORECAST HISTORY EXTRACTION' not in historical
    assert 'without LIMIT or future/recent-date filters' in forecast
    assert 'HISTORICAL ANALYSIS' not in forecast


def test_monthly_breakdown_can_contain_twelve_statements():
    statements = [{'text': f'Month {i} revenue is 10.', 'fact_ids': ['example']}
                  for i in range(1, 13)]
    assert len(parse_statements(json.dumps({'statements': statements}))) == 12


def test_sql_rejection_exposes_actual_reason():
    nodes = _pipeline_functions({'fallback_node'}, {})
    result = nodes['fallback_node']({'critic_reason': 'Requested year filter is missing.'})
    assert 'Requested year filter is missing.' in result['error']
    assert not result['critic_approved']


def test_missing_year_citations_get_actionable_feedback():
    evidence = build_evidence([{'period': '2014', 'revenue': 10}])
    statements = [{'text': 'Revenue in 2014 was 10.', 'fact_ids': ['rows[0].revenue']}]
    assert any('rows[0].period' in issue for issue in check_statements(statements, evidence))
    statements[0]['fact_ids'].append('rows[0].period')
    assert not check_statements(statements, evidence)
    assert evidence['scope'].startswith('All returned rows')


@pytest.mark.parametrize('value', ['true', 1, None])
def test_verdict_requires_actual_booleans(value):
    verdict = approved()
    verdict['passed'] = value
    with pytest.raises(ValueError):
        parse_verdict(json.dumps(verdict), ANSWER_CHECKS, {'summary_agent'})


def test_contradictory_verdict_is_rejected():
    verdict = approved()
    verdict['checks']['numbers_supported'] = False
    with pytest.raises(ValueError):
        parse_verdict(json.dumps(verdict), ANSWER_CHECKS, {'summary_agent'})


def test_malformed_review_is_repaired_without_changing_artifact():
    invalid = approved()
    invalid['passed'] = False
    replies = iter([json.dumps(invalid), json.dumps(approved())])
    seen = []
    model = SimpleNamespace(invoke=lambda messages: (
        seen.append(list(messages)) or SimpleNamespace(content=next(replies))))
    verdict = invoke_evaluator(model, 'Review', {'sql': 'SELECT 1'}, ANSWER_CHECKS, {'summary_agent'})
    assert verdict['passed']
    assert len(seen) == 2
    assert seen[0][1] == seen[1][1]
    assert 'contradictory verdict' in seen[1][-1][1]


@pytest.mark.parametrize('withdraw', [True, False])
def test_sql_rejection_is_audited_once_and_real_rejections_still_block(withdraw):
    rejected = failed_verdict(SQL_CHECKS, 'Incorrect aggregation.', 'sql_agent')
    replies = iter([json.dumps(rejected), json.dumps(approved(SQL_CHECKS) if withdraw else rejected)])
    seen = []
    model = SimpleNamespace(invoke=lambda messages: (
        seen.append(list(messages)) or SimpleNamespace(content=next(replies))))
    verdict = invoke_evaluator(model, 'Review SQL', {'candidate_sql': 'SELECT 1'}, SQL_CHECKS, {'sql_agent'})
    assert verdict['passed'] is withdraw
    assert len(seen) == 2
    assert seen[0][1] == seen[1][1]
    assert 'Audit your rejection' in seen[1][-1][1]


def test_full_result_statistics_are_not_sample_statistics():
    evidence = build_evidence([{'revenue': 10}, {'revenue': 20}], sample_limit=1)
    assert evidence['facts']['returned_rows.revenue.sum']['value'] == 30
    assert 'rows[1].revenue' not in evidence['facts']


def test_aggregate_year_context_is_computed_from_all_rows():
    rows = [{'period': '2016-01', 'orders': 10}, {'period': '2016-02', 'orders': 20}]
    statement = [{'text': 'The monthly counts sum to 30 in 2016.', 'fact_ids': ['returned_rows.orders.sum']}]
    assert not check_statements(statement, build_evidence(rows, sample_limit=1))
    rows[1]['period'] = '2017-02'
    assert check_statements(statement, build_evidence(rows, sample_limit=1))


@pytest.mark.parametrize('text,refs,passed', [
    ('Revenue is 12.35.', ['rows[0].revenue'], True),
    ('Revenue is 15.', ['rows[0].revenue'], False),
    ('Revenue is 12.35.', ['invented'], False),
    ('Revenue increased.', [], False),
])
def test_numeric_evidence(text, refs, passed):
    evidence = build_evidence([{'revenue': 12.3456}])
    assert (not check_statements([{'text': text, 'fact_ids': refs}], evidence)) is passed


def answer_nodes(response, retries=1):
    return _pipeline_functions({'answer_evaluation_node', 'route_after_evaluation', 'data_only_summary'}, {
        'ANSWER_CHECKS': ANSWER_CHECKS, 'ANSWER_EVALUATOR_PROMPT': 'evaluate',
        'failed_verdict': failed_verdict, 'check_statements': check_statements,
        'parse_verdict': parse_verdict, 'json': json, 'invoke_evaluator': invoke_evaluator,
        'apply_answer_applicability': apply_answer_applicability,
        'settings': SimpleNamespace(max_evaluation_retries=retries),
        'critic_llm': SimpleNamespace(invoke=lambda messages: SimpleNamespace(content=json.dumps(response))),
    })


def state():
    return {'question': 'What is revenue?', 'sql': 'SELECT 10 AS revenue',
            'evidence': build_evidence([{'revenue': 10}]),
            'summary_statements': [{'text': 'Revenue is 10.', 'fact_ids': ['rows[0].revenue']}],
            'summary': 'Revenue is 10.'}


def test_numeric_failure_overrides_model_approval_and_stops_at_budget():
    nodes = answer_nodes(approved())
    current = state()
    current['summary_statements'][0]['text'] = 'Revenue is 99.'
    revised = nodes['answer_evaluation_node'](current)
    assert nodes['route_after_evaluation'](revised) == 'summary_agent'
    blocked = nodes['answer_evaluation_node'](revised)
    assert nodes['route_after_evaluation'](blocked) == 'end'
    assert blocked['evaluation']['status'] == 'data_only'
    assert '99' not in blocked['summary']


def test_sql_retry_clears_stale_forecast_and_keeps_global_budget():
    nodes = answer_nodes(failed_verdict(ANSWER_CHECKS, 'Wrong metric.', 'sql_agent'))
    current = {**state(), 'forecast_result': 'stale', 'rows': [1], 'forecast_retry_count': 5}
    result = nodes['answer_evaluation_node'](current)
    assert nodes['route_after_evaluation'](result) == 'sql_agent'
    assert result['forecast_result'] is None and result['rows'] == []
    assert result['evaluation_retry_count'] == 1 and result['forecast_retry_count'] == 0
    assert result['critic_reason'] == 'Wrong metric.'


def test_malformed_evaluator_fails_closed():
    nodes = answer_nodes({'passed': 'true'}, retries=0)
    result = nodes['answer_evaluation_node'](state())
    assert result['evaluation']['status'] == 'data_only'


def test_verified_answer_proceeds_to_visualization():
    nodes = answer_nodes(approved())
    result = nodes['answer_evaluation_node'](state())
    assert nodes['route_after_evaluation'](result) == 'viz'


def test_review_fallback_preserves_rows_without_ai_analysis():
    nodes = answer_nodes(approved())
    result = nodes['answer_evaluation_node']({
        'review_fallback': True, 'rows': [{'monthly_orders': 10}],
        'summary': 'Unverified claim', 'sql': 'SELECT 10 AS monthly_orders',
    })
    assert result['evaluation']['status'] == 'data_only'
    assert not result['evaluation']['passed']
    assert result['rows'] == [{'monthly_orders': 10}]
    assert 'Unverified claim' not in result['summary']
    assert nodes['route_after_evaluation'](result) == 'end'


def test_execution_errors_do_not_become_data_only_answers():
    nodes = answer_nodes(approved())
    result = nodes['answer_evaluation_node']({'error': 'Access denied', 'rows': [], 'review_fallback': True})
    assert result['evaluation']['status'] == 'blocked'


def test_forecast_evidence_retains_error_metrics_and_warnings():
    evidence = build_evidence([], forecast={'warnings': ['Limited history'],
        'accuracy': {'mape_percent': 8.25, 'validation_points': 3}})
    assert evidence['facts']['forecast.accuracy.mape_percent']['display'] == '8.25'
    assert evidence['facts']['forecast.warnings[0]']['value'] == 'Limited history'
    assert check_statements([{'text': 'Forecast is available.', 'fact_ids': ['result.row_count']}], evidence)


def test_empty_or_unstructured_summary_is_rejected():
    for raw in ['plain text', '{"statements": []}', '{"statements": [{}]}']:
        with pytest.raises(ValueError):
            parse_statements(raw)


def test_reference_evaluation_dataset():
    from run_evals import run
    report = run()
    assert report['passed'] == report['total'] == 10


def test_real_graph_revises_summary_before_visualization():
    from langgraph.graph import StateGraph, END
    nodes = answer_nodes(approved())
    attempts = []
    visualized = []

    def summarize(current):
        attempts.append(1)
        value = 99 if len(attempts) == 1 else 10
        return {**current, 'summary': f'Revenue is {value}.',
                'summary_statements': [{'text': f'Revenue is {value}.', 'fact_ids': ['rows[0].revenue']}],
                'evidence': build_evidence([{'revenue': 10}])}

    def visualize(current):
        visualized.append(current['summary'])
        return current

    nodes.update(StateGraph=StateGraph, END=END,
        orchestrator_node=lambda s: {**s, 'forecast': False},
        sql_agent_node=lambda s: {**s, 'sql': 'SELECT 10 AS revenue'},
        critic_node=lambda s: {**s, 'critic_approved': True},
        route_after_critic=lambda s: 'proceed', fallback_node=lambda s: s,
        executor_node=lambda s: {**s, 'rows': [{'revenue': 10}]},
        route_after_executor=lambda s: 'summary',
        forecast_agent_node=lambda s: s, route_after_forecast=lambda s: 'summary',
        accuracy_agent_node=lambda s: s, summary_node=summarize, viz_node=visualize)
    graph = _pipeline_functions({'_build_graph'}, nodes)['_build_graph']()
    result = graph.invoke({'question': 'What is revenue?'})
    assert len(attempts) == 2
    assert visualized == ['Revenue is 10.']
    assert result['evaluation']['passed']


def test_sql_evaluation_receives_schema_and_fails_closed_on_outage():
    from evaluation import SQL_EVALUATOR_PROMPT
    seen = []

    def unavailable(messages):
        seen.append(messages)
        raise RuntimeError('unavailable')

    nodes = _pipeline_functions({'critic_node'}, {
        'SQL_CHECKS': SQL_CHECKS, 'SQL_EVALUATOR_PROMPT': SQL_EVALUATOR_PROMPT,
        'invoke_evaluator': invoke_evaluator,
        'sql_review_rules': sql_review_rules,
        'CRITIC_SYSTEM_PROMPT': 'Only SELECT. Respond with ONLY JSON',
        'APPROVED_SCHEMA': '{"orders": {"total_sales": "REAL"}}',
        'json': json, 're': re, 'failed_verdict': failed_verdict, 'parse_verdict': parse_verdict,
        '_split_statements': lambda sql: [sql],
        'critic_llm': SimpleNamespace(invoke=unavailable),
    })
    result = nodes['critic_node']({'question': 'total sales', 'sql': 'SELECT SUM(total_sales) FROM orders'})
    assert not result['critic_approved']
    assert result['retry_count'] == 1
    assert json.loads(seen[0][1][1])['approved_schema']['orders'] == {'total_sales': 'REAL'}


def test_summary_evidence_filters_fields_before_aggregation():
    from evaluation import STRUCTURED_SUMMARY_PROMPT
    seen = []
    response = json.dumps({'statements': [{'text': 'Revenue is 10.', 'fact_ids': ['rows[0].revenue']}]})
    nodes = _pipeline_functions({'summary_node', '_safe_rows_for_llm'}, {
        'Dict': dict, 'Any': object, 'List': list,
        'settings': SimpleNamespace(summary_sample_rows=1, llm_column_allowlist={'revenue'}),
        'json': json, 'build_evidence': build_evidence, 'parse_statements': parse_statements,
        'SUMMARY_SYSTEM_PROMPT': '', 'STRUCTURED_SUMMARY_PROMPT': STRUCTURED_SUMMARY_PROMPT,
        'summary_llm': SimpleNamespace(invoke=lambda messages: (seen.append(messages) or SimpleNamespace(content=response))),
    })
    result = nodes['summary_node']({'question': 'revenue', 'rows': [{'revenue': 10, 'secret': 12345}]})
    assert 'secret' not in seen[0][1][1]
    assert '12345' not in seen[0][1][1]
    assert result['summary'] == 'Revenue is 10.'


@pytest.mark.parametrize('forecast', [False, True])
def test_sql_schema_is_reference_not_output_template(forecast):
    nodes = _pipeline_functions({'_build_sql_prompt'}, {
        'APPROVED_SCHEMA': '{"orders": {"total_sales": "REAL"}}',
        'BASE_SQL_SYSTEM_PROMPT': 'OUTPUT FORMAT: SQL Query: <query>',
        'FORECAST_SQL_ADDENDUM': 'Use full history.',
    })
    prompt = nodes['_build_sql_prompt'](forecast)
    assert prompt.index('</approved_schema>') < prompt.index('OUTPUT FORMAT:')
    assert prompt.endswith('Do not append schema metadata, explanations, or other text after the SQL.')
