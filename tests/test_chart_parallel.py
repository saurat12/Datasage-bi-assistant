import threading
from types import SimpleNamespace

from chart_parallel import chart_request, chart_result, schedule_chart
from telemetry import measure_request, tracked_call
from test_database_gateway import _pipeline_functions


def test_chart_runs_while_summary_work_continues_and_is_counted():
    started = threading.Event()
    summary_started = threading.Event()

    def generate(rows, question):
        started.set()
        assert summary_started.wait(2)
        tracked_call('chart', 'gpt-4o-mini', lambda: SimpleNamespace(
            model='gpt-4o-mini', usage=SimpleNamespace(
                prompt_tokens=100, completion_tokens=20)))
        return {'mark': 'line', 'data': rows}, None

    def request():
        with chart_request():
            rows = [{'period': '2016-01', 'monthly_orders': 5}]
            schedule_chart('SELECT monthly_orders', rows, 'monthly orders', generate)
            assert started.wait(2)
            summary_started.set()
            tracked_call('summary', 'gpt-4o-mini', lambda: SimpleNamespace(
                model='gpt-4o-mini', usage=SimpleNamespace(
                    prompt_tokens=200, completion_tokens=40)))
            chart, note = chart_result('SELECT monthly_orders', rows)
            assert chart['mark'] == 'line' and note is None
        return {'evaluation': {'status': 'passed'}}

    metrics = measure_request(request)()['metrics']
    assert {call['stage'] for call in metrics['calls']} == {'chart', 'summary'}
    chart_call = next(call for call in metrics['calls'] if call['stage'] == 'chart')
    assert chart_call['seconds'] >= 0
    assert chart_call['input_tokens'] == 100
    assert chart_call['output_tokens'] == 20
    assert chart_call['estimated_cost_usd'] > 0
    assert metrics['model_calls'] == 2
    assert metrics['estimated_cost_usd'] > chart_call['estimated_cost_usd']


def test_sql_revision_discards_old_chart_and_uses_new_rows():
    old_started = threading.Event()
    release_old = threading.Event()

    def generate(rows, question):
        if rows[0]['monthly_orders'] == 5:
            old_started.set()
            assert release_old.wait(2)
        return {'mark': 'line', 'data': rows}, None

    with chart_request():
        old_rows = [{'period': '2016-01', 'monthly_orders': 5}]
        new_rows = [{'period': '2016-01', 'monthly_orders': 6}]
        schedule_chart('SELECT old', old_rows, 'monthly orders', generate)
        assert old_started.wait(2)
        schedule_chart('SELECT new', new_rows, 'monthly orders', generate)
        assert chart_result('SELECT old', old_rows) is None
        chart, _ = chart_result('SELECT new', new_rows)
        assert chart['data'] == new_rows
        assert chart_result('SELECT new', old_rows) is None
        release_old.set()


def test_executor_starts_only_approved_historical_charts():
    scheduled = []
    nodes = _pipeline_functions({'executor_node'}, {
        '_sanitize_single_statement': lambda sql: sql,
        '_run_sql': lambda sql, forecast: [{'period': '2016-01', 'monthly_orders': 5}],
        'schedule_chart': lambda *args: scheduled.append(args),
        'generate_chart_spec': lambda rows, question: ({'mark': 'line'}, None),
    })
    base = {'sql': 'SELECT monthly_orders', 'question': 'monthly orders'}
    approved = nodes['executor_node']({**base, 'critic_approved': True})
    assert approved['error'] is None and len(scheduled) == 1
    assert scheduled[0][:3] == (base['sql'], approved['rows'], base['question'])
    nodes['executor_node']({**base, 'critic_approved': True, 'forecast': True})
    nodes['executor_node']({**base, 'critic_approved': False, 'review_fallback': True})
    assert len(scheduled) == 1


def test_chart_start_failure_does_not_invalidate_executed_rows():
    def fail_to_start(*args):
        raise RuntimeError('Chart worker unavailable')

    nodes = _pipeline_functions({'executor_node'}, {
        '_sanitize_single_statement': lambda sql: sql,
        '_run_sql': lambda sql, forecast: [{'sales': 5}],
        'schedule_chart': fail_to_start,
        'generate_chart_spec': lambda rows, question: ({'mark': 'bar'}, None),
    })
    result = nodes['executor_node']({
        'sql': 'SELECT sales', 'critic_approved': True, 'question': 'sales',
    })
    assert result['error'] is None
    assert result['rows'] == [{'sales': 5}]


def test_viz_node_uses_prepared_chart_and_never_returns_stale_one():
    generated = []
    nodes = _pipeline_functions({'viz_node'}, {
        'chart_result': chart_result,
        'generate_chart_spec': lambda rows, question: (
            generated.append(rows) or {'mark': 'bar', 'data': rows}, None),
    })
    with chart_request():
        rows = [{'region': 'East', 'sales': 5}]
        schedule_chart('SELECT sales', rows, 'sales',
                       lambda data, question: ({'mark': 'bar', 'data': data}, None))
        state = {'sql': 'SELECT sales', 'rows': rows, 'question': 'sales'}
        assert nodes['viz_node'](state)['chart_spec']['data'] == rows
        assert generated == []
        changed = {**state, 'rows': [{'region': 'East', 'sales': 6}]}
        assert nodes['viz_node'](changed)['chart_spec']['data'] == changed['rows']
        assert generated == [changed['rows']]
        assert nodes['viz_node']({**state, 'forecast': True})['chart_spec'] is None


def test_graph_sql_retry_returns_chart_for_revised_rows_only():
    from langgraph.graph import END, StateGraph

    old_started = threading.Event()
    release_old = threading.Event()
    generated = []
    attempts = []

    def generate(rows, question):
        value = rows[0]['sales']
        generated.append(value)
        if value == 5:
            old_started.set()
            assert release_old.wait(2)
        return {'mark': 'bar', 'data': rows}, None

    def sql_agent(state):
        attempts.append(1)
        return {**state, 'sql': 'SELECT old' if len(attempts) == 1 else 'SELECT new'}

    def summary(state):
        if state['sql'] == 'SELECT old':
            assert old_started.wait(2)
        else:
            release_old.set()
        return state

    def evaluate(state):
        if state['sql'] == 'SELECT old':
            return {**state, 'evaluation': {'status': 'retrying',
                                            'retry_target': 'sql_agent', 'passed': False},
                    'sql': None, 'rows': [], 'critic_approved': False}
        return {**state, 'evaluation': {'status': 'passed', 'passed': True}}

    nodes = _pipeline_functions({'_build_graph', 'executor_node', 'viz_node',
                                 'route_after_evaluation'}, {
        'StateGraph': StateGraph, 'END': END,
        'orchestrator_node': lambda s: {**s, 'forecast': False},
        'sql_agent_node': sql_agent,
        'critic_node': lambda s: {**s, 'critic_approved': True},
        'route_after_critic': lambda s: 'proceed',
        'fallback_node': lambda s: s,
        '_sanitize_single_statement': lambda sql: sql,
        '_run_sql': lambda sql, forecast: [{'sales': 5 if sql == 'SELECT old' else 6}],
        'schedule_chart': schedule_chart,
        'generate_chart_spec': generate,
        'route_after_executor': lambda s: 'summary',
        'forecast_agent_node': lambda s: s,
        'route_after_forecast': lambda s: 'summary',
        'accuracy_agent_node': lambda s: s,
        'summary_node': summary,
        'answer_evaluation_node': evaluate,
        'chart_result': chart_result,
    })
    graph = nodes['_build_graph']()
    with chart_request():
        result = graph.invoke({'question': 'sales'}, {'recursion_limit': 50})

    assert len(attempts) == 2
    assert sorted(generated) == [5, 6]
    assert result['sql'] == 'SELECT new'
    assert result['rows'] == [{'sales': 6}]
    assert result['chart_spec']['data'] == result['rows']
