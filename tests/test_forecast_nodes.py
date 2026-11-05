from types import SimpleNamespace
import re

import pandas as pd

from test_database_gateway import _pipeline_functions
from telemetry import measure_request


def test_forecast_chart_formats_revenue_without_changing_values():
    build = _pipeline_functions({'_build_forecast_chart_spec'}, {
        'pd': pd, 'ForecastResult': object,
    })['_build_forecast_chart_spec']
    result = SimpleNamespace(
        history=pd.DataFrame({'date': [pd.Timestamp('2017-12-01')],
                              'value': [100.0]}),
        forecast=pd.DataFrame({'date': [pd.Timestamp('2018-01-01')],
                               'value': [120.0], 'lower': [110.0], 'upper': [130.0]}),
    )
    chart = build(result, currency=True)
    assert all(layer['encoding']['y']['axis']['format'] == '$,.2f'
               for layer in chart['layer'])
    assert chart['layer'][0]['encoding']['tooltip'][1]['format'] == '$,.2f'
    assert chart['layer'][0]['data']['values'][0]['value'] == 120.0
    count_chart = build(result)
    assert 'axis' not in count_chart['layer'][0]['encoding']['y']


def _forecast_result():
    return SimpleNamespace(
        method='ETS (linear trend, none seasonality)',
        forecast=pd.DataFrame({'forecast': [12.0], 'lower': [10.0], 'upper': [14.0]}),
        meta={},
    )


def _forecast_nodes(forecast_series, interpret=lambda question: {'mode': 'default'}, target=None):
    return _pipeline_functions({'forecast_agent_node', 'route_after_forecast'}, {
        'pd': pd, 're': re,
        'extract_target_window': lambda question: target,
        'extract_horizon': lambda question, default: (3, 'months'),
        'forecast_series': forecast_series,
        '_llm_interpret_horizon': interpret,
        '_build_forecast_chart_spec': lambda result, *, currency=False: {'mark': 'line'},
        '_is_currency_metric': lambda field: field in {'revenue', 'total_sales'},
        'MAX_FORECAST_RETRIES': 1,
    })


def test_forecast_success_routes_to_accuracy():
    expected = _forecast_result()
    expected.meta['warnings'] = ['Model limitation.']
    calls = []
    nodes = _forecast_nodes(lambda df, **kwargs: (
        calls.append(kwargs) or expected))
    state = {'forecast': True, 'question': 'Forecast revenue',
             'rows': [{'period': '2017-01', 'revenue': 10.0}]}
    result = nodes['forecast_agent_node'](state)
    assert result['forecast_result'] is expected
    assert result['forecast_chart_spec'] == {'mark': 'line'}
    assert result['forecast_error'] is None
    assert result['forecast_df']['revenue'].tolist() == [10.0]
    assert result['forecast_result'].meta['warnings'] == ['Model limitation.']
    assert calls[0]['periods'] == 3
    assert nodes['route_after_forecast'](result) == 'accuracy'


def test_forecast_followup_keeps_result_without_building_chart():
    nodes = _forecast_nodes(lambda df, **kwargs: _forecast_result())
    result = nodes['forecast_agent_node']({
        'forecast': True, 'question': 'Forecast revenue',
        'conversation_context': {'question': 'Previous insight'},
        'context_used': True,
        'rows': [{'period': '2017-01', 'revenue': 10.0}],
    })
    assert result['forecast_result'] is not None
    assert result['forecast_chart_spec'] is None


def test_absolute_additive_forecast_records_target_total():
    expected = SimpleNamespace(
        method='test model',
        forecast=pd.DataFrame({'value': [10.0, 20.0, 30.0],
                               'lower': [9.0, 19.0, 29.0]}),
        meta={'target_window': 'Q1 2018'},
    )
    nodes = _forecast_nodes(lambda df, **kwargs: expected,
                            target=('2018-01-01', '2018-03-31', 'Q1 2018'))
    state = nodes['forecast_agent_node']({
        'forecast': True, 'question': 'What will be the revenue in Q1 of 2018?',
        'sql': 'SELECT period, SUM(total_sales) FROM orders GROUP BY period',
        'rows': [{'period': '2017-12-01', 'total_sales': 100.0}],
    })
    assert state['forecast_result'].meta['target_total'] == 60.0
    assert state['forecast_error'] is None


def test_forecast_and_backtest_record_separate_local_spans():
    result = _forecast_result()
    nodes = _forecast_nodes(lambda df, **kwargs: result)
    nodes.update(_pipeline_functions({'accuracy_agent_node'}, {
        'pd': pd,
        'quick_accuracy': lambda df, horizon: SimpleNamespace(
            mape=5.0, n_splits=1, n_points=3),
    }))

    @measure_request
    def request():
        state = nodes['forecast_agent_node']({
            'forecast': True, 'question': 'Forecast revenue',
            'rows': [{'period': '2017-01', 'revenue': 10.0}],
        })
        nodes['accuracy_agent_node'](state)
        return {}

    stages = request()['metrics']['stages']
    assert [entry['stage'] for entry in stages] == [
        'forecast_fit', 'forecast_backtest']
    assert stages[0]['end_seconds'] <= stages[1]['start_seconds']


def test_empty_history_skips_forecast_retry():
    calls = []
    nodes = _forecast_nodes(lambda *args, **kwargs: calls.append(1))
    result = nodes['forecast_agent_node']({
        'forecast': True, 'question': 'Forecast revenue', 'rows': [],
        'forecast_retry_count': 0,
    })
    assert result['forecast_retry_count'] == 0
    assert 'No historical data' in result['forecast_error']
    assert nodes['route_after_forecast'](result) == 'summary'
    assert not calls


def test_failed_forecast_retries_with_reinterpreted_horizon():
    expected = _forecast_result()
    calls = []
    interpreted = []

    def forecast(df, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ValueError('Horizon needs revision')
        return expected

    def interpret(question):
        interpreted.append(question)
        return {'mode': 'relative', 'periods': 6, 'unit': 'months'}

    nodes = _forecast_nodes(forecast, interpret)
    state = {'forecast': True, 'question': 'Project revenue',
             'rows': [{'period': '2017-01', 'revenue': 10.0}]}
    first = nodes['forecast_agent_node'](state)
    assert first['forecast_retry_count'] == 1
    assert nodes['route_after_forecast'](first) == 'retry'
    second = nodes['forecast_agent_node'](first)
    assert second['forecast_result'] is expected
    assert nodes['route_after_forecast'](second) == 'accuracy'
    assert [call['periods'] for call in calls] == [3, 6]
    assert interpreted == ['Project revenue']


def test_forecast_failure_after_retry_routes_to_summary():
    nodes = _forecast_nodes(lambda df, **kwargs: (_ for _ in ()).throw(ValueError('No fit')))
    state = {'forecast': True, 'question': 'Forecast revenue',
             'rows': [{'period': '2017-01', 'revenue': 10.0}],
             'forecast_retry_count': 1}
    result = nodes['forecast_agent_node'](state)
    assert result['forecast_retry_count'] == 2
    assert nodes['route_after_forecast'](result) == 'summary'


def test_accuracy_node_attaches_backtest_metrics():
    expected = _forecast_result()
    metrics = SimpleNamespace(mape=8.5, n_splits=2, n_points=6)
    seen = []
    nodes = _pipeline_functions({'accuracy_agent_node'}, {
        'pd': pd,
        'quick_accuracy': lambda df, horizon: (
            seen.append((df.copy(), horizon)) or metrics),
    })
    state = {'forecast': True, 'forecast_result': expected,
             'forecast_df': pd.DataFrame({
                 'period': ['2017-01', '2017-02'], 'revenue': [10.0, 12.0],
             })}
    result = nodes['accuracy_agent_node'](state)
    assert result['forecast_result'].meta['accuracy'] is metrics
    assert seen[0][1] == 3
    assert seen[0][0]['revenue'].tolist() == [10.0, 12.0]
