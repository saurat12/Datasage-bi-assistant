from types import SimpleNamespace

import pandas as pd

from test_database_gateway import _pipeline_functions


def _forecast_result():
    return SimpleNamespace(
        method='ETS (linear trend, none seasonality)',
        forecast=pd.DataFrame({'forecast': [12.0], 'lower': [10.0], 'upper': [14.0]}),
        meta={},
    )


def _forecast_nodes(forecast_series, interpret=lambda question: {'mode': 'default'}):
    return _pipeline_functions({'forecast_agent_node', 'route_after_forecast'}, {
        'pd': pd,
        'extract_target_window': lambda question: None,
        'extract_horizon': lambda question, default: (3, 'months'),
        'forecast_series': forecast_series,
        '_llm_interpret_horizon': interpret,
        '_build_forecast_chart_spec': lambda result: {'mark': 'line'},
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
