from types import SimpleNamespace as NS
import pytest
from telemetry import (TrackedModel, measure_request, tracked_call,
                       tracked_stage, stage_label, mark_cache_hit, price)


def test_usage_includes_repeated_calls_and_chart():
    @measure_request
    def request():
        for _ in range(2):
            tracked_call('critic', 'gpt-4o-mini', lambda: NS(usage_metadata={'input_tokens': 1000, 'output_tokens': 100, 'input_token_details': {'cache_read': 200}}))
        tracked_call('chart', 'gpt-4o-mini', lambda: NS(model='gpt-4o-mini', usage=NS(prompt_tokens=1000, completion_tokens=100)))
        return {}
    m = request()['metrics']
    assert m['model_calls'] == 3
    assert m['input_tokens'] == 3000
    assert m['estimated_cost_usd'] == pytest.approx(0.00060)


def test_cached_response_metrics_do_not_mutate_original():
    original = {'metrics': {'input_tokens': 500}}
    @measure_request
    def request():
        mark_cache_hit()
        return original
    m = request()['metrics']
    assert m['cache_hit'] and m['estimated_cost_usd'] == 0
    assert original['metrics']['input_tokens'] == 500


def test_failure_and_unknown_pricing_are_not_free():
    @measure_request
    def request():
        try:
            tracked_call('sql', 'gpt-4o-mini', lambda: 1 / 0)
        except ZeroDivisionError:
            pass
        return {'evaluation': {'status': 'blocked'}}
    m = request()['metrics']
    assert m['calls'][0]['failed']
    assert not m['usage_complete']
    assert m['estimated_cost_usd'] is None
    assert price('unknown', 10, 0, 10) is None


def test_price_override(monkeypatch):
    monkeypatch.setenv('MODEL_PRICES_JSON', '{"custom": [1, 0.5, 2]}')
    assert price('custom', 1000, 200, 100) == pytest.approx(.0011)
    monkeypatch.setenv('MODEL_PRICES_JSON', 'invalid')
    assert price('custom', 1000, 0, 100) is None
    monkeypatch.delenv('MODEL_PRICES_JSON')
    assert price('gpt-4o', 1000, 0, 100) == pytest.approx(.0035)
    assert price('gpt-4o-2024-08-06', 1000, 0, 100) == pytest.approx(.0035)


@pytest.mark.parametrize('tracked', [False, True])
def test_chart_model_keyword_is_forwarded_without_conflicting(tracked):
    def create(*, model, messages):
        assert model == 'gpt-4o-mini'
        assert messages == []
        return NS(model=model, usage=NS(prompt_tokens=10, completion_tokens=5))

    def request():
        response = tracked_call('chart', 'gpt-4o-mini', create,
                                model='gpt-4o-mini', messages=[])
        assert response.model == 'gpt-4o-mini'
        return {}

    if tracked:
        result = measure_request(request)()
        assert result['metrics']['model_calls'] == 1
        assert result['metrics']['estimated_cost_usd'] is not None
    else:
        request()


def test_timeline_records_distinct_review_roles_and_local_stages():
    model = TrackedModel(NS(invoke=lambda messages: NS(usage_metadata={
        'input_tokens': 1, 'output_tokens': 1})), 'critic', 'gpt-4o-mini')

    @measure_request
    def request():
        with stage_label('sql_critic'):
            model.invoke([])
        with tracked_stage('database'):
            pass
        with stage_label('answer_critic'):
            model.invoke([])
        return {}

    metrics = request()['metrics']
    assert [entry['stage'] for entry in metrics['stages']] == [
        'sql_critic', 'database', 'answer_critic']
    assert [entry['stage'] for entry in metrics['calls']] == [
        'sql_critic', 'answer_critic']
    assert metrics['model_calls'] == 2
    for entry in metrics['stages']:
        assert 0 <= entry['start_seconds'] <= entry['end_seconds'] <= metrics['latency_seconds']
        assert entry['seconds'] >= 0 and not entry['failed']


def test_local_stage_failure_and_repeated_attempts_have_separate_spans():
    @measure_request
    def request():
        for attempt in range(2):
            try:
                with tracked_stage('forecast_fit'):
                    if attempt == 0:
                        raise ValueError('retry')
            except ValueError:
                pass
        return {}

    spans = request()['metrics']['stages']
    assert len(spans) == 2
    assert [span['failed'] for span in spans] == [True, False]
    assert spans[0]['end_seconds'] <= spans[1]['start_seconds']
