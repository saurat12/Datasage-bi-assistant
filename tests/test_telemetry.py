from types import SimpleNamespace as NS
import pytest
from telemetry import measure_request, tracked_call, mark_cache_hit, price


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
