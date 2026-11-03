"""Request-local telemetry; never stores prompts, rows, or credentials."""
from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
import json
import logging
import math
import os
from threading import Lock
import time
import uuid

_current = ContextVar('request_metrics', default=None)
_current_lock = ContextVar('request_metrics_lock', default=None)
_request_start = ContextVar('request_start', default=None)
_stage_override = ContextVar('telemetry_stage_override', default=None)
_concurrent_work = ContextVar('telemetry_concurrent_work', default=False)
DEFAULT_PRICES = {'gpt-4o-mini': [0.15, 0.075, 0.60],
                  'gpt-4o-mini-2024-07-18': [0.15, 0.075, 0.60],
                  'gpt-4o': [2.50, 1.25, 10.00],
                  'gpt-4o-2024-08-06': [2.50, 1.25, 10.00]}


def price(model, inputs, cached, outputs):
    try:
        rates = {**DEFAULT_PRICES, **json.loads(os.getenv('MODEL_PRICES_JSON', '{}'))}.get(model)
        if not rates or len(rates) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in rates):
            return None
        return ((inputs - cached) * rates[0] + cached * rates[1] + outputs * rates[2]) / 1_000_000
    except (ValueError, TypeError):
        return None


def _append_stage(entry, *, model_call=False):
    metrics = _current.get()
    lock = _current_lock.get()
    if lock is None:
        metrics['stages'].append(entry)
        if model_call:
            metrics['calls'].append(entry)
    else:
        with lock:
            metrics['stages'].append(entry)
            if model_call:
                metrics['calls'].append(entry)


def _timing(start, end):
    origin = _request_start.get()
    return dict(start_seconds=round(start - origin, 4),
                end_seconds=round(end - origin, 4),
                seconds=round(end - start, 4))


@contextmanager
def stage_label(name):
    """Name a shared model call by its review role without changing the call."""
    token = _stage_override.set(name)
    try:
        yield
    finally:
        _stage_override.reset(token)


@contextmanager
def concurrent_work():
    """Mark spans that run in a background worker alongside graph nodes."""
    token = _concurrent_work.set(True)
    try:
        yield
    finally:
        _concurrent_work.reset(token)


@contextmanager
def tracked_stage(name):
    """Record one local stage, including failures, in the request timeline."""
    if _current.get() is None:
        yield
        return
    start = time.perf_counter()
    failed = True
    try:
        yield
        failed = False
    finally:
        end = time.perf_counter()
        _append_stage(dict(stage=name, **_timing(start, end), failed=failed,
                           concurrent=_concurrent_work.get()))


def tracked_call(stage, model, call, /, *args, **kwargs):
    metrics = _current.get()
    if metrics is None:
        return call(*args, **kwargs)
    start = time.perf_counter()
    response = None
    failed = True
    try:
        response = call(*args, **kwargs)
        failed = False
        return response
    finally:
        end = time.perf_counter()
        usage = getattr(response, 'usage_metadata', None)
        actual_model = (getattr(response, 'response_metadata', None) or {}).get('model_name', model)
        if usage is not None:
            inputs, outputs = usage.get('input_tokens', 0), usage.get('output_tokens', 0)
            cached = (usage.get('input_token_details') or {}).get('cache_read', 0)
        else:
            usage = getattr(response, 'usage', None)
            actual_model = getattr(response, 'model', None) or actual_model
            inputs = getattr(usage, 'prompt_tokens', 0)
            outputs = getattr(usage, 'completion_tokens', 0)
            cached = getattr(getattr(usage, 'prompt_tokens_details', None), 'cached_tokens', 0) or 0
        cached = min(inputs, cached)
        cost = price(actual_model, inputs, cached, outputs) if usage is not None else None
        entry = dict(stage=stage, model=actual_model,
            **_timing(start, end), failed=failed,
            concurrent=_concurrent_work.get(),
            input_tokens=inputs, output_tokens=outputs, cached_input_tokens=cached,
            usage_reported=usage is not None, estimated_cost_usd=cost)
        _append_stage(entry, model_call=True)


class TrackedModel:
    def __init__(self, model, stage, model_name):
        self.model, self.stage, self.model_name = model, stage, model_name

    def invoke(self, *args, **kwargs):
        return tracked_call(_stage_override.get() or self.stage, self.model_name,
                            self.model.invoke, *args, **kwargs)


def mark_cache_hit():
    if _current.get() is not None:
        _current.get()['cache_hit'] = True


def measure_request(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        metrics = dict(request_id=str(uuid.uuid4()), cache_hit=False,
                       calls=[], stages=[])
        token = _current.set(metrics)
        lock_token = _current_lock.set(Lock())
        start_token = _request_start.set(start)
        try:
            result = fn(*args, **kwargs)
            metrics['outcome'] = result.get('evaluation', {}).get('status', 'unknown')
            calls = metrics['calls']
            metrics.update(latency_seconds=round(time.perf_counter() - start, 4),
                model_calls=len(calls),
                input_tokens=sum(c['input_tokens'] for c in calls),
                output_tokens=sum(c['output_tokens'] for c in calls),
                usage_complete=all(c['usage_reported'] for c in calls),
                estimated_cost_usd=(sum(c['estimated_cost_usd'] for c in calls)
                    if all(c['estimated_cost_usd'] is not None for c in calls) else None))
            logging.getLogger('datasage').info('request_metrics %s', json.dumps(metrics))
            return {**result, 'metrics': metrics}
        finally:
            _request_start.reset(start_token)
            _current_lock.reset(lock_token)
            _current.reset(token)
    return wrapped
