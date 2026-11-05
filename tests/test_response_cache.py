"""Verified response caching must preserve question and database identity."""

import time
import uuid
import hashlib
import json
from contextlib import nullcontext
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Optional

from telemetry import mark_cache_hit, measure_request
from test_database_gateway import _pipeline_functions


def _ask_with_stub(invoke, database_fingerprint):
    namespace = _pipeline_functions({'_cache_key', '_validated_context', 'ask_bi_agent'}, {
        'time': time, 'uuid': uuid, 'json': json, 'hashlib': hashlib,
        'measure_request': measure_request,
        'Optional': Optional,
        'mark_cache_hit': mark_cache_hit,
        '_request_id': ContextVar('cache_test_request_id', default='startup'),
        '_response_cache': {}, '_database_fingerprint': database_fingerprint,
        'chart_request': nullcontext, '_compiled_graph': SimpleNamespace(invoke=invoke),
        'settings': SimpleNamespace(cache_ttl_seconds=300, max_evaluation_retries=2,
                                    max_evaluator_technical_retries=1),
        'MAX_RETRIES': 2, 'MAX_FORECAST_RETRIES': 1,
    })
    return namespace['_cache_key'], namespace['ask_bi_agent']


def test_cache_key_keeps_meaningful_question_text():
    key = _pipeline_functions({'_cache_key'}, {})['_cache_key']
    assert key('  Revenue trend in 2016?  ') == 'Revenue trend in 2016?'
    assert key('Revenue trend in 2016?') != key('What is the revenue trend in 2016?')
    assert key('Revenue trend in 2016?') != key('Revenue trend in 2017?')
    assert key('Revenue in East?') != key('Revenue in West?')
    assert key('Revenue by region?') != key('Revenue by product?')
    assert key('Revenue for "ABC"?') != key('Revenue for "abc"?')


def test_verified_cache_requires_exact_question_and_same_database_version():
    calls = []
    version = [('main-v1',)]

    def invoke(state, config):
        calls.append(state['question'])
        return {'evaluation': {'passed': True, 'status': 'passed'},
                'summary': 'Revenue changed over the year.',
                'sql': 'SELECT period, revenue FROM sales', 'rows': []}

    _, ask = _ask_with_stub(invoke, lambda: version[0])
    first = ask('Revenue trend in 2016?')
    repeated = ask('Revenue trend in 2016?')
    paraphrase = ask('What is the revenue trend in 2016?')
    version[0] = ('main-v2',)
    after_update = ask('Revenue trend in 2016?')

    assert first['evaluation']['status'] == repeated['evaluation']['status'] == 'passed'
    assert first['sql'] == 'SELECT period, revenue FROM sales'
    assert first['followup_requested'] is False
    assert repeated['metrics']['cache_hit'] is True
    assert repeated['metrics']['model_calls'] == 0
    assert paraphrase['metrics']['cache_hit'] is False
    assert after_update['metrics']['cache_hit'] is False
    assert len(calls) == 3


def test_cache_cannot_reuse_answer_from_another_conversation():
    calls = []

    def invoke(state, config):
        calls.append(state.get('conversation_context'))
        return {'evaluation': {'passed': True, 'status': 'passed'},
                'context_used': True,
                'summary': 'Verified answer', 'rows': [],
                'sql': 'SELECT total_sales FROM orders',
                'chart_spec': {'mark': 'bar'}, 'chart_note': 'comparison'}

    _, ask = _ask_with_stub(invoke, lambda: ('main-v1',))
    east = {'question': 'Revenue in East?', 'summary': 'East revenue was 10.',
            'passed': True, 'database_version': ('main-v1',)}
    west = {'question': 'Revenue in West?', 'summary': 'West revenue was 20.',
            'passed': True, 'database_version': ('main-v1',)}

    first = ask('What about February?', context=east)
    repeated = ask('What about February?', context=east)
    other = ask('What about February?', context=west)

    assert first['metrics']['cache_hit'] is False
    assert repeated['metrics']['cache_hit'] is True
    assert other['metrics']['cache_hit'] is False
    assert first['followup_requested'] is True
    assert first['sql'] is None
    assert repeated['sql'] is None
    assert first['chart_spec'] is None
    assert first['chart_note'] is None
    assert repeated['chart_spec'] is None
    assert calls == [
        {'question': east['question'], 'summary': east['summary']},
        {'question': west['question'], 'summary': west['summary']},
    ]


def test_stale_context_cannot_hit_standalone_cache():
    calls = []

    def invoke(state, config):
        calls.append(state.get('conversation_context'))
        return {'evaluation': {'passed': True, 'status': 'passed'},
                'summary': 'Verified answer', 'rows': [],
                'sql': 'SELECT total_sales FROM orders',
                'chart_spec': {'mark': 'bar'}}

    _, ask = _ask_with_stub(invoke, lambda: ('main-v2',))
    ask('What about February?')
    stale = {'question': 'Revenue in 2015?', 'summary': 'Revenue fell in January.',
             'passed': True, 'database_version': ('main-v1',)}
    result = ask('What about February?', context=stale)

    assert result['metrics']['cache_hit'] is False
    assert result['followup_requested'] is False
    assert result['sql'] == 'SELECT total_sales FROM orders'
    assert result['chart_spec'] == {'mark': 'bar'}
    assert calls == [None, None]


def test_standalone_question_with_available_context_keeps_sql_and_chart():
    def invoke(state, config):
        assert state['conversation_context'] is not None
        return {'evaluation': {'passed': True, 'status': 'passed'},
                'context_used': False, 'summary': 'Verified answer', 'rows': [],
                'sql': 'SELECT 10 AS revenue', 'chart_spec': {'mark': 'bar'}}

    _, ask = _ask_with_stub(invoke, lambda: ('main-v1',))
    prior = {'question': 'Revenue in 2015?', 'summary': 'Revenue rose.',
             'passed': True, 'database_version': ('main-v1',)}
    result = ask('What is revenue in 2017?', context=prior)

    assert result['followup_requested'] is False
    assert result['sql'] == 'SELECT 10 AS revenue'
    assert result['chart_spec'] == {'mark': 'bar'}


def test_no_cache_when_database_version_is_unavailable():
    calls = []

    def invoke(state, config):
        calls.append(state['question'])
        return {'evaluation': {'passed': True, 'status': 'passed'},
                'summary': 'Verified answer', 'rows': []}

    _, ask = _ask_with_stub(invoke, lambda: None)
    ask('Revenue trend in 2016?')
    repeated = ask('Revenue trend in 2016?')

    assert repeated['metrics']['cache_hit'] is False
    assert len(calls) == 2


def test_database_change_during_request_prevents_caching():
    calls = []
    version = [('main-v1',)]

    def invoke(state, config):
        calls.append(state['question'])
        version[0] = ('main-v2',)
        return {'evaluation': {'passed': True, 'status': 'passed'},
                'summary': 'Verified answer', 'rows': []}

    _, ask = _ask_with_stub(invoke, lambda: version[0])
    ask('Revenue trend in 2016?')
    repeated = ask('Revenue trend in 2016?')

    assert repeated['metrics']['cache_hit'] is False
    assert len(calls) == 2


def test_unverified_answer_is_not_cached():
    calls = []

    def invoke(state, config):
        calls.append(state['question'])
        return {'evaluation': {'passed': False, 'status': 'qualified'},
                'summary': 'Unverified answer', 'rows': []}

    _, ask = _ask_with_stub(invoke, lambda: ('main-v1',))
    ask('Revenue trend in 2016?')
    repeated = ask('Revenue trend in 2016?')

    assert repeated['metrics']['cache_hit'] is False
    assert len(calls) == 2


def test_database_fingerprint_detects_main_and_wal_changes(tmp_path):
    database = tmp_path / 'analytics.sqlite'
    database.write_bytes(b'first')
    fingerprint = _pipeline_functions({'_database_fingerprint'}, {
        'DB_FILE': database, 'Optional': Optional,
    })['_database_fingerprint']
    initial = fingerprint()
    database.write_bytes(b'second version')
    updated = fingerprint()
    wal = tmp_path / 'analytics.sqlite-wal'
    wal.write_bytes(b'pending transaction')

    assert initial != updated
    assert updated != fingerprint()
