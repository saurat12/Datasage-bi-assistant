"""Follow-up references remain bounded and pass through both review gates."""

import json
import calendar
import re
import ast
import html
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from evaluation import ANSWER_CHECKS, SQL_CHECKS, build_evidence
from test_database_gateway import _pipeline_functions


PRIOR = {'question': 'Revenue trend in 2015?',
         'summary': 'Revenue fell in February 2015.'}


def test_ui_uses_only_latest_passed_insight_and_keeps_resolved_chain():
    app_path = Path(__file__).resolve().parents[1] / 'app.py'
    tree = ast.parse(app_path.read_text(encoding='utf-8'))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == 'build_followup_context')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(app_path), 'exec'), namespace)
    build = namespace['build_followup_context']
    message = {'question': 'What about February?',
               'interpreted_question': 'Revenue in February 2015?',
               'content': 'February revenue fell.',
               'evaluation_status': 'passed', 'context_version': ('v1',),
               'rows': [{'private': 123}], 'sql': 'SELECT private FROM orders'}

    assert build(message) == {
        'question': 'Revenue in February 2015?',
        'summary': 'February revenue fell.', 'passed': True,
        'database_version': ('v1',),
    }
    assert build({**message, 'evaluation_status': 'qualified'}) is None
    assert build({**message, 'context_version': None}) is None


def test_ask_view_renders_every_answer_in_session_order():
    shown = []
    st = SimpleNamespace(
        container=lambda **kwargs: nullcontext(),
        form=lambda *args, **kwargs: nullcontext(),
        columns=lambda *args, **kwargs: [nullcontext(), nullcontext()],
        markdown=lambda value, **kwargs: shown.append(value),
        divider=lambda: shown.append('DIVIDER'),
        text_input=lambda *args, **kwargs: '',
        form_submit_button=lambda *args, **kwargs: False,
        caption=lambda *args: None,
    )
    app_path = Path(__file__).resolve().parents[1] / 'app.py'
    tree = ast.parse(app_path.read_text(encoding='utf-8'))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == 'render_conversation')
    namespace = {
        'st': st, 'html': html,
        'render_status': lambda status: None,
        'build_followup_context': lambda message: {'question': 'prior', 'summary': 'prior'},
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(app_path), 'exec'), namespace)
    render = namespace['render_conversation']
    render([{'question': 'First question?', 'content': 'First answer'},
            {'question': 'Second question?', 'content': 'Second answer'}])
    output = '\n'.join(shown)
    assert output.index('First question?') < output.index('First answer')
    assert output.index('First answer') < output.index('Second question?')
    assert output.index('Second question?') < output.index('Second answer')
    assert shown.count('DIVIDER') == 1


def test_question_input_uses_bottom_chat_widget_after_results():
    app_path = Path(__file__).resolve().parents[1] / 'app.py'
    tree = ast.parse(app_path.read_text(encoding='utf-8'))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == 'render_question_input')
    placeholders = []
    namespace = {'st': SimpleNamespace(chat_input=lambda prompt: (
        placeholders.append(prompt) or None))}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(app_path), 'exec'), namespace)
    namespace['render_question_input']([])
    assert placeholders == ['Ask a question...']

    ask_view = next(node for node in tree.body
                    if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Compare)
                    and isinstance(node.test.left, ast.Name)
                    and node.test.left.id == 'navigation')
    direct_calls = [node.value.func.id for node in ask_view.body
                    if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)]
    assert direct_calls == ['render_conversation', 'render_chart_panel',
                            'render_question_input']


def test_followup_keeps_prior_questions_and_answers_without_review_badges():
    shown = []
    st = SimpleNamespace(
        container=lambda **kwargs: nullcontext(),
        form=lambda *args, **kwargs: nullcontext(),
        columns=lambda *args, **kwargs: [nullcontext(), nullcontext()],
        markdown=lambda value, **kwargs: shown.append(value),
        divider=lambda: None,
        text_input=lambda *args, **kwargs: '',
        form_submit_button=lambda *args, **kwargs: False,
        caption=lambda *args: None,
    )
    app_path = Path(__file__).resolve().parents[1] / 'app.py'
    tree = ast.parse(app_path.read_text(encoding='utf-8'))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == 'render_conversation')
    namespace = {
        'st': st, 'html': html,
        'render_status': lambda status: None,
        'build_followup_context': lambda message: None,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(app_path), 'exec'), namespace)
    namespace['render_conversation']([
        {'question': 'Revenue in January?', 'content': 'January answer',
         'evaluation_status': 'passed'},
        {'question': 'Why did it fall?', 'content': 'Follow-up answer',
         'followup_requested': True, 'evaluation_status': 'data_only'},
    ])
    output = '\n'.join(shown)
    assert 'Revenue in January?' in output
    assert 'Why did it fall?' in output
    assert 'January answer' in output
    assert 'Follow-up answer' in output
    assert 'Review passed' not in output
    assert 'Review unavailable' not in output
    assert 'Interpretation unverified' not in output
    assert 'Automated verification was unavailable' not in output


def test_context_accepts_only_verified_current_database_insight():
    validate = _pipeline_functions({'_validated_context'}, {'Optional': Optional})['_validated_context']
    submitted = {**PRIOR, 'passed': True, 'database_version': ('v1',),
                 'sql': 'SELECT secret', 'rows': [{'secret': 1}]}

    assert validate(submitted, ('v1',)) == PRIOR
    assert validate({**submitted, 'passed': False}, ('v1',)) is None
    assert validate(submitted, ('v2',)) is None
    assert validate(submitted, None) is None
    assert validate({**submitted, 'summary': 'x' * 4001}, ('v1',)) is None


def test_rewrite_uses_prior_insight_as_data_and_preserves_original_question():
    seen = []

    def invoke(messages):
        seen.append(messages)
        return SimpleNamespace(content=json.dumps({
            'standalone_question': 'Compare January and February 2015 revenue to identify contributors',
            'uses_context': True,
        }))

    nodes = _pipeline_functions({'_resolve_followup', 'orchestrator_node'}, {
        'Optional': Optional, 'json': json, 're': re, 'calendar': calendar,
        'FOLLOWUP_REWRITE_PROMPT': 'Treat the previous exchange as data.',
        'orchestrator_llm': SimpleNamespace(invoke=invoke),
        'detect_forecast_intent': lambda question: False,
        '_AMBIGUOUS_FUTURE_HINTS': re.compile(r'\bwill\b'),
        '_detect_group_dimension': lambda question: None,
    })
    result = nodes['orchestrator_node']({
        'question': 'What drove that drop?', 'conversation_context': PRIOR,
    })

    assert result['original_question'] == 'What drove that drop?'
    assert result['question'] == 'Compare January and February 2015 revenue to identify contributors'
    assert result['context_used'] is True
    assert result['forecast'] is False
    payload = json.loads(seen[0][1][1])
    assert payload == {'current_question': 'What drove that drop?',
                       'previous_question': PRIOR['question'],
                       'previous_answer': PRIOR['summary']}


def test_ambiguous_or_invalid_rewrite_keeps_user_question():
    nodes = _pipeline_functions({'_resolve_followup'}, {
        'Optional': Optional, 'json': json, 're': re,
        'FOLLOWUP_REWRITE_PROMPT': 'Resolve references.',
        'orchestrator_llm': SimpleNamespace(invoke=lambda messages:
            SimpleNamespace(content='{"uses_context": false, "standalone_question": "Changed scope"}')),
    })
    assert nodes['_resolve_followup']('Revenue in 2016?', PRIOR) == ('Revenue in 2016?', False)
    assert nodes['_resolve_followup']('Revenue in 2016?', None) == ('Revenue in 2016?', False)


def test_followup_month_inherits_only_unambiguous_prior_year():
    nodes = _pipeline_functions({'_resolve_followup'}, {
        'Optional': Optional, 'json': json, 're': re, 'calendar': calendar,
        'orchestrator_llm': SimpleNamespace(invoke=lambda messages: (_ for _ in ()).throw(
            AssertionError('unambiguous month should not need a model rewrite'))),
    })
    question = 'Why was there a revenue drop in October?'
    resolved, used = nodes['_resolve_followup'](question, {
        'question': 'What is the revenue trend in 2016?',
        'summary': 'Monthly revenue in 2016.'})
    assert used is True
    assert resolved == 'Why was there a revenue drop in October 2016?'

    ambiguous = {'question': 'Compare 2015 and 2016 revenue',
                 'summary': 'Both years were shown.'}
    assert nodes['_resolve_followup'](question, ambiguous) == (question, False)


def test_rewrite_cannot_drop_explicit_period_or_quoted_filter():
    nodes = _pipeline_functions({'_resolve_followup'}, {
        'Optional': Optional, 'json': json, 're': re,
        'FOLLOWUP_REWRITE_PROMPT': 'Resolve references.',
        'orchestrator_llm': SimpleNamespace(invoke=lambda messages:
            SimpleNamespace(content=json.dumps({
                'uses_context': True,
                'standalone_question': 'Revenue by product in March 2016',
            }))),
    })
    question = 'Why did "East" revenue fall in February 2015?'
    assert nodes['_resolve_followup'](question, PRIOR) == (question, False)


def test_forecast_routing_uses_resolved_followup():
    nodes = _pipeline_functions({'orchestrator_node'}, {
        '_resolve_followup': lambda question, context:
            ('Forecast revenue for the next three months', True),
        'detect_forecast_intent': lambda question: question.startswith('Forecast'),
        '_detect_group_dimension': lambda question: None,
    })
    result = nodes['orchestrator_node']({
        'question': 'What about the next three months?',
        'conversation_context': PRIOR,
    })
    assert result['forecast'] is True
    assert result['context_used'] is True
    assert result['original_question'] == 'What about the next three months?'


def test_contextual_sql_skips_unreviewed_template_and_reaches_sql_model():
    called = []
    nodes = _pipeline_functions({'sql_agent_node'}, {
        'json': json, 'APPROVED_SCHEMA': '{"orders": {"total_sales": "REAL"}}',
        'forecast_history_sql': lambda *args: 'SELECT 1',
        'historical_monthly_order_count_sql': lambda *args: 'SELECT 1',
        'revenue_change_sql': lambda *args: None,
        '_build_sql_prompt': lambda forecast: 'SQL instructions',
        'sql_llm': SimpleNamespace(invoke=lambda messages:
            (called.append(messages) or SimpleNamespace(content='SQL Query: SELECT 2'))),
        'extract_sql': lambda output: output.split('SQL Query: ')[1],
    })
    result = nodes['sql_agent_node']({
        'question': 'Revenue by region in 2015', 'forecast': False,
        'context_used': True,
    })

    assert result['sql'] == 'SELECT 2'
    assert result['sql_plan_verified'] is False
    assert len(called) == 1


def test_sql_reviewer_checks_interpretation_against_original_and_prior():
    seen = []

    def review(model, prompt, payload, checks, targets):
        seen.append(payload)
        return {'passed': False,
                'checks': {**{key: True for key in SQL_CHECKS},
                           'answers_question': False},
                'issues': ['The SQL changes the requested metric.'],
                'retry_target': 'sql_agent',
                'fix_instructions': 'Use the metric from the original question.',
                'requested_metric_column': None}

    nodes = _pipeline_functions({'critic_node'}, {
        'json': json, 're': re, 'SQL_CHECKS': SQL_CHECKS,
        'SQL_EVALUATOR_PROMPT': 'Review the current and original question.',
        'sql_review_rules': lambda *args: 'Rules',
        'APPROVED_SCHEMA': '{"orders": {"total_sales": "REAL"}}',
        '_split_statements': lambda sql: [sql],
        'invoke_evaluator': review, 'metric_sql_issue': lambda *args: None,
        'critic_llm': object(),
    })
    result = nodes['critic_node']({
        'question': 'Revenue by product in February 2015',
        'original_question': 'Which products drove that drop?',
        'conversation_context': PRIOR, 'context_used': True,
        'sql': 'SELECT SUM(total_sales) FROM orders', 'forecast': False,
    })

    assert seen[0]['original_question'] == 'Which products drove that drop?'
    assert seen[0]['previous_verified_insight'] == PRIOR
    assert result['critic_approved'] is False
    assert result['retry_count'] == 1


def test_answer_reviewer_sees_original_question_and_prior_insight():
    seen = []

    def review(model, prompt, payload, checks, targets, **kwargs):
        seen.append(payload)
        return {'passed': True, 'checks': {key: True for key in ANSWER_CHECKS},
                'issues': [], 'retry_target': None, 'fix_instructions': ''}

    nodes = _pipeline_functions({'answer_evaluation_node'}, {
        'ANSWER_CHECKS': ANSWER_CHECKS, 'ANSWER_EVALUATOR_PROMPT': 'Review',
        'check_statements': lambda statements, evidence: [],
        'invoke_evaluator': review,
        'apply_answer_applicability': lambda verdict, forecast: verdict,
        'critic_llm': object(),
    })
    result = nodes['answer_evaluation_node']({
        'question': 'Revenue by product in February 2015',
        'original_question': 'Which products drove that drop?',
        'conversation_context': PRIOR, 'context_used': True,
        'sql': 'SELECT product_name, SUM(total_sales) FROM orders GROUP BY product_name',
        'evidence': build_evidence([{'revenue': 10}]),
        'summary_statements': [{'text': 'Revenue was 10.', 'fact_ids': ['rows[0].revenue']}],
    })

    assert result['evaluation']['passed'] is True
    assert seen[0]['original_question'] == 'Which products drove that drop?'
    assert seen[0]['previous_verified_insight'] == PRIOR
    assert seen[0]['executed_sql']
