import ast
import json
import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from database_gateway import DatabaseGateway, DatabaseGatewayError
from sql_safety import validate_read_only_sql


@pytest.fixture
def gateway(tmp_path):
    path = tmp_path / 'business #1.db'
    with sqlite3.connect(path) as conn:
        conn.executescript('''
            CREATE TABLE orders (region TEXT, total_sales REAL, secret TEXT);
            INSERT INTO orders VALUES ('East', 10, 'private'), ('West', 20, 'hidden');
            CREATE TABLE payroll (salary REAL);
            INSERT INTO payroll VALUES (1000);
        ''')
    return DatabaseGateway(SimpleNamespace(
        database_file=path, db_allowed_tables='orders',
        db_allowed_columns='orders.region,orders.total_sales',
        sqlite_timeout_seconds=1, db_query_timeout_seconds=1,
        sql_row_limit=100, db_forecast_row_limit=10000,
        db_max_result_bytes=2048, db_max_concurrent_queries=1,
    ))


def test_schema_exposes_only_approved_metadata(gateway):
    assert json.loads(gateway.schema_description()) == {
        'orders': {'region': 'TEXT', 'total_sales': 'REAL'},
    }


def test_select_and_cte_aggregations(gateway):
    assert gateway.execute('SELECT SUM(total_sales) AS revenue FROM orders') == [{'revenue': 30.0}]
    assert gateway.execute('WITH x AS (SELECT region FROM orders) SELECT COUNT(*) AS n FROM x') == [{'n': 2}]


@pytest.mark.parametrize('sql', [
    'SELECT * FROM payroll',
    'SELECT COUNT(*) FROM payroll',
    'SELECT salary AS total_sales FROM payroll',
    'SELECT secret AS region FROM orders',
    "SELECT region FROM orders WHERE secret = 'private'",
    'SELECT region FROM orders ORDER BY secret',
    'SELECT * FROM orders',
    'SELECT rowid FROM orders',
    'SELECT name FROM sqlite_master',
    "SELECT * FROM pragma_table_info('orders')",
    'SELECT region FROM orders UNION SELECT secret FROM orders',
    "SELECT load_extension('anything')",
    'DELETE FROM orders',
    'SELECT 1; SELECT 2',
])
def test_permission_bypasses_are_denied(gateway, sql):
    with pytest.raises(ValueError):
        gateway.execute(sql)


def test_row_limits_reject_instead_of_truncating(gateway):
    gateway.settings.sql_row_limit = 1
    with pytest.raises(DatabaseGatewayError, match='row limit'):
        gateway.execute('SELECT region FROM orders')
    assert len(gateway.execute('SELECT region FROM orders', forecast=True)) == 2
    gateway.settings.db_forecast_row_limit = 1
    with pytest.raises(DatabaseGatewayError, match='row limit'):
        gateway.execute('SELECT region FROM orders', forecast=True)


def test_result_bytes_are_bounded(gateway):
    gateway.settings.db_max_result_bytes = 1024
    sql = "SELECT '" + ('x' * 600) + "' AS payload FROM orders"
    with pytest.raises(DatabaseGatewayError, match='byte limit'):
        gateway.execute(sql)


def test_query_deadline_and_slot_cleanup(gateway):
    gateway.settings.db_query_timeout_seconds = 0.01
    with pytest.raises(DatabaseGatewayError, match='time limit'):
        gateway.execute('''WITH RECURSIVE n(x) AS (
            SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x < 100000000
        ) SELECT SUM(x) FROM n''')
    gateway.settings.db_query_timeout_seconds = 1
    assert gateway.execute('SELECT 1 AS ok') == [{'ok': 1}]


def test_concurrency_limit(gateway):
    gateway._slots.acquire()
    try:
        with pytest.raises(DatabaseGatewayError, match='busy'):
            gateway.execute('SELECT 1')
    finally:
        gateway._slots.release()


def _pipeline_functions(names, namespace):
    # Import selected pure nodes without initializing models or reading .env.
    path = Path(__file__).resolve().parents[1] / 'bi.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace.update(BIState=dict, logger=logging.getLogger('test'))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def test_rejected_sql_never_executes(gateway):
    calls = []
    nodes = _pipeline_functions({'fallback_node', 'executor_node'}, {
        '_run_sql': lambda *args, **kwargs: calls.append(args),
    })
    state = nodes['fallback_node']({'sql': 'SELECT region FROM orders'})
    result = nodes['executor_node'](state)
    assert not calls
    assert result['error'] and result['rows'] == []


def test_approved_sql_still_passes_through_gateway(gateway):
    nodes = _pipeline_functions({'executor_node'}, {
        '_run_sql': gateway.execute, '_sanitize_single_statement': lambda sql: sql,
    })
    result = nodes['executor_node']({'sql': 'SELECT secret FROM orders', 'critic_approved': True})
    assert result['error'] and result['rows'] == []


def test_reviewer_uncertainty_returns_data_without_marking_sql_approved(gateway):
    nodes = _pipeline_functions({'fallback_node', 'executor_node'}, {
        'validate_read_only_sql': validate_read_only_sql,
        '_run_sql': gateway.execute, '_sanitize_single_statement': lambda sql: sql,
    })
    pending = nodes['fallback_node']({'sql': 'SELECT region FROM orders',
        'sql_check_issues': [], 'critic_approved': False})
    result = nodes['executor_node'](pending)
    assert result['review_fallback'] and not result['critic_approved']
    assert len(result['rows']) == 2 and result['error'] is None


@pytest.mark.parametrize('sql', ['DELETE FROM orders', 'SELECT secret FROM orders', 'SELECT * FROM payroll'])
def test_uncertain_review_never_bypasses_database_permissions(gateway, sql):
    nodes = _pipeline_functions({'fallback_node', 'executor_node'}, {
        'validate_read_only_sql': validate_read_only_sql,
        '_run_sql': gateway.execute, '_sanitize_single_statement': lambda sql: sql,
    })
    pending = nodes['fallback_node']({'sql': sql, 'sql_check_issues': [], 'critic_approved': False})
    result = nodes['executor_node'](pending)
    assert result['error'] and result['rows'] == []


def test_known_deterministic_issues_still_block(gateway):
    nodes = _pipeline_functions({'fallback_node'}, {'validate_read_only_sql': validate_read_only_sql})
    result = nodes['fallback_node']({'sql': 'SELECT region FROM orders',
        'sql_check_issues': ['Required forecast dimension missing.']})
    assert not result.get('review_fallback') and result['sql'] is None


def test_generator_has_no_database_tools():
    seen = []
    model = SimpleNamespace(invoke=lambda messages: (seen.append(messages) or SimpleNamespace(content='SQL Query: SELECT 1;')))
    nodes = _pipeline_functions({'sql_agent_node'}, {
        'sql_llm': model, '_build_sql_prompt': lambda forecast: 'approved schema',
        'extract_sql': lambda output: output.split('SQL Query:')[1].strip(),
    })
    result = nodes['sql_agent_node']({'question': 'count orders', 'forecast': False})
    assert result['sql'] == 'SELECT 1;'
    assert seen[0][0] == ('system', 'approved schema')
