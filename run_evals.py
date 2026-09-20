"""Offline reference regression suite; --live measures the actual BI pipeline."""

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace

from database_gateway import DatabaseGateway
from evaluation import build_evidence, check_statements


def run(live=False):
    dataset = json.loads((Path(__file__).parent / 'evals/cases.json').read_text())
    results = []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'eval.db'
        with closing(sqlite3.connect(path)) as conn:
            conn.execute('CREATE TABLE orders (order_date TEXT, region TEXT, total_sales REAL)')
            conn.executemany('INSERT INTO orders VALUES (:order_date, :region, :total_sales)', dataset['fixture'])
            conn.commit()
        gateway = DatabaseGateway(SimpleNamespace(database_file=path, db_allowed_tables='orders',
            db_allowed_columns='', sqlite_timeout_seconds=1, db_query_timeout_seconds=5,
            sql_row_limit=100, db_forecast_row_limit=10000, db_max_result_bytes=2097152,
            db_max_concurrent_queries=1))
        if live:
            # This CLI is a separate process; never run against the user's database.
            os.environ.update(DB_FILE=str(path), DB_ALLOWED_TABLES='orders', DB_ALLOWED_COLUMNS='',
                              ALLOWED_LLM_COLUMNS='', CACHE_TTL_SECONDS='0')
            from bi import ask_bi_agent
        for case in dataset['cases']:
            try:
                response = ask_bi_agent(case['question']) if live else None
                rows = response['rows'] if live else gateway.execute(case['sql'])
                actual = [list(row.values()) for row in rows]
                matches = actual == case['expected']
                accepted = response['evaluation']['passed'] if live else True
                results.append({'id': case['id'], 'passed': matches and accepted,
                                'result_match': matches, 'evaluation_passed': accepted if live else None})
            except Exception as exc:
                results.append({'id': case['id'], 'passed': False, 'error': type(exc).__name__})
        evidence = build_evidence([{'total_sales': 425.0}])
        for case in dataset['numeric_cases']:
            actual = not check_statements([case], evidence)
            results.append({'id': case['id'], 'passed': actual == case['passed']})
    return {'mode': 'live_pipeline' if live else 'offline_reference_regression',
            'passed': sum(case['passed'] for case in results), 'total': len(results), 'cases': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Call configured LLMs; incurs API usage.')
    args = parser.parse_args()
    report = run(args.live)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] == report['total'] else 1)
