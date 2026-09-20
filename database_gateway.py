"""The BI application's only SQLite access boundary (Python 3.11+).

Policies are application-wide, not per-user or tenant row permissions.
"""

import json
import sqlite3
import time
from contextlib import closing
from threading import BoundedSemaphore

from sql_safety import validate_read_only_sql


class DatabaseGatewayError(ValueError):
    """A query was denied, failed, or exceeded its resource budget."""


class DatabaseGateway:
    def __init__(self, settings):
        self.settings = settings
        self._tables = {x.strip().lower() for x in settings.db_allowed_tables.split(',') if x.strip()}
        if not self._tables:
            raise ValueError("DB_ALLOWED_TABLES must name at least one table.")
        # Entries are table.column; an empty list allows all columns of approved tables.
        self._columns = {x.strip().lower() for x in settings.db_allowed_columns.split(',') if x.strip()}
        if any(len(x.split('.')) != 2 or not all(x.split('.')) for x in self._columns):
            raise ValueError("DB_ALLOWED_COLUMNS must contain table.column entries.")
        self._slots = BoundedSemaphore(settings.db_max_concurrent_queries)

    def _connect(self):
        conn = sqlite3.connect(
            self.settings.database_file.as_uri() + '?mode=ro', uri=True,
            timeout=min(self.settings.sqlite_timeout_seconds, self.settings.db_query_timeout_seconds),
        )
        try:
            conn.execute('PRAGMA query_only = ON')
            conn.execute('PRAGMA trusted_schema = OFF')
            conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, self.settings.db_max_result_bytes)
            conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 65536)
            conn.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception:
            conn.close()
            raise

    def _column_allowed(self, table, column):
        return not self._columns or f'{table.lower()}.{column.lower()}' in self._columns

    def _authorize(self, action, arg1, arg2, database, source):
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            table, column = (arg1 or '').lower(), arg2 or ''
            # Empty column means COUNT(*) / row existence access.
            # SQLite reports database=None for some optimized row-count reads.
            approved_database = database == 'main' or (database is None and not column)
            if approved_database and table in self._tables and (
                not column or self._column_allowed(table, column)
            ):
                return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            allowed = {
                'abs', 'avg', 'coalesce', 'count', 'date', 'datetime', 'dense_rank',
                'first_value', 'ifnull', 'julianday', 'lag', 'last_value', 'lead',
                'length', 'like', 'lower', 'ltrim', 'max', 'min', 'nullif', 'rank',
                'round', 'row_number', 'rtrim', 'strftime', 'substr', 'substring',
                'sum', 'total', 'trim', 'upper',
            }
            if (arg2 or '').lower() in allowed:
                return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    def schema_description(self):
        """Trusted metadata lookup; never samples business rows or accepts SQL."""
        schema = {}
        with closing(self._connect()) as conn:
            for table in sorted(self._tables):
                entry = conn.execute(
                    "SELECT name FROM sqlite_schema WHERE type='table' AND lower(name)=?",
                    (table,),
                ).fetchone()
                if entry is None:
                    raise ValueError(f"Approved table does not exist: {table}")
                columns = conn.execute('SELECT name, type FROM pragma_table_info(?)', (entry['name'],))
                schema[entry['name']] = {
                    row['name']: row['type'] for row in columns
                    if self._column_allowed(table, row['name'])
                }
        return json.dumps(schema)

    def execute(self, sql, *, forecast=False):
        if not isinstance(sql, str) or len(sql.encode('utf-8')) > 65000:
            raise DatabaseGatewayError('Missing SQL or query text exceeds the size limit.')
        safe = validate_read_only_sql(sql)
        if not self._slots.acquire(blocking=False):
            raise DatabaseGatewayError('Database is busy. Please try again shortly.')
        deadline = time.monotonic() + self.settings.db_query_timeout_seconds
        limit = self.settings.db_forecast_row_limit if forecast else self.settings.sql_row_limit
        try:
            with closing(self._connect()) as conn:
                conn.set_authorizer(self._authorize)
                conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
                cursor = conn.execute(f'SELECT * FROM (\n{safe}\n) AS bi_result LIMIT {limit + 1}')
                rows, size = [], 2
                for row in cursor:
                    if time.monotonic() >= deadline:
                        raise DatabaseGatewayError('Database query exceeded its time limit.')
                    if len(rows) == limit:
                        raise DatabaseGatewayError('Result exceeds the row limit. Narrow the requested scope.')
                    record = dict(row)
                    size += len(json.dumps(record, default=str, ensure_ascii=False).encode('utf-8')) + 2
                    if size > self.settings.db_max_result_bytes:
                        raise DatabaseGatewayError('Result exceeds the byte limit. Narrow the requested scope.')
                    rows.append(record)
                return rows
        except sqlite3.Error as exc:
            if time.monotonic() >= deadline:
                raise DatabaseGatewayError('Database query exceeded its time limit.') from exc
            raise DatabaseGatewayError('Database query failed or was denied by the access/resource policy.') from exc
        finally:
            self._slots.release()
