"""Deterministic SQL controls used before the database is accessed."""

import re
from typing import Optional


FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex)\b",
    re.IGNORECASE,
)


def split_statements(sql: str) -> list[str]:
    statements, current = [], []
    in_single = in_double = False
    index = 0
    while index < len(sql):
        char = sql[index]
        if char == "'" and not in_double:
            if in_single and index + 1 < len(sql) and sql[index + 1] == "'":
                current.extend([char, sql[index + 1]])
                index += 2
                continue
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        if char == ";" and not in_single and not in_double:
            if "".join(current).strip():
                statements.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    if "".join(current).strip():
        statements.append("".join(current).strip())
    return statements


def validate_read_only_sql(sql: Optional[str]) -> str:
    if not sql or not sql.strip():
        raise ValueError("No SQL statement was provided.")
    statements = split_statements(sql)
    if len(statements) != 1:
        raise ValueError("Exactly one SQL statement is allowed.")
    statement = statements[0].strip()
    if not re.match(r"^(select|with)\b", statement, re.IGNORECASE):
        raise ValueError("Only SELECT queries and SELECT-producing CTEs are allowed.")
    scrubbed = re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", "", statement)
    if FORBIDDEN_SQL.search(scrubbed):
        raise ValueError("The query contains a forbidden SQL operation.")
    return statement.rstrip(";")


def apply_row_cap(sql: str, limit: int) -> str:
    safe = validate_read_only_sql(sql)
    return f"SELECT * FROM ({safe}) AS datasage_result LIMIT {int(limit)};"


def looks_like_future_filter(sql: str) -> bool:
    lowered = (sql or "").lower()
    if any(token in lowered for token in ("date('now')", "datetime('now')", "current_date", "current_timestamp")):
        return True
    return bool(re.search(r"where[^;]*order_date\s*(?:>|>=)\s*", lowered))
