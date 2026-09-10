import pytest

from sql_safety import apply_row_cap, split_statements, validate_read_only_sql


def test_allows_select_and_cte():
    assert validate_read_only_sql("SELECT region FROM orders;") == "SELECT region FROM orders"
    assert validate_read_only_sql("WITH x AS (SELECT 1) SELECT * FROM x;").startswith("WITH")


@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "SELECT 1; DROP TABLE orders",
    "PRAGMA table_info(orders)",
    "ATTACH DATABASE 'x.db' AS x",
])
def test_blocks_unsafe_sql(sql):
    with pytest.raises(ValueError):
        validate_read_only_sql(sql)


def test_semicolon_inside_literal_is_not_a_second_statement():
    assert split_statements("SELECT 'a;b' AS value;") == ["SELECT 'a;b' AS value"]


def test_execution_cap_wraps_query():
    capped = apply_row_cap("SELECT * FROM orders", 100)
    assert capped.startswith("SELECT * FROM (SELECT * FROM orders)")
    assert capped.endswith("LIMIT 100;")
