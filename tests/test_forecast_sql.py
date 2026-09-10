import ast
from pathlib import Path


def _load_builder():
    tree = ast.parse(Path("bi.py").read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_build_forecast_sql"]
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = {"Optional": __import__("typing").Optional}
    exec(compile(module, "bi.py", "exec"), namespace)
    return namespace["_build_forecast_sql"]


def test_grouped_forecast_sql_preserves_full_history():
    sql = _load_builder()("region")
    assert "GROUP BY period, region" in sql
    assert "ORDER BY period ASC, region ASC" in sql
    assert "LIMIT" not in sql.upper()
    assert "WHERE" not in sql.upper()
