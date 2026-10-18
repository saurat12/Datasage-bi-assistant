"""Query Planning: Conservative templates for clear historical and forecast questions."""

import re


_SQL_TOKEN = re.compile(r'"(?:""|[^"])*"|`(?:``|[^`])*`|\[[^]]*\]|[A-Za-z_]\w*|[(),.*]')
_AGGREGATES = {'sum', 'count', 'avg', 'min', 'max', 'total', 'group_concat'}


def _sql_tokens(sql):
    """Discard comments and literals before inspecting SQL identifiers."""
    scrubbed = re.sub(r"--[^\n]*|/\*[\s\S]*?\*/|'(?:''|[^'])*'", ' ', sql)
    return _SQL_TOKEN.findall(scrubbed)


def _identifier(token):
    if token.startswith(('"', '`', '[')):
        return token[1:-1].replace(token[0] * 2, token[0]).casefold()
    return token.casefold()


def metric_sql_issue(metric_column, sql):
    """Check the evaluator's schema column against direct SELECT aggregates.

    Complex CTE/subquery projections remain the evaluator's responsibility:
    their outer aggregates may legitimately reference a derived alias instead.
    """
    if metric_column is None or not sql:
        return None
    tokens = _sql_tokens(sql)
    if not tokens or _identifier(tokens[0]) != 'select':
        return None
    depth = 0
    projection = []
    for token in tokens[1:]:
        word = _identifier(token)
        if depth == 0 and word == 'from':
            break
        if word == 'select':
            return None  # Derived projection: do not reject equivalent SQL.
        projection.append(token)
        depth += (token == '(') - (token == ')')
    metric = metric_column.casefold()
    for index, token in enumerate(projection[:-1]):
        if _identifier(token) not in _AGGREGATES or projection[index + 1] != '(':
            continue
        depth = 1
        for member in projection[index + 2:]:
            depth += (member == '(') - (member == ')')
            if depth == 0:
                break
            if _identifier(member) == metric:
                return None
    return f"The SQL aggregation does not use the requested metric column {metric_column!r}."


def historical_monthly_order_count_sql(question, schema):
    """Orders-dataset optimization only; return None for any other schema/scope.

    This template deliberately knows the orders vocabulary and columns. The
    general SQL agent handles every question it does not recognize.
    """
    words = re.findall(r"[a-z]+|\d+", question.lower())
    allowed = set('what is are was were the total number count of how many orders order '
                  'each every per by month monthly in for during year please'.split())
    years = [word for word in words if re.fullmatch(r'[12]\d{3}', word)]
    if (len(years) != 1 or any(word not in allowed and word != years[0] for word in words)
            or not {'order', 'orders'} & set(words)
            or not re.search(r'\bmonthly\b|\b(?:each|every|per|by)\s+month\b', question, re.IGNORECASE)
            or not re.search(r'\b(?:in|for|during)\s+(?:the\s+)?(?:year\s+)?[12]\d{3}\b', question, re.IGNORECASE)
            or not {'order_date', 'order_id'}.issubset(schema.get('orders', {}))):
        return None
    return ("SELECT strftime('%Y-%m', order_date) AS period, "
            "COUNT(DISTINCT order_id) AS monthly_orders "
            "FROM orders WHERE strftime('%Y', order_date) = '" + years[0]
            + "' GROUP BY period ORDER BY period ASC;")


def forecast_history_sql(question, schema):
    # Orders-dataset optimization only; unknown schemas and vocabulary fall
    # through to the general SQL agent without applying this template.
    # Unknown words deliberately fall back to the normal generator, so filters
    # such as 'West', 'excluding returns' or 'using history since 2015' aren't lost.
    words = re.findall(r"[a-z]+|\d+", question.lower())
    allowed = set('what will would be the forecast forecasted forecasting predict predicted '
                  'prediction project projected projection estimate estimated total number '
                  'count of order orders revenue sales in for each every month months monthly '
                  'next year years annual annually yearly please'.split())
    if not words or any(word not in allowed and not word.isdigit() for word in words):
        return None
    order_count = bool({'order', 'orders'} & set(words))
    revenue = bool({'revenue', 'sales'} & set(words))
    if order_count == revenue:
        return None
    columns = schema.get('orders', {})
    metric = 'order_id' if order_count else 'total_sales'
    if not {'order_date', metric}.issubset(columns):
        return None
    annual = bool({'annual', 'annually', 'yearly'} & set(words)) or bool(re.search(r'\b(?:each|every)\s+year\b', question.lower()))
    if annual and {'month', 'months', 'monthly'} & set(words):
        return None
    period_format = '%Y-01-01' if annual else '%Y-%m-01'
    alias = 'annual_orders' if annual else 'monthly_orders'
    expression = f'COUNT(DISTINCT order_id) AS {alias}' if order_count else 'SUM(total_sales) AS total_sales'
    return (f"SELECT strftime('{period_format}', order_date) AS period, " + expression
            + ' FROM orders GROUP BY period ORDER BY period ASC;')
