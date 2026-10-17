"""Conservative templates for clear historical and forecast questions."""

import re


def requested_metric(question):
    """Identify a single, explicitly requested aggregate metric, if clear."""
    text = (question or '').lower()
    aggregate = bool(re.search(
        r'\b(total|sum|number|count|how many|monthly|yearly|weekly|daily|'
        r'by|per|each|every)\b', text))
    if not aggregate or re.search(
        r'\b(average|avg|median|ratio|rate|percent|amount|value|price|cost|profit|margin)\b',
        text,
    ):
        return None
    order_count = bool(re.search(
        r'\b(?:orders?|order count)\b', text))
    revenue = bool(re.search(r'\b(?:sales|revenue)\b', text))
    quantity = bool(re.search(r'\b(?:quantity|units|items sold)\b', text))
    matches = [name for name, present in (
        ('order_count', order_count), ('revenue', revenue), ('quantity', quantity)) if present]
    return matches[0] if len(matches) == 1 else None


def metric_sql_issue(metric, sql):
    """Catch clear metric substitutions without judging SQL equivalence generally."""
    if metric is None or not sql:
        return None
    column = r'(?:\w+\.)?'
    count_orders = bool(re.search(r'\bcount\s*\(\s*distinct\s+' + column + r'order_id\s*\)', sql, re.IGNORECASE))
    sum_sales = bool(re.search(r'\bsum\s*\(\s*' + column + r'total_sales\s*\)', sql, re.IGNORECASE))
    sum_quantity = bool(re.search(r'\bsum\s*\(\s*' + column + r'quantity\s*\)', sql, re.IGNORECASE))
    if metric == 'order_count' and sum_sales and not count_orders:
        return 'The question requests an order count, but SQL sums sales. Count distinct order_id.'
    if metric == 'revenue' and count_orders and not sum_sales:
        return 'The question requests sales or revenue, but SQL counts orders. Sum total_sales.'
    if metric == 'quantity' and sum_sales and not sum_quantity:
        return 'The question requests quantity, but SQL sums sales. Sum quantity.'
    return None


def historical_monthly_order_count_sql(question, schema):
    """Return a monthly distinct-order count only for an unfiltered year."""
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
