"""Conservative templates for unfiltered monthly or annual forecast histories."""

import re


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
