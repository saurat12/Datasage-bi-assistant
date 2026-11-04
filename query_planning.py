"""Query Planning: Conservative templates for clear historical and forecast questions."""

import re
import calendar
from datetime import date, timedelta


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


def _top_level_selects(tokens):
    depth = 0
    selects = []
    for index, token in enumerate(tokens):
        if depth == 0 and _identifier(token) == 'select':
            selects.append(index)
        depth += (token == '(') - (token == ')')
        if depth < 0:
            return []
    return selects if depth == 0 else []


def _projection_end(tokens, start):
    depth = 0
    for index in range(start + 1, len(tokens)):
        token = tokens[index]
        if depth == 0 and _identifier(token) == 'from':
            return index
        depth += (token == '(') - (token == ')')
    return None


def _simple_cte_select(tokens, metric, outer_select):
    """Accept only a single CTE that passes the metric through unchanged."""
    if len(tokens) < 7 or _identifier(tokens[1]) == 'recursive':
        return False
    cte_name = _identifier(tokens[1])
    if tokens[2:4] != ['AS', '('] and [_identifier(t) for t in tokens[2:4]] != ['as', '(']:
        return False  # Includes CTE column lists, which can rename fields.
    depth = 1
    close = 4
    while close < outer_select and depth:
        depth += (tokens[close] == '(') - (tokens[close] == ')')
        close += 1
    if depth or close != outer_select:
        return False  # Multiple CTEs, joins between CTEs, or malformed shape.
    body = tokens[4:close - 1]
    if not body or _identifier(body[0]) != 'select':
        return False
    if sum(_identifier(token) == 'select' for token in body) != 1:
        return False
    source_from = _projection_end(body, 0)
    if source_from is None or source_from + 1 >= len(body):
        return False
    projection = body[1:source_from]
    fields = []
    field = []
    for token in projection + [',']:
        if token == ',':
            fields.append(field)
            field = []
        else:
            field.append(token)
    if fields == [['*']]:
        passes_metric = True
    else:
        passes_metric = False
        for field in fields:
            if len(field) >= 3 and field[1] == '.':
                field = field[2:]
            if len(field) == 3 and _identifier(field[1]) == 'as':
                if _identifier(field[0]) != _identifier(field[2]):
                    return False
                field = field[:1]
            if len(field) != 1 or field[0] in {'*', '.', '('}:
                return False
            passes_metric |= _identifier(field[0]) == metric
    if not passes_metric:
        return False
    # A single source table and optional WHERE may filter rows without
    # changing column identity. Other relational operations need lineage.
    source_tail = body[source_from + 1:]
    where = next((i for i, token in enumerate(source_tail)
                  if _identifier(token) == 'where'), len(source_tail))
    if len(source_tail[:where]) != 1 or any(
        _identifier(token) in {'join', 'group', 'having', 'union', 'intersect', 'except'}
        for token in source_tail[where:]
    ):
        return False
    outer_from = _projection_end(tokens, outer_select)
    if outer_from is None or outer_from + 1 >= len(tokens):
        return False
    if _identifier(tokens[outer_from + 1]) != cte_name:
        return False
    outer_tail = tokens[outer_from + 2:]
    return not any(_identifier(token) in {'join', 'union', 'intersect', 'except'}
                   for token in outer_tail)


def metric_sql_issue(metric_column, sql):
    """Check the evaluator's schema column against direct SELECT projections.

    Detail queries may project a metric without aggregating it. Only simple
    passthrough CTEs are checked. Complex queries remain the evaluator's
    responsibility; derived aliases can hide a base metric column.
    """
    if metric_column is None or not sql:
        return None
    tokens = _sql_tokens(sql)
    selects = _top_level_selects(tokens)
    if len(selects) != 1:
        return None  # UNION and other compound queries need semantic review.
    start = selects[0]
    if start and (not tokens or _identifier(tokens[0]) != 'with'
                  or not _simple_cte_select(tokens, metric_column.casefold(), start)):
        return None
    if not start and _identifier(tokens[0]) != 'select':
        return None
    depth = 0
    projection = []
    for token in tokens[start + 1:]:
        word = _identifier(token)
        if depth == 0 and word == 'from':
            break
        if word == 'select':
            return None  # Derived projection: do not reject equivalent SQL.
        projection.append(token)
        depth += (token == '(') - (token == ')')
    metric = metric_column.casefold()
    has_aggregate = False
    for index, token in enumerate(projection[:-1]):
        if _identifier(token) not in _AGGREGATES or projection[index + 1] != '(':
            continue
        has_aggregate = True
        depth = 1
        for member in projection[index + 2:]:
            depth += (member == '(') - (member == ')')
            if depth == 0:
                break
            if _identifier(member) == metric:
                return None
    if not has_aggregate:
        if any(_identifier(token) == metric for token in projection):
            return None
        return f"The SQL projection does not use the requested metric column {metric_column!r}."
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


def revenue_change_sql(question, schema):
    """Plan a bounded, read-only month comparison for the supported orders grain.

    Return None when the period, intent, or required columns are unclear; the
    general SQL agent then handles the request. This diagnostic is an orders
    capability, not a claim about arbitrary datasets or causal attribution.
    """
    columns = schema.get('orders', {})
    required = {'order_date', 'order_id', 'total_sales', 'product_name'}
    if not required <= set(columns):
        return None
    if not re.search(r'\b(?:revenue|sales)\b', question, re.IGNORECASE):
        return None
    if not re.search(r'\b(?:why|driv\w*|caus\w*|led|explain|reason\w*)\b',
                     question, re.IGNORECASE):
        return None
    if not re.search(r'\b(?:drop\w*|fell|fall\w*|declin\w*|decreas\w*|dip\w*|increas\w*|ris\w*|rose|grow\w*|growth)\b', question, re.IGNORECASE):
        return None
    names = '|'.join(list(calendar.month_name[1:]) + list(calendar.month_abbr[1:]))
    months = list(re.finditer(rf'\b({names})\s+([12]\d{{3}})\b', question, re.IGNORECASE))
    if len(months) != 1:
        return None
    target = date(int(months[0].group(2)),
                  next(i for i in range(1, 13) if months[0].group(1).casefold()
                       in (calendar.month_name[i].casefold(),
                           calendar.month_abbr[i].casefold())), 1)
    if target.year < 2:
        return None
    prior = (target - timedelta(days=1)).replace(day=1)
    after = (date(target.year + 1, 1, 1) if target.month == 12
             else date(target.year, target.month + 1, 1))
    # Other dates or requested dimensions need the general planner so this
    # template cannot silently discard a comparison or filter.
    if len(re.findall(r'\b[12]\d{3}\b', question)) != 1 or re.search(
            r'\b(?:region|state|city|segment|customer|category|quarter|year)\b',
            question, re.IGNORECASE):
        return None
    # This bounded template has no filters other than the compared months.
    # Extra terms may denote an unknown product, location, or other scope.
    allowed = set('why what s did was were is there a the revenue sales drop dropped '
                  'dip dipped decline declined decrease decreased increase increased '
                  'rise rose grow grew growth fell fall falling driving drove cause '
                  'caused led explain reason reasons behind in for during of month '
                  'monthly compared to previous prior last'.split()).union(
                      name.casefold() for name in
                      list(calendar.month_name[1:]) + list(calendar.month_abbr[1:]))
    if any(word not in allowed and word != str(target.year)
           for word in re.findall(r'[a-z]+|\d+', question.casefold())):
        return None
    unit = 'SUM(quantity)' if 'quantity' in columns else 'NULL'
    discount = 'AVG(discount)' if 'discount' in columns else 'NULL'
    discount_rows = ("UNION ALL SELECT period, 'discount' AS breakdown, "
                     "CASE WHEN discount > 0 THEN 'Discounted' ELSE 'Full price' END AS item, "
                     "SUM(total_sales) AS revenue, COUNT(DISTINCT order_id) AS order_count, "
                     f"{unit} AS units, {discount} AS avg_discount, "
                     "NULL AS revenue_per_order, NULL AS revenue_per_unit "
                     "FROM base WHERE discount IS NOT NULL "
                     "GROUP BY period, item ") if 'discount' in columns else ''
    return (
        "WITH base AS (SELECT strftime('%Y-%m', order_date) AS period, "
        "order_id, product_name, total_sales, "
        + ("quantity" if 'quantity' in columns else "NULL AS quantity") + ", "
        + ("discount" if 'discount' in columns else "NULL AS discount")
        + " FROM orders WHERE order_date >= '" + prior.isoformat()
        + "' AND order_date < '" + after.isoformat() + "'), "
        "periods AS (SELECT '" + prior.strftime('%Y-%m')
        + "' AS period UNION ALL SELECT '" + target.strftime('%Y-%m')
        + "' AS period), "
        "products AS (SELECT period, product_name AS item, SUM(total_sales) AS revenue, "
        "COUNT(DISTINCT order_id) AS order_count, " + unit + " AS units, "
        + discount + " AS avg_discount FROM base WHERE product_name IS NOT NULL "
        "GROUP BY period, product_name), "
        "monthly_change AS (SELECT SUM(CASE WHEN period = '"
        + target.strftime('%Y-%m') + "' THEN total_sales ELSE -total_sales END) "
        "AS delta FROM base), "
        "top_products AS (SELECT item FROM products CROSS JOIN monthly_change "
        "GROUP BY item ORDER BY CASE WHEN MAX(monthly_change.delta) < 0 THEN "
        "SUM(CASE WHEN period = '" + target.strftime('%Y-%m')
        + "' THEN revenue ELSE -revenue END) ELSE -SUM(CASE WHEN period = '"
        + target.strftime('%Y-%m')
        + "' THEN revenue ELSE -revenue END) END ASC LIMIT 4), "
        "combined AS (SELECT period, 'overall' AS breakdown, 'All orders' AS item, "
        "SUM(total_sales) AS revenue, COUNT(DISTINCT order_id) AS order_count, "
        + unit + " AS units, " + discount + " AS avg_discount, "
        "SUM(total_sales) / NULLIF(COUNT(DISTINCT order_id), 0) AS revenue_per_order "
        + (", SUM(total_sales) / NULLIF(SUM(quantity), 0) AS revenue_per_unit "
           if 'quantity' in columns else ", NULL AS revenue_per_unit ")
        + "FROM base GROUP BY period UNION ALL "
        "SELECT periods.period, 'product' AS breakdown, top_products.item, "
        "COALESCE(products.revenue, 0) AS revenue, "
        "COALESCE(products.order_count, 0) AS order_count, "
        "COALESCE(products.units, 0) AS units, products.avg_discount, "
        "NULL AS revenue_per_order, NULL AS revenue_per_unit "
        "FROM top_products CROSS JOIN periods LEFT JOIN products "
        "ON products.item = top_products.item AND products.period = periods.period "
        + discount_rows
        + ") SELECT period, breakdown, item, revenue, order_count, units, "
        "avg_discount, revenue_per_order, revenue_per_unit, "
        "LAG(revenue) OVER (PARTITION BY breakdown, item ORDER BY period) "
        "AS previous_revenue, "
        "revenue - LAG(revenue) OVER (PARTITION BY breakdown, item "
        "ORDER BY period) AS revenue_change "
        "FROM combined ORDER BY breakdown, item, period;"
    )


def forecast_history_sql(question, schema):
    # Orders-dataset optimization only; unknown schemas and vocabulary fall
    # through to the general SQL agent without applying this template.
    # Unknown words deliberately fall back to the normal generator, so filters
    # such as 'West', 'excluding returns' or 'using history since 2015' aren't lost.
    # An absolute quarter names the forecast target, not a historical filter.
    # Strip only one unambiguous target; comparisons and extra scope still
    # fall through to general SQL generation.
    quarter_target = re.compile(
        r'\b(?:q[1-4]\s*(?:of\s+)?[12]\d{3}|'
        r'(?:first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s*(?:of\s+)?[12]\d{3}|'
        r'[12]\d{3}\s+q[1-4])\b', re.IGNORECASE,
    )
    targets = list(quarter_target.finditer(question))
    if len(targets) > 1:
        return None
    normalized = quarter_target.sub(' ', question) if targets else question
    words = re.findall(r"[a-z]+|\d+", normalized.lower())
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
