import pytest

from evaluation import build_evidence, check_statements, executed_order_filters, executed_order_year


SQL = """SELECT region, SUM(total_sales) AS total_sales
FROM orders WHERE strftime('%Y', order_date) = '2017'
GROUP BY region ORDER BY total_sales DESC LIMIT 100;"""


def test_executed_year_supports_regional_statement_without_becoming_a_sales_value():
    evidence = build_evidence([{'region': 'West', 'total_sales': 123.45}], executed_sql=SQL)
    statement = {'text': 'West sales in 2017 were 123.45.', 'fact_ids': ['rows[0].region', 'rows[0].total_sales']}
    assert not check_statements([statement], evidence)
    assert evidence['facts']['query.order_year']['value'] == '2017'
    assert evidence['facts']['rows[0].total_sales']['value'] == 123.45
    for text in ['West sales in 2018 were 123.45.', 'West sales in 2017 were 999.99.']:
        assert check_statements([{**statement, 'text': text}], evidence)


@pytest.mark.parametrize('sql', [
    "SELECT 2017 AS total_sales FROM orders",
    "SELECT region FROM orders -- WHERE strftime('%Y', order_date) = '2017'",
    "SELECT 'WHERE strftime(''%Y'', order_date) = ''2017''' FROM orders",
    SQL.replace("= '2017'", "= '2017' OR region = 'West'"),
    SQL.replace("= '2017'", "!= '2017'"),
    SQL + ' SELECT 2018;',
    "SELECT * FROM (SELECT region FROM orders WHERE strftime('%Y', order_date) = '2017') UNION SELECT region FROM orders",
    SQL.replace('FROM orders', 'FROM orders JOIN customers USING (customer_id)'),
    None,
])
def test_unproven_year_context_is_not_added(sql):
    assert executed_order_year(sql) is None
    evidence = build_evidence([{'region': 'West', 'total_sales': 123.45}], executed_sql=sql)
    assert 'query.order_year' not in evidence['facts']


def test_missing_sql_does_not_assume_year_from_question():
    evidence = build_evidence([{'region': 'West', 'total_sales': 123.45}])
    assert check_statements([{'text': 'Sales in 2017 were 123.45.', 'fact_ids': ['rows[0].total_sales']}], evidence)


def test_executed_month_filter_supports_year_in_a_grouped_answer():
    sql = ("SELECT product_name, SUM(total_sales) AS total_sales FROM orders "
           "WHERE strftime('%Y-%m', order_date) = '2016-06' "
           "GROUP BY product_name ORDER BY total_sales DESC LIMIT 100")
    evidence = build_evidence([{'product_name': 'Desk', 'total_sales': 123.45}],
                              executed_sql=sql)
    assert evidence['facts']['query.order_month']['value'] == '2016-06'
    refs = ['rows[0].product_name', 'rows[0].total_sales']
    assert not check_statements([{'text': 'Desk sales in June 2016 were 123.45.',
                                  'fact_ids': refs}], evidence)
    assert check_statements([{'text': 'Desk sales in June 2017 were 123.45.',
                              'fact_ids': refs}], evidence)


def test_exact_month_range_proves_the_same_scope():
    sql = ("SELECT region, product_name, SUM(total_sales) AS total_sales "
           "FROM orders WHERE order_date >= '2015-06-01' "
           "AND order_date < '2015-07-01' "
           "GROUP BY region, product_name ORDER BY total_sales DESC LIMIT 100")
    evidence = build_evidence([{'region': 'East', 'product_name': 'Desk',
                                'total_sales': 123.45}], executed_sql=sql)
    assert evidence['facts']['query.order_month']['value'] == '2015-06'
    refs = ['rows[0].region', 'rows[0].product_name', 'rows[0].total_sales']
    assert not check_statements([{'text': 'East Desk sales in June 2015 were 123.45.',
                                  'fact_ids': refs}], evidence)
    assert executed_order_filters(sql.replace("'2015-07-01'", "'2015-08-01'")) is None


@pytest.mark.parametrize('unsafe_sql', [
    "SELECT product_name, SUM(total_sales) FROM orders WHERE "
    "strftime('%Y-%m', order_date) = '2016-06' OR region = 'West' GROUP BY product_name",
    "SELECT product_name FROM orders WHERE strftime('%Y-%m', order_date) != '2016-06'",
    "SELECT product_name FROM orders WHERE strftime('%Y-%m', order_date) = '2016-13'",
])
def test_unproven_month_context_is_not_added(unsafe_sql):
    assert executed_order_filters(unsafe_sql) is None

@pytest.mark.parametrize('where', [
    "strftime('%Y', order_date) = '2017' AND region = 'East'",
    "region = 'East' AND strftime('%Y', order_date) = '2017'",
])
def test_year_and_region_filters_support_order_count(where):
    sql = f'SELECT COUNT(DISTINCT order_id) AS total_orders FROM orders WHERE {where};'
    evidence = build_evidence([{'total_orders': 123}], executed_sql=sql)
    assert evidence['facts']['query.region']['value'] == 'East'
    statement = {'text': 'East had 123 orders in 2017.', 'fact_ids': ['rows[0].total_orders']}
    assert not check_statements([statement], evidence)
    assert check_statements([{**statement, 'text': 'East had 123 orders in 2018.'}], evidence)
    assert check_statements([{**statement, 'text': 'East had 999 orders in 2017.'}], evidence)
    unsafe = sql.replace("region = 'East'", "region = 'East' OR region = 'West'")
    assert executed_order_year(unsafe) is None
