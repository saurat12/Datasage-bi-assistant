import json
import re
from typing import Optional

import pandas as pd
import pytest

from test_database_gateway import _pipeline_functions


_classify_chart_type = _pipeline_functions({'_classify_chart_type'}, {
    're': re, 'json': json, 'pd': pd, 'Optional': Optional,
})['_classify_chart_type']


@pytest.mark.parametrize('question, expected', [
    ('Show the monthly trend', 'line'),
    ('Compare sales', 'bar'),
    ('Relationship between price and sales', 'point'),
    ('Distribution of sales', 'bar'),
    ('MONTHLY proportion by region', 'line'),
    ('Sales by region', 'bar'),
    ('Does price affect demand', 'point'),
    ('How are sales spread out', 'bar'),
])
def test_intent_rules(question, expected):
    assert _classify_chart_type(question, []) == expected


@pytest.mark.parametrize('count, expected', [(1, 'arc'), (7, 'arc'), (8, 'bar')])
def test_share_category_limit(count, expected):
    rows = [{'sales': i + 1, 'region': str(i)} for i in range(count)]
    assert _classify_chart_type('Percentage of sales by region', rows + rows) == expected


@pytest.mark.parametrize('rows, expected', [
    ([{'period': '2017-01', 'sales': 1}], 'line'),
    ([{'recorded': '2017-01-01', 'sales': 1}], 'line'),
    ([{'year': 2017, 'sales': 1}], 'line'),
    ([{'period': '2017-01', 'region': 'East', 'sales': 1}], 'line'),
    ([{'region': 'East', 'sales': 1}], 'bar'),
    ([{'a': 1, 'b': 2, 'c': 3, 'd': 4}], None),
    ([{'a': 1, 'b': 2}], None),
    ([], None),
    ([{'region': None, 'sales': 1}], None),
])
def test_data_shapes(rows, expected):
    assert _classify_chart_type('Show the results', rows) == expected


def test_share_without_categories_defers():
    assert _classify_chart_type('Percentage of total', [{'sales': 1}]) is None


@pytest.mark.parametrize('date_column', ['period', 'recorded'])
@pytest.mark.parametrize('category_count, expected', [(2, 'arc'), (8, 'bar')])
def test_share_counts_categories_instead_of_dates(date_column, category_count, expected):
    rows = [{date_column: f'2017-{month:02d}-01', 'region': f'Region {region}',
             'total_sales': 10}
            for month in range(1, 10) for region in range(category_count)]
    assert _classify_chart_type('Proportion of sales', rows) == expected


def test_share_with_only_date_and_metric_defers():
    assert _classify_chart_type('Percentage of total',
                                [{'period': '2017-01', 'sales': 10}]) is None
