import json
import re
from typing import Optional

import pandas as pd
import pytest

from test_database_gateway import _pipeline_functions


_classify_chart_type = _pipeline_functions({'_classify_chart_type'}, {
    're': re, 'json': json, 'pd': pd, 'Optional': Optional,
})['_classify_chart_type']
_simple_bar_chart = _pipeline_functions({'_simple_bar_chart'}, {
    'pd': pd, 'List': list, 'Dict': dict, 'Any': object, 'Optional': Optional,
})['_simple_bar_chart']
_regional_product_chart = _pipeline_functions({'_regional_product_chart'}, {
    'pd': pd, 'List': list, 'Dict': dict, 'Any': object, 'Optional': Optional,
})['_regional_product_chart']


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


def test_simple_category_metric_chart_is_complete_and_bounded():
    rows = [{'product_name': f'Product {i}', 'total_sales': float(i)}
            for i in range(100)]
    spec, note = _simple_bar_chart(rows)
    assert spec['mark'] == 'bar'
    assert spec['encoding']['x']['field'] == 'total_sales'
    assert spec['encoding']['y']['field'] == 'product_name'
    assert len(spec['data']['values']) == 20
    assert spec['data']['values'][0]['total_sales'] == 99.0
    assert '20 of 100' in note


@pytest.mark.parametrize('rows', [
    [{'period': '2016-06', 'total_sales': 100}],
    [{'product_name': 'Desk', 'total_sales': None}],
    [{'product_name': 'Desk', 'total_sales': 100, 'region': 'East'}],
])
def test_simple_bar_chart_defers_when_shape_is_not_categorical_metric(rows):
    assert _simple_bar_chart(rows) == (None, None)


def test_regional_product_chart_keeps_both_dimensions():
    rows = [{'region': region, 'product_name': f'Product {i}', 'total_sales': i}
            for region in ('East', 'West') for i in range(12)]
    spec, note = _regional_product_chart(rows)
    assert spec['facet']['field'] == 'region'
    assert spec['spec']['encoding']['y']['field'] == 'product_name'
    assert spec['spec']['encoding']['x']['field'] == 'total_sales'
    assert len(spec['data']['values']) == 20
    assert '10 products per region' in note
    assert _regional_product_chart([{'region': 'East', 'total_sales': 1}]) == (None, None)
