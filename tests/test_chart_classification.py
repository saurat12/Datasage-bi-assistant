import json
import re
from typing import Optional
from types import SimpleNamespace

import pandas as pd
import pytest

from test_database_gateway import _pipeline_functions


_classify_chart_type = _pipeline_functions({'_classify_chart_type'}, {
    're': re, 'json': json, 'pd': pd, 'Optional': Optional,
})['_classify_chart_type']
_simple_bar_chart = _pipeline_functions({'_simple_bar_chart'}, {
    'pd': pd, 'List': list, 'Dict': dict, 'Any': object, 'Optional': Optional,
})['_simple_bar_chart']
_chart_spec_issue = _pipeline_functions({'_chart_spec_issue'}, {
    'Any': object, 'List': list, 'Dict': dict, 'Optional': Optional,
})['_chart_spec_issue']


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


def test_chart_spec_rejects_missing_fields_and_empty_rows():
    spec = {'mark': 'line', 'encoding': {
        'x': {'field': 'period'}, 'y': {'field': 'revenue'},
    }}
    assert _chart_spec_issue(spec, [{'period': '2016-05', 'total_sales': 100}])
    assert _chart_spec_issue(spec, [])
    assert _chart_spec_issue(spec, [{'period': '2016-05', 'revenue': None}])
    assert _chart_spec_issue({**spec, 'transform': [{'filter': 'datum.revenue > 1000'}]},
                             [{'period': '2016-05', 'revenue': 100}])


def test_chart_spec_accepts_returned_fields():
    spec = {'mark': 'line', 'encoding': {
        'x': {'field': 'period'}, 'y': {'field': 'revenue'},
        'tooltip': [{'field': 'period'}, {'field': 'revenue'}],
    }}
    assert _chart_spec_issue(spec, [{'period': '2016-05', 'revenue': 100}]) is None


def test_generated_chart_uses_returned_rows_and_visible_line_points():
    rows = [{'period': '2016-06', 'revenue': 90}]
    model_spec = {'data': {'values': [{'period': '2016-06', 'revenue': 999}]},
                  'mark': 'line', 'encoding': {
                      'x': {'field': 'period'}, 'y': {'field': 'revenue'},
                  }}
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=json.dumps(model_spec)))])
    nodes = _pipeline_functions({'generate_chart_spec'}, {
        'List': list, 'Dict': dict, 'Any': object, 'Optional': Optional,
        'settings': SimpleNamespace(chart_sample_rows=100),
        '_safe_rows_for_llm': lambda data, limit: data[:limit],
        '_simple_bar_chart': lambda data: (None, None),
        '_downsample_for_chart': lambda data: (data, False),
        '_classify_chart_type': lambda question, data: 'line',
        '_chart_spec_issue': _chart_spec_issue,
        'tracked_call': lambda *args, **kwargs: response,
        '_viz_client': SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response))),
        'json': json, 're': re,
    })
    chart, note = nodes['generate_chart_spec'](rows, 'Revenue in June')
    assert chart['data']['values'] == rows
    assert chart['mark'] == {'type': 'line', 'point': True}
    assert note is None
