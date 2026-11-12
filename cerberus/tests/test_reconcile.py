# -*- coding: utf-8 -*-

from cerberus import Validator, errors
from cerberus.tests import assert_has_error


def make_customer_validator(**extra_item_rules):
    item_schema = {
        'type': 'dict',
        'reconcile': 'code',
        'schema': {
            'code': {'type': 'string'},
            'source_no': {'type': 'integer', 'readonly': True},
            'name': {'type': 'string'},
        },
    }
    item_schema['schema'].update(extra_item_rules)
    schema = {
        'customers': {
            'type': 'list',
            'schema': item_schema,
        }
    }
    return Validator(schema)


def test_readonly_feedback_is_not_a_change():
    # The read-only source number is carried back with the snapshot's value.
    validator = make_customer_validator()
    original = {
        'customers': [
            {'code': 'A', 'source_no': 100, 'name': 'alpha'},
            {'code': 'B', 'source_no': 200, 'name': 'bravo'},
        ]
    }
    patch = {
        'customers': [
            {'code': 'A', 'source_no': 100, 'name': 'alpha'},
            {'code': 'B', 'source_no': 200, 'name': 'bravo'},
        ]
    }
    assert validator.validate(patch, original=original)
    assert validator.errors == {}


def test_reconcile_aligns_reordered_details():
    validator = make_customer_validator()
    original = {
        'customers': [
            {'code': 'A', 'source_no': 100, 'name': 'alpha'},
            {'code': 'B', 'source_no': 200, 'name': 'bravo'},
        ]
    }
    # Keys match after normalization (case); the two details merely swapped place.
    patch = {
        'customers': [
            {'code': 'b', 'source_no': 200, 'name': 'bravo'},
            {'code': 'a', 'source_no': 100, 'name': 'alpha'},
        ]
    }
    assert validator.validate(patch, original=original)

    report = validator.reconciliation[('customers',)]
    assert report['matched'] == {0: 1, 1: 0}
    assert report['added'] == []
    assert report['removed'] == []
    assert report['moved'] == [
        {'key': ('b',), 'from': 1, 'to': 0},
        {'key': ('a',), 'from': 0, 'to': 1},
    ]


def test_reconcile_reports_added_and_removed_details():
    validator = make_customer_validator()
    original = {
        'customers': [
            {'code': 'A', 'source_no': 100, 'name': 'alpha'},
            {'code': 'B', 'source_no': 200, 'name': 'bravo'},
        ]
    }
    patch = {
        'customers': [
            {'code': 'B', 'source_no': 200, 'name': 'bravo'},
            {'code': 'C', 'source_no': 300, 'name': 'charlie'},
        ]
    }
    assert validator.validate(patch, original=original)
    report = validator.reconciliation[('customers',)]
    assert report['added'] == [1]
    assert report['removed'] == [0]
    assert report['matched'] == {0: 1}


def test_reconcile_detects_genuine_readonly_renumber():
    validator = make_customer_validator()
    original = {
        'customers': [
            {'code': 'A', 'source_no': 100, 'name': 'alpha'},
            {'code': 'B', 'source_no': 200, 'name': 'bravo'},
        ]
    }
    # Detail 'A' moved to the second position and its read-only number changed.
    patch = {
        'customers': [
            {'code': 'B', 'source_no': 200, 'name': 'bravo'},
            {'code': 'A', 'source_no': 999, 'name': 'alpha'},
        ]
    }
    assert not validator.validate(patch, original=original)
    assert_has_error(
        validator._errors,
        ('customers', 1, 'source_no'),
        ('customers', 'schema', 'schema', 'source_no', 'readonly'),
        errors.READONLY_VALUE_CHANGED,
        True,
        (100, 999),
    )


def test_reconcile_detects_renumber_without_reorder():
    validator = make_customer_validator()
    original = {'customers': [{'code': 'A', 'source_no': 100, 'name': 'alpha'}]}
    patch = {'customers': [{'code': 'A', 'source_no': 101, 'name': 'alpha'}]}
    assert not validator.validate(patch, original=original)
    assert_has_error(
        validator._errors,
        ('customers', 0, 'source_no'),
        ('customers', 'schema', 'schema', 'source_no', 'readonly'),
        errors.READONLY_VALUE_CHANGED,
        True,
        (100, 101),
    )


def test_reconcile_key_collision_isolates_details_but_keeps_others_judgeable():
    validator = make_customer_validator()
    # The stored snapshot itself holds two details that normalize to key 'a'.
    original = {
        'customers': [
            {'code': 'A', 'source_no': 100, 'name': 'alpha'},
            {'code': 'a', 'source_no': 200, 'name': 'alfa'},
            {'code': 'C', 'source_no': 300, 'name': 'charlie'},
        ]
    }
    patch = {
        'customers': [
            {'code': 'A', 'source_no': 100, 'name': 'alpha'},
            {'code': 'a', 'source_no': 200, 'name': 'alfa'},
            {'code': 'C', 'source_no': 301, 'name': 'charlie'},
        ]
    }
    assert not validator.validate(patch, original=original)

    report = validator.reconciliation[('customers',)]
    assert report['unmatched'] == [
        {'key': ('a',), 'old_positions': (0, 1), 'new_positions': (0, 1)}
    ]
    assert report['matched'] == {2: 2}

    assert_has_error(
        validator._errors,
        ('customers',),
        ('customers', 'reconcile'),
        errors.UNMATCHED_DETAIL,
        ('code',),
        ((0, 1), (0, 1)),
    )
    # The isolated details must not produce additional read-only errors ...
    assert not any(
        error.document_path in (('customers', 0, 'source_no'),
                                 ('customers', 1, 'source_no'))
        for error in validator._errors
    )
    # ... while the remaining detail is still judged and its renumber reported.
    assert_has_error(
        validator._errors,
        ('customers', 2, 'source_no'),
        ('customers', 'schema', 'schema', 'source_no', 'readonly'),
        errors.READONLY_VALUE_CHANGED,
        True,
        (300, 301),
    )


def test_omitted_readonly_column_is_not_a_change():
    validator = make_customer_validator(
        source_no={'type': 'integer', 'readonly': True, 'default': 100}
    )
    original = {'customers': [{'code': 'A', 'source_no': 100, 'name': 'alpha'}]}
    # The patch does not carry the read-only column at all; the default
    # writes it back during normalization, which must not count as a change.
    patch = {'customers': [{'code': 'A', 'name': 'alpha'}]}
    assert validator.validate(patch, original=original)


def test_stored_value_is_not_recomputed_under_new_rules():
    # The new document coerces strings to integers, but the stored snapshot
    # already holds an integer. A genuine renumber must not be masked.
    validator = Validator(
        {'id': {'readonly': True, 'coerce': int}, 'name': {'type': 'string'}}
    )
    original = {'id': 5, 'name': 'x'}
    assert not validator.validate({'id': '9', 'name': 'x'}, original=original)
    assert_has_error(
        validator._errors,
        ('id',),
        ('id', 'readonly'),
        errors.READONLY_VALUE_CHANGED,
        True,
        (5, 9),
    )


def test_top_level_readonly_accepted_when_unchanged():
    validator = Validator(
        {'id': {'readonly': True}, 'name': {'type': 'string'}}
    )
    assert validator.validate(
        {'id': 5, 'name': 'x'}, original={'id': 5, 'name': 'x'}
    )


def test_top_level_readonly_change_is_reported():
    validator = Validator(
        {'id': {'readonly': True}, 'name': {'type': 'string'}}
    )
    assert not validator.validate(
        {'id': 6, 'name': 'x'}, original={'id': 5, 'name': 'x'}
    )
    assert_has_error(
        validator._errors,
        ('id',),
        ('id', 'readonly'),
        errors.READONLY_VALUE_CHANGED,
        True,
        (5, 6),
    )


def test_without_original_legacy_readonly_behavior_is_kept():
    validator = Validator({'id': {'readonly': True}})
    assert not validator.validate({'id': 1})
    assert validator._errors[0].code == errors.READONLY_FIELD.code


def test_original_ignored_without_normalization():
    validator = Validator({'id': {'readonly': True}})
    assert not validator.validate(
        {'id': 1}, original={'id': 1}, normalize=False
    )
    assert validator._errors[0].code == errors.READONLY_FIELD.code


def test_composite_business_key_and_nested_readonly():
    schema = {
        'orders': {
            'type': 'list',
            'schema': {
                'type': 'dict',
                'reconcile': ['tenant', 'seq'],
                'schema': {
                    'tenant': {'type': 'string'},
                    'seq': {'type': 'integer'},
                    'ref': {'readonly': True},
                    'meta': {
                        'type': 'dict',
                        'schema': {'tag': {'readonly': True}},
                    },
                },
            },
        }
    }
    validator = Validator(schema)
    original = {
        'orders': [
            {'tenant': 't1', 'seq': 1, 'ref': 'r1', 'meta': {'tag': 'g'}},
            {'tenant': 't1', 'seq': 2, 'ref': 'r2', 'meta': {'tag': 'h'}},
        ]
    }
    reordered = {
        'orders': [
            {'tenant': 't1', 'seq': 2, 'ref': 'r2', 'meta': {'tag': 'h'}},
            {'tenant': 't1', 'seq': 1, 'ref': 'r1', 'meta': {'tag': 'g'}},
        ]
    }
    assert validator.validate(reordered, original=original)
    assert validator.reconciliation[('orders',)]['key'] == ('tenant', 'seq')

    changed_nested = {
        'orders': [
            {'tenant': 't1', 'seq': 2, 'ref': 'r2', 'meta': {'tag': 'h'}},
            {'tenant': 't1', 'seq': 1, 'ref': 'r1', 'meta': {'tag': 'Z'}},
        ]
    }
    assert not validator.validate(changed_nested, original=original)
    assert_has_error(
        validator._errors,
        ('orders', 1, 'meta', 'tag'),
        (
            'orders', 'schema', 'schema', 'meta', 'schema', 'tag', 'readonly'
        ),
        errors.READONLY_VALUE_CHANGED,
        True,
        ('g', 'Z'),
    )
