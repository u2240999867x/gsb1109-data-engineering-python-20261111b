# -*- coding: utf-8 -*-

from datetime import date, datetime

import pytest

import cerberus
from cerberus import (
    BatchGate,
    ReferenceCheckingValidator,
    ReferenceEntry,
    ReferenceError,
    ReferenceTable,
    recheck,
    verify,
)
from cerberus.tests import assert_fail, assert_success


def entry(version, start, until=None, code='SKU1', source='catalog'):
    return ReferenceEntry(source, code, version, start, until)


@pytest.fixture
def clean_table():
    return ReferenceTable(
        [
            entry('v1', date(2020, 1, 1), date(2020, 6, 1)),
            entry('v2', date(2020, 6, 1), None),
        ]
    )


@pytest.fixture
def batch_lines():
    return [
        {'id': 'L1', 'source': 'catalog', 'code': 'SKU1', 'at': date(2020, 2, 1)},
        {'id': 'L2', 'source': 'catalog', 'code': 'SKU1', 'at': date(2020, 4, 1)},
        {'id': 'L3', 'source': 'catalog', 'code': 'SKU1', 'at': date(2020, 7, 1)},
    ]


def test_half_open_interval_lookup(clean_table):
    # lower bound is included
    assert clean_table.lookup('catalog', 'SKU1', date(2020, 1, 1)).version == 'v1'
    # upper bound is excluded: the next version starts there
    assert clean_table.lookup('catalog', 'SKU1', date(2020, 6, 1)).version == 'v2'
    # open-ended interval stays effective
    assert clean_table.lookup('catalog', 'SKU1', date(2030, 1, 1)).version == 'v2'
    # points before the first version do not match anything
    assert clean_table.lookup('catalog', 'SKU1', date(2019, 12, 31)) is None


def test_lookup_works_with_datetime_points():
    table = ReferenceTable(
        [
            ReferenceEntry(
                'catalog',
                'SKU1',
                'v1',
                datetime(2020, 1, 1),
                datetime(2020, 6, 1),
            ),
            ReferenceEntry('catalog', 'SKU1', 'v2', datetime(2020, 6, 1), None),
        ]
    )
    point = datetime(2020, 6, 1, 0, 0, 0)
    assert table.lookup('catalog', 'SKU1', point).version == 'v2'


def test_mixing_point_and_bound_types_is_rejected(clean_table):
    with pytest.raises(ReferenceError):
        clean_table.lookup('catalog', 'SKU1', datetime(2020, 3, 1))


def test_audit_clean_for_adjacent_intervals(clean_table):
    assert clean_table.audit() == []


def test_audit_reports_gap():
    table = ReferenceTable(
        [
            entry('v1', date(2020, 1, 1), date(2020, 2, 1)),
            entry('v2', date(2020, 3, 1), None),
        ]
    )
    issues = table.audit()
    assert len(issues) == 1
    issue = issues[0]
    assert issue.kind == 'gap'
    assert issue.start == date(2020, 2, 1)
    assert issue.end == date(2020, 3, 1)
    assert issue.versions == ('v1', 'v2')
    assert table.gap_at('catalog', 'SKU1', date(2020, 2, 15)) == issue
    assert table.gap_at('catalog', 'SKU1', date(2020, 1, 15)) is None


def test_audit_reports_overlap():
    table = ReferenceTable(
        [
            entry('v1', date(2020, 1, 1), date(2020, 3, 1)),
            entry('v2', date(2020, 2, 1), None),
        ]
    )
    issues = table.audit()
    assert len(issues) == 1
    issue = issues[0]
    assert issue.kind == 'overlap'
    assert issue.start == date(2020, 2, 1)
    assert issue.end == date(2020, 3, 1)
    assert issue.versions == ('v1', 'v2')


def test_audit_is_scoped_by_reference():
    table = ReferenceTable(
        [
            entry('v1', date(2020, 1, 1), date(2020, 2, 1)),
            entry('v2', date(2020, 3, 1), None),
            entry(
                'w1',
                date(2020, 1, 1),
                None,
                code='SKU2',
            ),
        ]
    )
    assert len(table.audit()) == 1
    assert table.audit('catalog', 'SKU2') == []


def test_malformed_entries_are_rejected():
    bad_entries = (
        ReferenceEntry('catalog', 'SKU1', 'v', date(2020, 2, 1), date(2020, 1, 1)),
        ReferenceEntry('catalog', 'SKU1', None, date(2020, 1, 1), None),
        ReferenceEntry(None, 'SKU1', 'v', date(2020, 1, 1), None),
        ReferenceEntry('catalog', None, 'v', date(2020, 1, 1), None),
        ReferenceEntry('catalog', 'SKU1', 'v', None, None),
    )
    for bad in bad_entries:
        with pytest.raises(ReferenceError):
            ReferenceTable([bad])


def test_batch_gate_freezes_and_isolates_late_versions(clean_table, batch_lines):
    gate = BatchGate(clean_table, at=date(2020, 11, 1))

    verdicts = verify(batch_lines, gate)
    assert [verdict.hit for verdict in verdicts] == ['v1', 'v1', 'v2']

    # a version that arrives while the batch is running must stay invisible
    clean_table.add(entry('v9', date(2019, 1, 1), None, code='SKU2'))
    late_line = [
        {'id': 'L4', 'source': 'catalog', 'code': 'SKU2', 'at': date(2020, 1, 1)}
    ]
    assert verify(late_line, gate)[0].hit is None
    assert len(gate) == 2

    # the same version applies to the next batch via a new gate
    next_gate = BatchGate(clean_table, at=date(2020, 12, 1))
    assert verify(late_line, next_gate)[0].hit == 'v9'


def test_gate_cannot_be_frozen_twice(clean_table):
    gate = BatchGate(clean_table)
    with pytest.raises(ReferenceError):
        gate.freeze()


def test_verify_reports_gap_and_overlap():
    gap_table = ReferenceTable(
        [
            entry('v1', date(2020, 1, 1), date(2020, 2, 1)),
            entry('v2', date(2020, 3, 1), None),
        ]
    )
    gap_verdict = verify(
        [{'id': 'G', 'source': 'catalog', 'code': 'SKU1', 'at': date(2020, 2, 15)}],
        gap_table,
    )[0]
    assert gap_verdict.hit is None
    assert gap_verdict.ambiguity == 'gap'

    overlap_table = ReferenceTable(
        [
            entry('v1', date(2020, 1, 1), date(2020, 3, 1)),
            entry('v2', date(2020, 2, 1), None),
        ]
    )
    overlap_verdict = verify(
        [{'id': 'O', 'source': 'catalog', 'code': 'SKU1', 'at': date(2020, 2, 15)}],
        overlap_table,
    )[0]
    assert overlap_verdict.hit is None
    assert overlap_verdict.ambiguity == 'overlap'
    assert overlap_verdict.versions == ('v1', 'v2')


def test_recheck_split_returns_only_affected_lines(clean_table, batch_lines):
    gate = BatchGate(clean_table, at=date(2020, 11, 1))

    # retroactive correction splits the former v1 interval in two
    completed = ReferenceTable(
        [
            entry('v1a', date(2020, 1, 1), date(2020, 3, 1)),
            entry('v1b', date(2020, 3, 1), date(2020, 6, 1)),
            entry('v2', date(2020, 6, 1), None),
        ]
    )

    affected = recheck(batch_lines, gate, completed)
    assert [item.identifier for item in affected] == ['L1', 'L2']

    first, second = affected
    assert (first.old_hit, first.new_hit) == ('v1', 'v1a')
    assert (second.old_hit, second.new_hit) == ('v1', 'v1b')
    assert first.ambiguity is None and second.ambiguity is None

    # full verification against the completed table: every line outside the
    # affected list keeps the exact same verdict as against the frozen gate
    old_verdicts = {v.identifier: v for v in verify(batch_lines, gate)}
    new_verdicts = {v.identifier: v for v in verify(batch_lines, completed)}
    unaffected = set(new_verdicts) - {item.identifier for item in affected}
    for identifier in unaffected:
        assert (
            old_verdicts[identifier].hit == new_verdicts[identifier].hit
        )
        assert (
            old_verdicts[identifier].versions
            == new_verdicts[identifier].versions
        )
        assert (
            old_verdicts[identifier].ambiguity
            == new_verdicts[identifier].ambiguity
        )


def test_recheck_flags_gap_and_overlap_ambiguity(clean_table):
    gate = BatchGate(clean_table)
    line = [
        {'id': 'L1', 'source': 'catalog', 'code': 'SKU1', 'at': date(2020, 2, 15)}
    ]

    gap_table = ReferenceTable(
        [
            entry('v1', date(2020, 1, 1), date(2020, 2, 1)),
            entry('v2', date(2020, 3, 1), None),
        ]
    )
    gap_item = recheck(line, gate, gap_table)[0]
    assert gap_item.old_hit == 'v1'
    assert gap_item.new_hit is None
    assert gap_item.ambiguity == 'gap'

    overlap_table = ReferenceTable(
        [
            entry('v1', date(2020, 1, 1), date(2020, 3, 1)),
            entry('v2', date(2020, 2, 1), None),
        ]
    )
    overlap_item = recheck(line, gate, overlap_table)[0]
    assert overlap_item.old_hit == 'v1'
    assert overlap_item.new_hit is None
    assert overlap_item.new_versions == ('v1', 'v2')
    assert overlap_item.ambiguity == 'overlap'


def test_recheck_rejects_duplicate_identifiers(clean_table):
    gate = BatchGate(clean_table)
    lines = [
        {'id': 'L1', 'source': 'catalog', 'code': 'SKU1', 'at': date(2020, 2, 1)},
        {'id': 'L1', 'source': 'catalog', 'code': 'SKU1', 'at': date(2020, 7, 1)},
    ]
    with pytest.raises(ReferenceError):
        recheck(lines, gate, ReferenceTable(clean_table.entries()))


def test_recheck_returns_empty_list_when_nothing_changed(clean_table, batch_lines):
    gate = BatchGate(clean_table)
    rebuilt = ReferenceTable(clean_table.entries())
    assert recheck(batch_lines, gate, rebuilt) == []


def test_recheck_accepts_custom_field_names(clean_table):
    gate = BatchGate(clean_table)
    lines = [
        {
            'line_no': 7,
            'origin': 'catalog',
            'product': 'SKU1',
            'booked_on': date(2020, 2, 1),
        }
    ]
    completed = ReferenceTable(
        [
            entry('v1a', date(2020, 1, 1), date(2020, 3, 1)),
            entry('v1b', date(2020, 3, 1), date(2020, 6, 1)),
            entry('v2', date(2020, 6, 1), None),
        ]
    )
    affected = recheck(
        lines,
        gate,
        completed,
        source_field='origin',
        code_field='product',
        point_field='booked_on',
        id_field='line_no',
    )
    assert len(affected) == 1
    assert affected[0].identifier == 7
    assert (affected[0].old_hit, affected[0].new_hit) == ('v1', 'v1a')


def test_reference_rule_success_and_not_found(clean_table):
    schema = {
        'sku': {
            'type': 'string',
            'reference': {'source': 'catalog', 'at_field': 'issued_at'},
        },
        'issued_at': {'type': 'date'},
    }
    validator = ReferenceCheckingValidator(
        schema, reference_gate=BatchGate(clean_table)
    )
    assert_success(
        {'sku': 'SKU1', 'issued_at': date(2020, 3, 1)}, validator=validator
    )
    assert_fail(
        {'sku': 'SKU1', 'issued_at': date(2019, 1, 1)},
        validator=validator,
        error=(
            'sku',
            ('sku', 'reference'),
            cerberus.errors.REFERENCE_NOT_FOUND,
            {'source': 'catalog', 'at_field': 'issued_at'},
            (),
        ),
    )


def test_reference_rule_flags_gap_and_overlap():
    schema = {
        'sku': {
            'type': 'string',
            'reference': {'source': 'catalog', 'at_field': 'issued_at'},
        },
        'issued_at': {'type': 'date'},
    }
    gap_table = ReferenceTable(
        [
            entry('v1', date(2020, 1, 1), date(2020, 2, 1)),
            entry('v2', date(2020, 3, 1), None),
        ]
    )
    gap_validator = ReferenceCheckingValidator(
        schema, reference_gate=BatchGate(gap_table)
    )
    assert_fail(
        {'sku': 'SKU1', 'issued_at': date(2020, 2, 15)},
        validator=gap_validator,
        error=(
            'sku',
            ('sku', 'reference'),
            cerberus.errors.REFERENCE_AMBIGUOUS,
            {'source': 'catalog', 'at_field': 'issued_at'},
            ('gap between versions ' "('v1', 'v2') on [2020-02-01, 2020-03-01)",),
        ),
    )

    overlap_table = ReferenceTable(
        [
            entry('v1', date(2020, 1, 1), date(2020, 3, 1)),
            entry('v2', date(2020, 2, 1), None),
        ]
    )
    overlap_validator = ReferenceCheckingValidator(
        schema, reference_gate=BatchGate(overlap_table)
    )
    assert_fail(
        {'sku': 'SKU1', 'issued_at': date(2020, 2, 15)},
        validator=overlap_validator,
        error=(
            'sku',
            ('sku', 'reference'),
            cerberus.errors.REFERENCE_AMBIGUOUS,
            {'source': 'catalog', 'at_field': 'issued_at'},
            ("overlapping versions ('v1', 'v2')",),
        ),
    )


def test_reference_rule_with_literal_at_and_code_field(clean_table):
    schema = {
        'product': {
            'reference': {
                'source': 'catalog',
                'code_field': 'sku',
                'at': date(2020, 7, 1),
            }
        }
    }
    validator = ReferenceCheckingValidator(
        schema, reference_gate=BatchGate(clean_table), allow_unknown=True
    )
    assert_success({'product': 'anything', 'sku': 'SKU1'}, validator=validator)


def test_reference_rule_without_gate_reports_error():
    validator = ReferenceCheckingValidator(
        {
            'sku': {
                'reference': {'source': 'catalog', 'at_field': 'issued_at'}
            },
            'issued_at': {'type': 'date'},
        }
    )
    result = validator({'sku': 'SKU1', 'issued_at': date(2020, 1, 1)})
    assert result is False
    sku_errors = [error for error in validator._errors if error.field == 'sku']
    assert sku_errors[0].code == cerberus.errors.CUSTOM.code


def test_invalid_reference_constraint_raises_schema_error():
    with pytest.raises(cerberus.SchemaError):
        ReferenceCheckingValidator(
            {'sku': {'reference': {'at_field': 'issued_at'}}}
        )

    with pytest.raises(cerberus.SchemaError):
        ReferenceCheckingValidator(
            {
                'sku': {
                    'reference': {
                        'source': 'catalog',
                        'at': date(2020, 1, 1),
                        'at_field': 'issued_at',
                    }
                }
            }
        )


def test_recheck_accepts_generator_input(clean_table, batch_lines):
    gate = BatchGate(clean_table)
    completed = ReferenceTable(
        [
            entry('v1a', date(2020, 1, 1), date(2020, 3, 1)),
            entry('v1b', date(2020, 3, 1), date(2020, 6, 1)),
            entry('v2', date(2020, 6, 1), None),
        ]
    )
    affected = recheck((line for line in batch_lines), gate, completed)
    assert [item.identifier for item in affected] == ['L1', 'L2']


def test_entry_payload_is_kept_on_match():
    table = ReferenceTable(
        [
            ReferenceEntry(
                'catalog',
                'SKU1',
                'v1',
                date(2020, 1, 1),
                date(2020, 6, 1),
                {'price': 10},
            )
        ]
    )
    verdict = verify(
        [{'id': 'L1', 'source': 'catalog', 'code': 'SKU1', 'at': date(2020, 2, 1)}],
        table,
    )[0]
    assert verdict.entries[0].payload == {'price': 10}


def test_reference_rule_works_in_nested_schema(clean_table):
    schema = {
        'items': {
            'type': 'list',
            'schema': {
                'type': 'dict',
                'schema': {
                    'sku': {
                        'type': 'string',
                        'reference': {
                            'source': 'catalog',
                            'at_field': 'issued_at',
                        },
                    },
                    'issued_at': {'type': 'date'},
                },
            },
        }
    }
    validator = ReferenceCheckingValidator(
        schema, reference_gate=BatchGate(clean_table)
    )
    assert_success(
        {
            'items': [
                {'sku': 'SKU1', 'issued_at': date(2020, 3, 1)},
                {'sku': 'SKU1', 'issued_at': date(2020, 7, 1)},
            ]
        },
        validator=validator,
    )
    assert_fail(
        {'items': [{'sku': 'SKU1', 'issued_at': date(2019, 1, 1)}]},
        validator=validator,
    )
    assert (
        validator.document_error_tree['items'][0]['sku'].errors[0].code
        == cerberus.errors.REFERENCE_NOT_FOUND.code
    )
