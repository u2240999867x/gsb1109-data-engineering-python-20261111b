from datetime import datetime

from product_change_stream import (
    ChangeStreamValidator,
    Conflict,
    validate_change_stream,
)


def dt(text):
    return datetime.fromisoformat(text)


def upsert(event_no, row_no, external_id, start, end=None, source='erp'):
    return {
        'event_no': event_no,
        'row_no': row_no,
        'op': 'upsert',
        'source': source,
        'external_id': external_id,
        'valid_from': start,
        'valid_to': end,
    }


def terminate(event_no, row_no, external_id, at, source='erp'):
    return {
        'event_no': event_no,
        'row_no': row_no,
        'op': 'terminate',
        'source': source,
        'external_id': external_id,
        'terminated_at': at,
    }


def replace(event_no, row_no, target, payload):
    return {
        'event_no': event_no,
        'row_no': row_no,
        'op': 'replace',
        'replaces_event_no': target,
        'payload': payload,
    }


def payload_upsert(external_id, start, end=None, source='erp'):
    return {
        'op': 'upsert',
        'source': source,
        'external_id': external_id,
        'valid_from': start,
        'valid_to': end,
    }


def test_disjoint_periods_of_same_identity_do_not_conflict():
    report = validate_change_stream(
        [
            upsert(1, 10, '1001', '2026-01-01', '2026-02-01'),
            upsert(2, 11, 1001, '2026-02-01', '2026-03-01'),
            upsert(3, 12, 1001, '2026-04-01'),
        ]
    )
    assert report.conflicts == []
    assert report.introduced == []
    assert report.quarantined == []


def test_overlap_reports_true_segment_and_original_rows():
    report = validate_change_stream(
        [
            upsert(1, 10, 1001, '2026-01-01', '2026-03-01'),
            upsert(2, 11, 1001, '2026-02-01', '2026-04-01'),
            upsert(3, 12, 1002, '2026-02-01', '2026-04-01'),
        ]
    )
    expected = Conflict(
        source='erp',
        external_id=1001,
        start=dt('2026-02-01'),
        end=dt('2026-03-01'),
        row_nos=(10, 11),
    )
    assert report.conflicts == [expected]
    assert report.introduced == [expected]
    assert report.resolved == []


def test_partial_overlaps_split_into_true_segments():
    report = validate_change_stream(
        [
            upsert(1, 10, 1001, '2026-01-01', '2026-05-01'),
            upsert(2, 11, 1001, '2026-03-01', '2026-07-01'),
            upsert(3, 12, 1001, '2026-04-01', '2026-06-01'),
        ]
    )
    assert report.conflicts == [
        Conflict('erp', 1001, dt('2026-03-01'), dt('2026-04-01'), (10, 11)),
        Conflict('erp', 1001, dt('2026-04-01'), dt('2026-05-01'), (10, 11, 12)),
        Conflict('erp', 1001, dt('2026-05-01'), dt('2026-06-01'), (11, 12)),
    ]


def test_open_ended_overlap_reports_open_segment():
    report = validate_change_stream(
        [
            upsert(1, 10, 1001, '2026-01-01', '2026-06-01'),
            upsert(2, 11, 1001, '2026-03-01'),
        ]
    )
    assert report.conflicts == [
        Conflict('erp', 1001, dt('2026-03-01'), dt('2026-06-01'), (10, 11))
    ]


def test_terminate_closes_old_interval_and_resolves_conflict():
    validator = ChangeStreamValidator()
    validator.apply(
        [
            upsert(1, 10, 1001, '2026-01-01', '2026-06-01'),
            upsert(2, 11, 1001, '2026-03-01'),
        ]
    )
    report = validator.apply([terminate(3, 12, 1001, '2026-02-01')])
    assert report.conflicts == []
    assert report.introduced == []
    assert report.resolved == [
        Conflict('erp', 1001, dt('2026-03-01'), dt('2026-06-01'), (10, 11))
    ]


def test_replace_can_resolve_old_conflict():
    validator = ChangeStreamValidator()
    first = validator.apply(
        [
            upsert(1, 10, 1001, '2026-01-01', '2026-06-01'),
            upsert(2, 11, 1001, '2026-03-01', '2026-04-01'),
        ]
    )
    assert len(first.introduced) == 1
    report = validator.apply(
        [replace(3, 12, 2, payload_upsert(1001, '2026-07-01', '2026-08-01'))]
    )
    assert report.conflicts == []
    assert report.introduced == []
    assert report.resolved == first.conflicts


def test_replace_can_introduce_new_conflict():
    validator = ChangeStreamValidator()
    validator.apply(
        [
            upsert(1, 10, 1001, '2026-01-01', '2026-06-01'),
            upsert(2, 11, 1001, '2026-07-01', '2026-08-01'),
        ]
    )
    report = validator.apply(
        [replace(3, 12, 2, payload_upsert(1001, '2026-05-01', '2026-08-01'))]
    )
    assert report.resolved == []
    assert report.introduced == [
        Conflict('erp', 1001, dt('2026-05-01'), dt('2026-06-01'), (10, 12))
    ]
    assert report.conflicts == report.introduced


def test_replace_of_a_replace_follows_the_chain():
    validator = ChangeStreamValidator()
    validator.apply([upsert(1, 10, 1001, '2026-01-01', '2026-06-01')])
    validator.apply(
        [replace(2, 11, 1, payload_upsert(1001, '2026-01-01', '2026-03-01'))]
    )
    assert validator.conflicts == []
    report = validator.apply(
        [replace(3, 12, 1, payload_upsert(1001, '2026-01-01'))]
    )
    assert report.conflicts == []
    follow_up = validator.apply([upsert(4, 13, 1001, '2026-05-01', '2026-07-01')])
    assert follow_up.introduced == [
        Conflict('erp', 1001, dt('2026-05-01'), dt('2026-07-01'), (12, 13))
    ]


def test_processing_is_idempotent_by_event_no():
    validator = ChangeStreamValidator()
    rows = [
        upsert(1, 10, 1001, '2026-01-01', '2026-06-01'),
        upsert(2, 11, 1001, '2026-03-01'),
    ]
    first = validator.apply(rows)
    second = validator.apply(rows)
    assert len(first.introduced) == 1
    assert second.introduced == []
    assert second.resolved == []
    assert sorted(second.duplicates) == [1, 2]
    assert validator.conflicts == first.conflicts


def test_identity_conversion_failures_are_quarantined_apart():
    report = validate_change_stream(
        [
            upsert(1, 10, 'not-an-int', '2026-01-01', '2026-06-01'),
            upsert(2, 11, 1001, '2026-01-01', '2026-06-01', source=''),
            upsert(3, 12, 1001, '2026-01-01', '2026-06-01'),
            {'row_no': 13, 'op': 'upsert'},
        ]
    )
    assert [q.row_no for q in report.quarantined] == [10, 11, 13]
    assert 'external_id' in report.quarantined[0].errors
    assert 'source' in report.quarantined[1].errors
    assert report.conflicts == []


def test_invalid_validity_window_is_quarantined():
    report = validate_change_stream(
        [upsert(1, 10, 1001, '2026-06-01', '2026-01-01')]
    )
    assert [q.row_no for q in report.quarantined] == [10]
    assert 'valid_to' in report.quarantined[0].errors


def test_replay_on_empty_state_matches_one_pass_validation():
    rows = [
        upsert(1, 10, 1001, '2026-01-01', '2026-06-01'),
        upsert(2, 11, 1001, '2026-03-01', '2026-09-01'),
        terminate(3, 12, 1001, '2026-05-01'),
        replace(4, 13, 2, payload_upsert(1001, '2026-07-01', '2026-08-01')),
        upsert(5, 14, 1001, '2026-07-15', '2026-07-20'),
        upsert(6, 15, 'bad-id', '2026-01-01'),
    ]
    validator = ChangeStreamValidator()
    for row in rows:
        validator.apply([row])

    one_pass = validate_change_stream(rows)
    replayed = validator.replay()

    assert validator.conflicts == one_pass.conflicts
    assert replayed.conflicts == one_pass.conflicts
    assert [q.row_no for q in one_pass.quarantined] == [15]
    assert one_pass.conflicts == [
        Conflict('erp', 1001, dt('2026-07-15'), dt('2026-07-20'), (13, 14))
    ]
