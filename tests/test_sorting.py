import json
import math

import pytest

import pydash as _
from pydash.exceptions import StaleCursorError
from pydash.sorting import ColumnSortSpec, KeyedRecord, MergeStore, SortPolicy


parametrize = pytest.mark.parametrize


def make_policy(version="1", **overrides):
    spec = {
        "name": "qty",
        "direction": "asc",
        "null_position": "last",
        "type_zones": ("number", "text"),
    }
    spec.update(overrides)
    return SortPolicy([spec], version=version)


def merged_qtys(store):
    qtys = []
    for record in store.records():
        value = record.payload.get("qty")
        if isinstance(value, float) and math.isnan(value):
            value = None
        qtys.append(value)
    return qtys


@parametrize(
    "null_position,expected",
    [
        ("last", [1, 2, "ten", None, None, None]),
        ("first", [None, None, None, 1, 2, "ten"]),
    ],
)
def test_null_position_covers_none_nan_and_missing(null_position, expected):
    policy = make_policy(null_position=null_position)
    store = MergeStore(policy)
    store.ingest(
        "p1",
        [
            {"qty": None},
            {"qty": "ten"},
            {"qty": 2},
            {},
            {"qty": math.nan},
            {"qty": 1},
        ],
    )
    assert merged_qtys(store) == expected


def test_type_zones_order_is_configurable_per_column():
    rows = [{"qty": 1}, {"qty": "apple"}, {"qty": 2}, {"qty": "banana"}]

    number_first = MergeStore(make_policy(type_zones=("number", "text")))
    number_first.ingest("p1", rows)
    assert merged_qtys(number_first) == [1, 2, "apple", "banana"]

    text_first = MergeStore(make_policy(type_zones=("text", "number")))
    text_first.ingest("p1", rows)
    assert merged_qtys(text_first) == ["apple", "banana", 1, 2]


def test_uncovered_zone_raises_clear_error():
    policy = make_policy(type_zones=("number",))
    with pytest.raises(ValueError, match="not covered by type_zones"):
        policy.sort_key({"qty": "text"})


@parametrize(
    "direction,expected",
    [
        ("asc", [1, 2, 3]),
        ("desc", [3, 2, 1]),
    ],
)
def test_number_direction(direction, expected):
    store = MergeStore(make_policy(direction=direction))
    store.ingest("p1", [{"qty": 2}, {"qty": 3}, {"qty": 1}])
    assert merged_qtys(store) == expected


@parametrize(
    "direction,expected",
    [
        ("asc", ["ab", "abc", "b"]),
        ("desc", ["b", "abc", "ab"]),
    ],
)
def test_text_direction_handles_prefixes(direction, expected):
    store = MergeStore(make_policy(direction=direction))
    store.ingest("p1", [{"qty": "abc"}, {"qty": "b"}, {"qty": "ab"}])
    assert merged_qtys(store) == expected


def test_compare_detail_distinguishes_zone_value_and_sequence():
    policy = make_policy()
    key_number = policy.sort_key({"qty": 5})
    key_other_number = policy.sort_key({"qty": 6})
    key_text = policy.sort_key({"qty": "5"})
    key_equal_numeric = policy.sort_key({"qty": 5.0})

    result, reason = policy.compare_detail(key_number, 0, key_text, 1)
    assert (result, reason) == (-1, "zone")

    result, reason = policy.compare_detail(key_number, 0, key_other_number, 1)
    assert (result, reason) == (-1, "value")

    # Equal numeric values produce identical natural keys; the global
    # ingestion sequence breaks the tie.
    assert key_number == key_equal_numeric
    result, reason = policy.compare_detail(key_number, 0, key_equal_numeric, 1)
    assert (result, reason) == (-1, "sequence")

    result, reason = policy.compare_detail(key_number, 0, key_number, 0)
    assert (result, reason) == (0, "equal")


@parametrize("direction", ["asc", "desc"])
def test_ties_broken_by_original_ingest_seq_even_when_descending(direction):
    policy = make_policy(direction=direction)
    store = MergeStore(policy)
    store.ingest("p1", [{"qty": 3}, {"qty": 3}, {"qty": 9}])
    records = store.records()
    tied = [record.seq for record in records if record.payload["qty"] == 3]
    assert tied == [0, 1]
    expected_first = 3 if direction == "asc" else 9
    assert records[0].payload["qty"] == expected_first


def test_multi_column_policy_uses_later_columns_as_tiebreak():
    policy = SortPolicy(
        [
            {"name": "qty", "direction": "desc", "null_position": "last"},
            {"name": "label", "direction": "asc", "null_position": "first"},
        ]
    )
    store = MergeStore(policy)
    store.ingest(
        "p1",
        [
            {"qty": 2, "label": "b"},
            {"qty": 3, "label": "a"},
            {"qty": 2, "label": "a"},
            {"qty": 2},
        ],
    )
    assert [(r.payload.get("qty"), r.payload.get("label")) for r in store.records()] == [
        (3, "a"),
        (2, None),
        (2, "a"),
        (2, "b"),
    ]


def test_sort_keys_are_persistable():
    policy = make_policy(direction="desc")
    for value in (5, 2.5, "text", None, math.nan):
        key = policy.sort_key({"qty": value})
        assert policy.decode_key(json.loads(json.dumps(policy.encode_key(key)))) == key


def test_partitions_generate_keys_independently_and_merge_elsewhere():
    policy = make_policy()
    batches = {
        "pA": [{"qty": 1}, {"qty": "4"}, {"qty": None}],
        "pB": [{"qty": 2}, {"qty": 3}],
    }
    # Partition side: only the policy is needed to generate persistable keys.
    shipped = []
    seq = 0
    for partition, rows in batches.items():
        for row in rows:
            shipped.append(
                {
                    "partition": partition,
                    "seq": seq,
                    "key": json.loads(json.dumps(policy.encode_key(policy.sort_key(row)))),
                    "payload": row,
                }
            )
            seq += 1

    # Merger side (another process): only keys and global sequence numbers.
    store = MergeStore(policy)
    for record in shipped:
        store.ingest_keyed(record["partition"], record["key"], record["seq"], record["payload"])

    assert [record.seq for record in store.records()] == [0, 3, 4, 1, 2]


def test_store_roundtrip_through_json_preserves_merge_and_pagination():
    policy = make_policy()
    original = MergeStore(policy)
    original.ingest("pA", [{"qty": 1}, {"qty": 3}, {"qty": 5}])
    original.ingest("pB", [{"qty": 2}, {"qty": 4}, {"qty": 6}])

    restored = MergeStore.from_keyed(policy, json.loads(json.dumps(original.dump())))

    assert restored.records() == original.records()
    for store in (original, restored):
        first = store.page(4)
        second = store.page(4, cursor=first.cursor)
        assert [r.payload["qty"] for r in first.items] == [1, 2, 3, 4]
        assert [r.payload["qty"] for r in second.items] == [5, 6]
        assert second.has_more is False


def test_page_resume_uses_only_cursor_token():
    policy = make_policy()
    store = MergeStore(policy)
    store.ingest("p1", [{"qty": value} for value in (1, 2, 3, 4)])

    first = store.page(2)
    assert [r.payload["qty"] for r in first.items] == [1, 2]
    assert first.has_more is True

    token = json.loads(json.dumps(first.cursor))  # persist and reload the cursor
    second = store.page(2, cursor=token)
    assert [r.payload["qty"] for r in second.items] == [3, 4]
    assert second.has_more is False

    # Re-reading from a page boundary reproduces that page.
    again = store.page(2, cursor=second.start)
    assert [r.payload["qty"] for r in again.items] == [3, 4]


def test_late_record_marks_segment_for_redo_and_only_affected_pages():
    policy = make_policy()
    store = MergeStore(policy)
    store.ingest("p1", [{"qty": value} for value in (1, 2, 3, 4, 5, 6)])

    pages = [store.page(2)]
    while pages[-1].has_more:
        pages.append(store.page(2, cursor=pages[-1].cursor))
    boundaries = [page.start for page in pages]
    assert [r.payload["qty"] for page in pages for r in page.items] == [1, 2, 3, 4, 5, 6]

    # A late record arrives in the same partition, sorting inside page 3.
    store.ingest("p1", [{"qty": 4}])

    follow_up = store.page(2, cursor=pages[-1].cursor)
    assert follow_up.redo is True
    # The redo segment starts right where page 3 started.
    assert follow_up.redo_from == boundaries[2]

    # Only the pages at/after the redo segment are reported as affected.
    assert store.affected_pages(boundaries) == [boundaries[2]]

    # Re-emitting from the redo position yields the corrected segment only.
    redo_page = store.page(2, cursor=follow_up.redo_from)
    assert [r.payload["qty"] for r in redo_page.items] == [4, 5]
    tail = store.page(2, cursor=redo_page.cursor)
    assert [r.payload["qty"] for r in tail.items] == [6]
    assert tail.has_more is False

    store.acknowledge_redo()
    assert store.redo_from is None
    assert store.affected_pages(boundaries) == []
    assert store.page(2, cursor=tail.cursor).redo is False


def test_late_record_before_everything_redoes_from_the_start():
    policy = make_policy()
    store = MergeStore(policy)
    store.ingest("p1", [{"qty": 2}, {"qty": 3}])
    first = store.page(1)
    second = store.page(1, cursor=first.cursor)

    store.ingest("p1", [{"qty": 1}])

    assert store.redo_from is not None
    assert store.affected_pages([first.start, second.start]) == [first.start, second.start]
    redo = store.page(3, cursor=store.redo_from)
    assert [r.payload["qty"] for r in redo.items] == [1, 2, 3]


def test_record_arriving_after_high_water_is_not_late():
    policy = make_policy()
    store = MergeStore(policy)
    store.ingest("p1", [{"qty": 1}, {"qty": 2}])
    store.page(2)

    store.ingest("p1", [{"qty": 3}])

    assert store.redo_from is None
    assert store.page(1).redo is False


def test_stale_cursor_from_changed_policy_is_recognized():
    policy_v1 = make_policy()
    store_v1 = MergeStore(policy_v1)
    store_v1.ingest("p1", [{"qty": 1}, {"qty": 2}])
    cursor = store_v1.page(1).cursor

    # Changing any aspect of the policy (here the version) changes its
    # fingerprint, so cursors issued under the old policy are recognized.
    policy_v2 = make_policy(version="2")
    assert policy_v2.fingerprint != policy_v1.fingerprint
    assert policy_v2.check_cursor(cursor) is False
    assert policy_v1.check_cursor(cursor) is True

    store_v2 = MergeStore(policy_v2)
    store_v2.ingest("p1", [{"qty": 1}, {"qty": 2}])
    with pytest.raises(StaleCursorError):
        store_v2.page(1, cursor=cursor)

    store_v2.page(2)
    store_v2.ingest("p1", [{"qty": 0}])  # late record to arm the redo horizon
    with pytest.raises(StaleCursorError):
        store_v2.affected_pages([cursor])

    # A changed column specification is also a different policy.
    policy_v3 = make_policy(direction="desc")
    assert policy_v3.check_cursor(cursor) is False


def test_malformed_cursor_raises_value_error():
    policy = make_policy()
    with pytest.raises(ValueError, match="malformed cursor"):
        policy.decode_cursor("not json")
    with pytest.raises(ValueError, match="malformed cursor"):
        policy.decode_cursor(json.dumps({"p": policy.fingerprint}))


@parametrize(
    "spec,match",
    [
        ({"name": "qty", "direction": "up"}, "direction"),
        ({"name": "qty", "null_position": "middle"}, "null_position"),
        ({"name": "qty", "type_zones": ()}, "type_zones"),
        ({"name": "qty", "type_zones": ("null",)}, "reserved"),
        ({"name": "qty", "type_zones": ("number", "number")}, "duplicates"),
    ],
)
def test_column_spec_validation(spec, match):
    with pytest.raises(ValueError, match=match):
        ColumnSortSpec(**spec)


def test_policy_validation():
    with pytest.raises(ValueError, match="at least one column"):
        SortPolicy([])
    with pytest.raises(ValueError, match="unique"):
        SortPolicy([{"name": "qty"}, {"name": "qty"}])
    with pytest.raises(TypeError):
        SortPolicy(["qty"])


def test_policy_serialization_roundtrip():
    policy = SortPolicy(
        [
            {"name": "qty", "direction": "desc", "null_position": "first"},
            {"name": "label", "type_zones": ("text", "number")},
        ],
        version="2026-11",
    )
    restored = SortPolicy.from_dict(json.loads(json.dumps(policy.to_dict())))
    assert restored.fingerprint == policy.fingerprint


def test_keyed_record_serialization_roundtrip():
    record = KeyedRecord(key=((0, 5),), seq=7, partition="p1", payload={"qty": 5})
    assert KeyedRecord.from_dict(json.loads(json.dumps(record.to_dict()))) == record


def test_objects_as_rows():
    class Row:
        def __init__(self, qty):
            self.qty = qty

    policy = make_policy()
    store = MergeStore(policy)
    store.ingest("p1", [Row(2), Row(1)])
    assert [record.payload.qty for record in store.records()] == [1, 2]


def test_exported_from_top_level():
    assert _.SortPolicy is SortPolicy
    assert _.MergeStore is MergeStore
    assert _.ColumnSortSpec is ColumnSortSpec
    assert _.KeyedRecord is KeyedRecord
    assert _.Page is not None
    assert _.StaleCursorError is StaleCursorError
