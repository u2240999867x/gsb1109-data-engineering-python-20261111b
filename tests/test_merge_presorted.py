"""Tests for SortedList.merge_presorted and SortedKeyList.merge_presorted.

The merge operation consumes a single pre-sorted partition (for example a
fact-table slice already ordered by the same business key) without sorting it
again, computes each incoming key exactly once, touches only the internal
blocks that overlap the incoming rows, and leaves the list untouched when the
partition is rejected.

"""

import itertools
import random

import pytest

from sortedcontainers import SortedKeyList, SortedList


def modulo3(value):
    return value % 3


class IncomparableRow:
    """Row carrying an ordering key but no value-level comparison support."""

    def __init__(self, key, tag):
        self.key = key
        self.tag = tag

    def __repr__(self):
        return f'Row({self.key!r}, {self.tag!r})'


def row_key(row):
    return row.key


# ---------------------------------------------------------------------------
# Sparse complement: a handful of new rows landing in a few adjacent blocks
# must not cause unrelated old blocks to be rebuilt (or re-keyed).
# ---------------------------------------------------------------------------


def test_sparse_complement_touches_only_adjacent_blocks():
    load = 8
    old_values = list(range(100))
    incoming = [9, 10, 17, 18]

    slt = SortedList(old_values)
    slt._reset(load)
    assert len(slt._lists) == 13

    block_ids_before = [id(block) for block in slt._lists]
    far_block = slt._lists[-1]
    far_contents = list(far_block)

    slt.merge_presorted(iter(incoming))
    slt._check()

    replaced = sum(id(block) not in set(block_ids_before) for block in slt._lists)
    assert replaced == 2
    assert slt._lists[-1] is far_block
    assert list(slt._lists[-1]) == far_contents
    assert list(slt) == sorted(old_values + incoming)


def test_sparse_complement_key_list_reuses_stored_keys():
    load = 8
    old_values = list(range(100))
    incoming = [9, 10, 17, 18]
    key_calls = []

    def counting_key(value):
        key_calls.append(value)
        return value

    skl = SortedKeyList(old_values, key=counting_key)
    skl._reset(load)
    key_calls.clear()

    block_ids_before = [id(block) for block in skl._lists]
    key_block_ids_before = [id(block) for block in skl._keys]

    skl.merge_presorted(iter(incoming))

    # Each incoming key is generated exactly once and no old row is re-keyed.
    assert key_calls == incoming
    skl._check()

    replaced_value_blocks = sum(
        id(block) not in set(block_ids_before) for block in skl._lists
    )
    replaced_key_blocks = sum(
        id(block) not in set(key_block_ids_before) for block in skl._keys
    )
    assert replaced_value_blocks == 2
    assert replaced_key_blocks == 2
    assert list(skl) == sorted(old_values + incoming)


def test_sparse_complement_block_count_is_bounded_under_splits():
    load = 4
    old_values = list(range(0, 40, 2))
    incoming = [1, 1, 1, 1, 1, 9]

    skl = SortedKeyList(old_values, key=lambda value: value)
    skl._reset(load)
    block_ids_before = {id(block) for block in skl._lists}
    far_block = skl._lists[-1]
    far_contents = list(far_block)

    skl.merge_presorted(iter(incoming))
    skl._check()

    replaced = sum(id(block) not in block_ids_before for block in skl._lists)
    assert replaced <= 3
    assert skl._lists[-1] is far_block
    assert list(skl._lists[-1]) == far_contents
    assert all(len(block) <= load * 2 for block in skl._lists)
    assert list(skl) == sorted(old_values + incoming)


# ---------------------------------------------------------------------------
# Exhaustive verification on small tables, including stable equal-key order.
# ---------------------------------------------------------------------------


def test_sorted_list_small_table_matches_sorted_and_add_reference():
    universe = list(range(7))

    for load in range(4, 7):
        for mask in range(1 << len(universe)):
            incoming = [v for v in universe if mask & (1 << v)]
            existing = [v for v in universe if not (mask & (1 << v))]

            slt = SortedList(existing)
            slt._reset(load)
            slt.merge_presorted(iter(incoming))
            slt._check()

            assert list(slt) == sorted(existing + incoming)

            reference = SortedList()
            for value in existing:
                reference.add(value)
            for value in incoming:
                reference.add(value)
            assert list(slt) == list(reference)


def test_sorted_key_list_small_table_matches_sorted_and_add_reference():
    universe = list(range(7))

    for load in range(4, 7):
        for mask in range(1 << len(universe)):
            incoming = sorted((v for v in universe if mask & (1 << v)), key=modulo3)
            existing = [v for v in universe if not (mask & (1 << v))]

            skl = SortedKeyList(existing, key=modulo3)
            skl._reset(load)
            skl.merge_presorted(iter(incoming))
            skl._check()

            assert list(skl) == sorted(existing + incoming, key=modulo3)

            reference = SortedKeyList(key=modulo3)
            for value in existing:
                reference.add(value)
            for value in incoming:
                reference.add(value)
            assert list(skl) == list(reference)


def test_stable_merge_keeps_old_rows_before_new_rows_on_equal_keys():
    old_rows = [
        IncomparableRow(1, 'old-1'),
        IncomparableRow(2, 'old-2'),
        IncomparableRow(1, 'old-3'),
        IncomparableRow(3, 'old-4'),
    ]
    incoming = [
        IncomparableRow(1, 'new-1'),
        IncomparableRow(1, 'new-2'),
        IncomparableRow(2, 'new-3'),
        IncomparableRow(3, 'new-4'),
    ]

    skl = SortedKeyList(old_rows, key=row_key)
    skl._reset(4)
    skl.merge_presorted(iter(incoming))
    skl._check()

    key_one_tags = [row.tag for row in skl if row.key == 1]
    assert key_one_tags == ['old-1', 'old-3', 'new-1', 'new-2']

    key_two_tags = [row.tag for row in skl if row.key == 2]
    assert key_two_tags == ['old-2', 'new-3']

    reference = SortedKeyList(key=row_key)
    for row in old_rows:
        reference.add(row)
    for row in incoming:
        reference.add(row)
    assert [row.tag for row in skl] == [row.tag for row in reference]


def test_merge_matches_update_on_random_small_tables():
    random.seed(20261113)

    for _ in range(200):
        existing = [random.randrange(20) for _ in range(random.randrange(20))]
        incoming = sorted(
            (random.randrange(20) for _ in range(random.randrange(10))),
            key=modulo3,
        )

        merged = SortedKeyList(existing, key=modulo3)
        updated = SortedKeyList(existing, key=modulo3)
        merged._reset(5)
        updated._reset(5)

        merged.merge_presorted(iter(incoming))
        updated.update(incoming)
        merged._check()
        updated._check()

        assert list(merged) == list(updated)


def test_merge_equivalent_to_sorted_concatenation_across_loads():
    old_values = list(range(0, 50, 2))
    incoming = list(range(1, 25, 2))

    for load in range(4, 12):
        skl = SortedKeyList(old_values, key=lambda value: value)
        skl._reset(load)
        skl.merge_presorted(iter(incoming))
        skl._check()
        assert list(skl) == sorted(old_values + incoming)
        assert len(skl) == len(old_values) + len(incoming)


# ---------------------------------------------------------------------------
# Failure atomicity: rejected partitions must not change visible state.
# ---------------------------------------------------------------------------


def snapshot(slt):
    snap = (
        list(slt),
        slt._len,
        [list(block) for block in slt._lists],
        list(slt._maxes),
        list(slt._index),
        slt._offset,
    )
    if isinstance(slt, SortedKeyList):
        return snap + ([list(block) for block in slt._keys],)
    return snap + (None,)


def assert_snapshot_unchanged(slt, snap):
    assert list(slt) == snap[0]
    assert slt._len == snap[1]
    assert [list(block) for block in slt._lists] == snap[2]
    assert list(slt._maxes) == snap[3]
    assert list(slt._index) == snap[4]
    assert slt._offset == snap[5]
    if isinstance(slt, SortedKeyList):
        assert [list(block) for block in slt._keys] == snap[6]
    slt._check()


def test_inversion_raises_valueerror_and_preserves_rows_ranks_and_groups():
    old_rows = [
        IncomparableRow(1, 'old-1'),
        IncomparableRow(2, 'old-2'),
        IncomparableRow(2, 'old-3'),
        IncomparableRow(4, 'old-4'),
    ]
    skl = SortedKeyList(old_rows, key=row_key)
    skl._reset(4)
    skl._build_index()
    snap = snapshot(skl)
    rank_before = skl.bisect_key_right(2)

    with pytest.raises(ValueError, match='ascending iterable'):
        skl.merge_presorted(
            iter(
                [
                    IncomparableRow(1, 'new-1'),
                    IncomparableRow(3, 'new-2'),
                    IncomparableRow(2, 'new-3'),
                ]
            )
        )

    assert_snapshot_unchanged(skl, snap)
    assert skl.bisect_key_right(2) == rank_before
    assert [row.tag for row in skl if row.key == 2] == ['old-2', 'old-3']


def test_inversion_in_sorted_list_preserves_state():
    slt = SortedList([1, 1, 3, 5, 7])
    slt._reset(4)
    slt._build_index()
    snap = snapshot(slt)
    rank_before = slt.bisect_right(1)

    with pytest.raises(ValueError, match='ascending iterable'):
        slt.merge_presorted(iter([2, 6, 4]))

    assert_snapshot_unchanged(slt, snap)
    assert slt.bisect_right(1) == rank_before


def test_type_incompatibility_raises_typeerror_and_preserves_state():
    skl = SortedKeyList([1, 2, 3], key=lambda value: value)
    skl._reset(4)
    skl._build_index()
    snap = snapshot(skl)

    with pytest.raises(TypeError):
        skl.merge_presorted(iter([2, 'incompatible']))

    assert_snapshot_unchanged(skl, snap)

    slt = SortedList([1, 2, 3])
    slt._reset(4)
    slt._build_index()
    base_snap = snapshot(slt)

    with pytest.raises(TypeError):
        slt.merge_presorted(iter([2, 'incompatible']))

    assert_snapshot_unchanged(slt, base_snap)


def test_cross_partition_type_incompatibility_aborts_before_mutation():
    # The incoming partition is internally homogeneous and ordered, so staging
    # succeeds; the type clash only appears when comparing with the stored
    # partition. It must still be caught before any block is touched.
    skl = SortedKeyList([1, 2, 3], key=lambda value: value)
    skl._reset(4)
    skl._build_index()
    snap = snapshot(skl)
    lists_before = [id(block) for block in skl._lists]
    keys_before = [id(block) for block in skl._keys]

    with pytest.raises(TypeError):
        skl.merge_presorted(iter(['a', 'b', 'c']))

    assert_snapshot_unchanged(skl, snap)
    assert [id(block) for block in skl._lists] == lists_before
    assert [id(block) for block in skl._keys] == keys_before

    slt = SortedList([1, 2, 3])
    slt._reset(4)
    slt._build_index()
    base_snap = snapshot(slt)
    base_lists_before = [id(block) for block in slt._lists]

    with pytest.raises(TypeError):
        slt.merge_presorted(iter(['a', 'b', 'c']))

    assert_snapshot_unchanged(slt, base_snap)
    assert [id(block) for block in slt._lists] == base_lists_before


def test_failing_key_function_aborts_before_any_mutation():
    old_rows = [
        IncomparableRow(1, 'old-1'),
        IncomparableRow(3, 'old-2'),
    ]

    def flaky_key(row):
        if row.tag == 'boom':
            raise RuntimeError('key failure')
        return row.key

    skl = SortedKeyList(old_rows, key=flaky_key)
    skl._reset(4)
    snap = snapshot(skl)

    with pytest.raises(RuntimeError, match='key failure'):
        skl.merge_presorted(
            iter(
                [
                    IncomparableRow(2, 'ok'),
                    IncomparableRow(4, 'boom'),
                ]
            )
        )

    assert_snapshot_unchanged(skl, snap)


def test_failure_against_empty_list_preserves_empty_state():
    slt = SortedList()
    with pytest.raises(ValueError, match='ascending iterable'):
        slt.merge_presorted(iter([2, 1]))
    assert len(slt) == 0
    assert slt._lists == [] and slt._maxes == []
    slt._check()

    def flaky_key(row):
        if row.tag == 'boom':
            raise RuntimeError('key failure')
        return row.key

    skl = SortedKeyList(key=flaky_key)
    with pytest.raises(RuntimeError, match='key failure'):
        skl.merge_presorted(
            iter(
                [
                    IncomparableRow(1, 'ok'),
                    IncomparableRow(2, 'boom'),
                ]
            )
        )
    assert len(skl) == 0
    assert skl._lists == [] and skl._keys == [] and skl._maxes == []
    skl._check()


def test_empty_iterable_is_a_noop_that_keeps_the_index():
    skl = SortedKeyList(range(20), key=modulo3)
    skl._reset(6)
    skl._build_index()
    index_before = list(skl._index)
    snap = snapshot(skl)

    skl.merge_presorted(iter(()))

    assert list(skl._index) == index_before
    assert_snapshot_unchanged(skl, snap)


# ---------------------------------------------------------------------------
# Key generation and edge cases.
# ---------------------------------------------------------------------------


def test_each_incoming_key_is_generated_exactly_once():
    calls = []

    def counting_key(value):
        calls.append(value)
        return modulo3(value)

    skl = SortedKeyList(range(0, 30, 3), key=counting_key)
    skl._reset(5)
    calls.clear()

    incoming = sorted(range(1, 22, 3), key=modulo3)
    skl.merge_presorted(iter(incoming))

    assert calls == incoming
    skl._check()
    assert list(skl) == sorted(list(range(0, 30, 3)) + incoming, key=modulo3)


def test_merge_into_empty_list_chunks_by_load_factor():
    incoming = list(range(25))

    slt = SortedList()
    slt._reset(7)
    slt.merge_presorted(iter(incoming))
    slt._check()
    assert list(slt) == incoming
    assert all(len(block) <= 14 for block in slt._lists)

    skl = SortedKeyList(key=modulo3)
    skl._reset(7)
    skl.merge_presorted(iter(sorted(incoming, key=modulo3)))
    skl._check()
    assert list(skl) == sorted(incoming, key=modulo3)


def test_merge_accepts_equal_consecutive_keys_and_values():
    skl = SortedKeyList([IncomparableRow(1, 'old')], key=row_key)
    skl.merge_presorted(
        iter(
            [
                IncomparableRow(1, 'a'),
                IncomparableRow(1, 'b'),
                IncomparableRow(2, 'c'),
            ]
        )
    )
    skl._check()
    assert [row.tag for row in skl] == ['old', 'a', 'b', 'c']

    slt = SortedList([1, 2])
    slt.merge_presorted(iter([2, 2, 3]))
    slt._check()
    assert list(slt) == [1, 2, 2, 2, 3]


def test_merge_tail_extends_last_block_and_splits_correctly():
    old_values = list(range(40))
    incoming = list(range(100, 115))

    for load in range(4, 10):
        slt = SortedList(old_values)
        slt._reset(load)
        slt.merge_presorted(iter(incoming))
        slt._check()
        assert list(slt) == old_values + incoming
        assert all(len(block) <= load * 2 for block in slt._lists)


def test_generator_input_is_consumed_once():
    consumed = []

    def source():
        for value in range(5):
            consumed.append(value)
            yield value

    slt = SortedList([-2, -1])
    slt._reset(4)
    slt.merge_presorted(source())
    slt._check()
    assert consumed == [0, 1, 2, 3, 4]
    assert list(slt) == [-2, -1, 0, 1, 2, 3, 4]


def test_repeated_merges_remain_consistent_with_one_shot_merge():
    chunks = [sorted(range(start, start + 6)) for start in range(0, 30, 6)]
    one_shot = SortedList(itertools.chain.from_iterable(chunks))

    incremental = SortedList()
    incremental._reset(6)
    for chunk in chunks:
        incremental.merge_presorted(iter(chunk))
        incremental._check()

    assert list(incremental) == list(one_shot)
