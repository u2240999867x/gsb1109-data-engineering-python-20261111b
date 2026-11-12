"Test sortedcollections.recipes"

import pickle

import pytest

from sortedcollections import (
    IndexableDict,
    IndexableSet,
    SegmentList,
    replace_column_slices,
)


def test_index_dict():
    mapping = IndexableDict(enumerate(range(10)))
    iloc = mapping.keys()
    for value in range(10):
        assert iloc[value] == value


def test_index_set():
    set_values = IndexableSet(range(10))
    for index in range(10):
        assert set_values[index] == index


def test_index_set_pickle():
    set_values1 = IndexableSet(range(10))
    data = pickle.dumps(set_values1)
    set_values2 = pickle.loads(data)
    assert set_values1 == set_values2


def test_segment_list():
    values = [5, 1, 3, 2, 4, 8, 6, 7, 9, 0]
    sl = SegmentList(values)
    assert list(sl) == values
    sl.sort()
    assert list(sl) == list(range(10))
    sl.reverse()
    assert list(sl) == list(reversed(range(10)))
    sl.reverse()
    sl.append(10)
    assert list(sl) == list(range(11))
    sl.extend(range(11, 15))
    assert list(sl) == list(range(15))
    del sl[5:]
    assert list(sl) == list(range(5))
    sl[2] = 'c'
    sl.insert(3, 'd')
    sl.insert(6, 'e')
    assert list(sl) == [0, 1, 'c', 'd', 3, 4, 'e']


def test_segment_list_bisect():
    sl = SegmentList()
    with pytest.raises(NotImplementedError):
        sl.bisect(0)


def test_segment_list_setitem_slice():
    sl = SegmentList()
    with pytest.raises(NotImplementedError):
        sl[:] = [0]


def test_replace_column_slices_basic():
    col_a = SegmentList(range(10))
    col_b = SegmentList(range(100, 110))
    mapping = replace_column_slices([col_a, col_b], slice(2, 5), [[20, 21], [30, 31]])
    assert list(col_a) == [0, 1, 20, 21, 5, 6, 7, 8, 9]
    assert list(col_b) == [100, 101, 30, 31, 105, 106, 107, 108, 109]
    assert mapping == {
        0: 0, 1: 1, 2: None, 3: None, 4: None,
        5: 4, 6: 5, 7: 6, 8: 7, 9: 8,
    }


def test_replace_column_slices_growth_mapping():
    col = SegmentList(range(5))
    mapping = replace_column_slices([col], slice(1, 2), [[10, 11, 12]])
    assert list(col) == [0, 10, 11, 12, 2, 3, 4]
    assert mapping == {0: 0, 1: None, 2: 4, 3: 5, 4: 6}


def test_replace_column_slices_swap_columns():
    col_a = SegmentList(range(5))
    col_b = SegmentList(range(100, 105))
    mapping = replace_column_slices([col_a, col_b], slice(None), [col_b, col_a])
    assert list(col_a) == list(range(100, 105))
    assert list(col_b) == list(range(5))
    assert mapping == {row: row for row in range(5)}


def test_replace_column_slices_generator_snapshot():
    col = SegmentList(range(10))
    gen = (col[index] for index in range(4, 8))
    mapping = replace_column_slices([col], slice(2, 6), [gen])
    assert list(col) == [0, 1, 4, 5, 6, 7, 6, 7, 8, 9]
    assert mapping == {row: row for row in range(10)}


def test_replace_column_slices_negative_stride():
    col_a = SegmentList(range(10))
    col_b = SegmentList(range(100, 110))
    mapping = replace_column_slices(
        [col_a, col_b],
        slice(8, 2, -2),
        [[80, 60, 40], (value for value in [800, 600, 400])],
    )
    assert list(col_a) == [0, 1, 2, 3, 40, 5, 60, 7, 80, 9]
    assert list(col_b) == [100, 101, 102, 103, 400, 105, 600, 107, 800, 109]
    assert mapping == {row: row for row in range(10)}


def test_replace_column_slices_count_mismatch_atomic():
    col_a = SegmentList(range(5))
    col_b = SegmentList(range(10, 15))
    with pytest.raises(ValueError):
        replace_column_slices([col_a, col_b], slice(1, 3), [[1, 2], [1, 2, 3]])
    assert list(col_a) == list(range(5))
    assert list(col_b) == list(range(10, 15))


def test_replace_column_slices_generator_failure_atomic():
    col_a = SegmentList(range(5))
    col_b = SegmentList(range(10, 15))

    def bad_gen():
        yield 1
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError):
        replace_column_slices([col_a, col_b], slice(1, 3), [[7, 8], bad_gen()])
    assert list(col_a) == list(range(5))
    assert list(col_b) == list(range(10, 15))


def test_replace_column_slices_extended_mismatch_atomic():
    col_a = SegmentList(range(6))
    col_b = SegmentList(range(10, 16))
    with pytest.raises(ValueError):
        replace_column_slices([col_a, col_b], slice(0, 6, 2), [[1, 2, 3], [1, 2]])
    assert list(col_a) == list(range(6))
    assert list(col_b) == list(range(10, 16))


def test_replace_column_slices_unaligned_columns():
    col_a = SegmentList(range(3))
    col_b = SegmentList(range(4))
    with pytest.raises(ValueError):
        replace_column_slices([col_a, col_b], slice(0, 1), [[0], [0]])
    assert list(col_a) == list(range(3))
    assert list(col_b) == list(range(4))


def test_replace_column_slices_segment_locality():
    col = SegmentList(range(10000))
    before = list(col._lists)
    pos_lo, _ = col._pos(5000)
    pos_hi, _ = col._pos(5010)
    replace_column_slices([col], slice(5000, 5010), [[-1] * 5])
    after = col._lists
    head = before[:pos_lo]
    tail = before[pos_hi + 1:]
    assert all(after[pos] is segment for pos, segment in enumerate(head))
    offset = len(after) - len(tail)
    assert all(after[offset + pos] is segment for pos, segment in enumerate(tail))
    assert list(col) == list(range(5000)) + [-1] * 5 + list(range(5010, 10000))
