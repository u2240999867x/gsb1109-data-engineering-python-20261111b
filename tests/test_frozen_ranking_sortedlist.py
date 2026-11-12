import pytest

from sortedcontainers import (
    FrozenRanking,
    RankConflictError,
    SortedKeyList,
    SortedList,
    StaleRankError,
)


def test_freeze_records_version_and_rows():
    slt = SortedList(range(10))
    ranking = slt.freeze(token='ranking-1')

    assert isinstance(ranking, FrozenRanking)
    assert ranking.token == 'ranking-1'
    assert ranking.version == slt.version
    assert len(ranking) == 10
    assert list(ranking) == list(range(10))
    assert ranking[0] == 0
    assert ranking[9] == 9


def test_frozen_rows_are_independent_of_live_mutations():
    slt = SortedList(range(5))
    ranking = slt.freeze()

    slt.add(-1)
    slt.add(5)
    slt.remove(2)

    assert len(ranking) == 5
    assert list(ranking) == [0, 1, 2, 3, 4]


def test_negative_old_rank_is_relative_to_frozen_list():
    slt = SortedList(range(6))
    ranking = slt.freeze()

    assert ranking[-1] == 5
    assert ranking[-6] == 0

    current_rank, value = ranking.locate(-2)
    assert value == 4
    assert slt[current_rank] is slt[4]

    with pytest.raises(IndexError):
        ranking.locate(-7)


def test_old_rank_out_of_range():
    ranking = SortedList(range(3)).freeze()

    with pytest.raises(IndexError):
        ranking.locate(3)

    with pytest.raises(IndexError):
        ranking.quarantine(-4)


def test_old_rank_must_be_integer():
    ranking = SortedList(range(3)).freeze()

    with pytest.raises(TypeError):
        ranking.locate(1.0)


def test_insertions_elsewhere_do_not_drift_target():
    slt = SortedList(range(10))
    ranking = slt.freeze()

    slt.update([-5, -4, 100, 200])

    current_rank, value = ranking.locate(3)
    assert value == 3
    assert current_rank == 5
    assert slt[current_rank] == 3


def test_removals_elsewhere_do_not_drift_target():
    slt = SortedList(range(10))
    ranking = slt.freeze()

    slt.remove(0)
    slt.remove(1)

    current_rank, value = ranking.locate(4)
    assert value == 4
    assert current_rank == 2


def test_quarantine_removes_located_row_and_returns_receipt():
    slt = SortedList(range(10))
    ranking = slt.freeze(token='batch-42')

    slt.add(-1)
    receipt = ranking.quarantine(4)

    assert receipt['token'] == 'batch-42'
    assert receipt['old_rank'] == 4
    assert receipt['current_rank'] == 5
    assert receipt['value'] == 4
    assert receipt['frozen_version'] == ranking.version
    assert receipt['action'] == 'quarantined'
    assert 4 not in slt
    assert len(slt) == 10
    slt._check()


def test_duplicate_quarantine_is_idempotent_with_receipt():
    slt = SortedList(range(6))
    ranking = slt.freeze()

    first = ranking.quarantine(2)
    size = len(slt)
    second = ranking.quarantine(2)

    assert first['action'] == 'quarantined'
    assert second['action'] == 'duplicate'
    assert second['old_rank'] == 2
    assert second['value'] == 2
    assert second['token'] == first['token']
    assert len(slt) == size
    assert ranking.quarantined == {2: first}


def test_removed_row_is_stale():
    slt = SortedList(range(5))
    ranking = slt.freeze()

    slt.remove(2)

    with pytest.raises(StaleRankError):
        ranking.quarantine(2)

    assert len(slt) == 4


def test_mutated_row_conflicts():
    class Row:
        def __init__(self, key, name):
            self.key = key
            self.name = name

        def __lt__(self, other):
            return self.key < other.key

        def __eq__(self, other):
            return (
                isinstance(other, Row)
                and self.key == other.key
                and self.name == other.name
            )

    rows = [Row(index, f'name-{index}') for index in range(5)]
    slt = SortedList(rows)
    ranking = slt.freeze()

    rows[2].name = 'renamed'

    with pytest.raises(RankConflictError):
        ranking.locate(2)

    assert len(slt) == 5


def test_rank_conflict_error_is_stale_rank_error():
    assert issubclass(RankConflictError, StaleRankError)


def test_quarantine_all_validates_before_any_deletion():
    slt = SortedList(range(8))
    ranking = slt.freeze()

    slt.add(-9)
    slt.remove(5)
    snapshot = list(slt)

    with pytest.raises(StaleRankError):
        ranking.quarantine_all([1, 5, 2])

    assert list(slt) == snapshot
    assert len(slt) == 8
    slt._check()


def test_quarantine_all_stale_rank_leaves_list_intact():
    slt = SortedList(range(6))
    ranking = slt.freeze()

    slt.remove(3)
    slt.add(30)

    with pytest.raises(StaleRankError):
        ranking.quarantine_all([0, 3, 5])

    assert list(slt) == [0, 1, 2, 4, 5, 30]
    slt._check()


def test_quarantine_all_one_receipt_per_request():
    slt = SortedList(range(20))
    ranking = slt.freeze()

    slt.add(-1)
    receipts = ranking.quarantine_all([0, 3, 0, -1])

    assert [receipt['action'] for receipt in receipts] == [
        'quarantined',
        'quarantined',
        'duplicate',
        'quarantined',
    ]
    assert [receipt['old_rank'] for receipt in receipts] == [0, 3, 0, 19]
    assert [receipt['current_rank'] for receipt in receipts] == [1, 4, 1, 20]
    assert len(slt) == 18
    assert 0 not in slt and 3 not in slt and 19 not in slt
    slt._check()


def test_quarantine_all_repeated_after_fact_all_duplicates():
    slt = SortedList(range(10))
    ranking = slt.freeze()

    ranking.quarantine_all([1, 4])
    size = len(slt)
    receipts = ranking.quarantine_all([1, 4])

    assert all(receipt['action'] == 'duplicate' for receipt in receipts)
    assert len(slt) == size


def test_version_increments_on_every_mutation():
    slt = SortedList()
    initial = slt.version

    slt.add(1)
    assert slt.version == initial + 1

    slt.update([2, 3])
    assert slt.version == initial + 2

    slt.remove(2)
    assert slt.version == initial + 3

    slt.clear()
    assert slt.version == initial + 4


def test_sorted_key_list_freeze():
    skl = SortedKeyList(range(6), key=lambda value: -value)
    ranking = skl.freeze()

    skl.add(2)
    current_rank, value = ranking.locate(0)

    assert value == 5
    assert skl[current_rank] == 5

    receipt = ranking.quarantine(-1)
    assert receipt['value'] == 0
    skl._check()


def test_quarantine_with_load_factor_reshape():
    slt = SortedList(range(40))
    slt._reset(8)
    ranking = slt.freeze()

    for value in range(-10, 0):
        slt.add(value)

    receipt = ranking.quarantine_all([0, 10, 20, 39])
    assert len(slt) == 46
    slt._check()
    assert all(item['action'] == 'quarantined' for item in receipt)


def test_repr_contains_token_and_version():
    ranking = SortedList(range(3)).freeze(token='abc')
    text = repr(ranking)

    assert 'FrozenRanking' in text
    assert 'abc' in text
