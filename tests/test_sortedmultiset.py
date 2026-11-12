import math
import random
from collections import Counter

import pytest

from sortedcontainers import (
    NegativeCountError,
    SortedMultiset,
    SortedMultisetView,
)


def test_init_and_basic_queries():
    ms = SortedMultiset([(200, 5), (404, 2), (500, 3)])
    ms._check()

    assert len(ms) == 10
    assert ms.distinct_count() == 3
    assert list(ms) == [200, 404, 500]
    assert list(ms.items()) == [(200, 5), (404, 2), (500, 3)]
    assert list(ms.elements()) == [200] * 5 + [404] * 2 + [500] * 3

    assert 200 in ms
    assert 999 not in ms
    assert ms.multiplicity(404) == 2
    assert ms.count(404) == 2
    assert ms.multiplicity(999) == 0

    assert SortedMultiset().distinct_count() == 0
    assert len(SortedMultiset()) == 0

    mapping = SortedMultiset({1: 2, 3: 4})
    assert list(mapping.items()) == [(1, 2), (3, 4)]

    duplicate_keys = SortedMultiset([(1, 2), (1, 3), (2, 1)])
    assert list(duplicate_keys.items()) == [(1, 5), (2, 1)]


def test_from_values():
    ms = SortedMultiset.from_values([3, 1, 2, 3, 1, 3])
    assert list(ms.items()) == [(1, 2), (2, 1), (3, 3)]
    ms._check()


def test_add_and_subtract():
    ms = SortedMultiset()
    ms.add(200, 5_000_000)
    ms.add(200, 12)
    ms.add(404)
    ms._check()

    assert ms.multiplicity(200) == 5_000_012
    assert ms.multiplicity(404) == 1

    ms.subtract(200, 12)
    assert ms.multiplicity(200) == 5_000_000

    ms.subtract(404)
    assert 404 not in ms
    ms._check()

    with pytest.raises(NegativeCountError) as exc:
        ms.subtract(200, 5_000_001)

    assert exc.value.offending == {200: -1}
    assert ms.multiplicity(200) == 5_000_000

    ms.adjust(200, -5_000_000)
    assert 200 not in ms
    ms.adjust(7, 3)
    assert ms.multiplicity(7) == 3
    ms._check()


def test_set_count_discard_remove_clear():
    ms = SortedMultiset([(1, 3), (2, 5)])

    ms.set_count(1, 9)
    assert ms.multiplicity(1) == 9
    ms.set_count(1, 0)
    assert 1 not in ms
    ms.set_count(9, 4)
    assert ms.multiplicity(9) == 4

    ms.discard(2)
    ms.discard(2)
    assert 2 not in ms

    with pytest.raises(KeyError):
        ms.remove(2)

    ms.remove(9)
    assert ms.distinct_count() == 0

    ms.add(1, 2)
    ms.clear()
    assert len(ms) == 0
    ms._check()


def test_invalid_counts():
    ms = SortedMultiset()

    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            ms.add(1, bad)
        with pytest.raises(ValueError):
            ms.subtract(1, bad)

    with pytest.raises(ValueError):
        ms.set_count(1, -1)

    with pytest.raises(ValueError):
        ms.adjust(1, 2.0)

    with pytest.raises(ValueError):
        SortedMultiset([(1, 0)])


def test_rank_queries_and_slicing():
    ms = SortedMultiset([('a', 3), ('b', 2), ('c', 4)])

    assert ms.bisect_left('b') == 3
    assert ms.bisect_right('b') == 5
    assert ms.rank_range('b') == (3, 5)
    assert ms.rank_range('a') == (0, 3)
    assert ms.rank_range('c') == (5, 9)

    with pytest.raises(KeyError):
        ms.rank_range('z')

    assert ms[0] == 'a'
    assert ms[3] == 'b'
    assert ms[-1] == 'c'
    assert ms[3:5] == ['b', 'b']
    assert ms[::3] == ['a', 'b', 'c']

    with pytest.raises(IndexError):
        ms[9]

    with pytest.raises(IndexError):
        SortedMultiset()[0]


def test_quantiles():
    ms = SortedMultiset([(1, 10), (2, 10), (3, 10)])

    assert ms.percentile(0) == 1
    assert ms.percentile(50) == 2
    assert ms.percentile(100) == 3
    assert ms.quantile(0.25) == 1
    assert ms.quantile(0.9) == 3

    with pytest.raises(ValueError):
        ms.percentile(101)

    with pytest.raises(IndexError):
        SortedMultiset().percentile(50)


def test_merge_and_withdraw():
    left = SortedMultiset([(200, 5), (404, 1)])
    right = SortedMultiset([(200, 2), (500, 7)])

    left.merge(right)
    assert list(left.items()) == [(200, 7), (404, 1), (500, 7)]
    left._check()

    left.merge({404: 3})
    assert left.multiplicity(404) == 4

    left.withdraw([(200, 7), (500, 7)])
    assert list(left.items()) == [(404, 4)]
    left._check()

    before = list(left.items())

    with pytest.raises(NegativeCountError) as exc:
        left.withdraw({404: 5, 200: 1})

    assert exc.value.offending == {404: -1, 200: -1}
    assert list(left.items()) == before
    left._check()

    with pytest.raises(ValueError):
        left.withdraw({404: 0})


def test_apply_deltas_atomicity():
    ms = SortedMultiset([(1, 2), (2, 4)])

    with pytest.raises(NegativeCountError) as exc:
        ms.apply_deltas({1: 5, 2: -10})

    assert exc.value.offending == {2: -6}
    assert list(ms.items()) == [(1, 2), (2, 4)]

    ms.apply_deltas([(1, 5), (2, -4), (3, 2)])
    assert list(ms.items()) == [(1, 7), (3, 2)]
    ms._check()

    with pytest.raises(ValueError):
        ms.apply_deltas({1: 1.5})


def test_transaction_context():
    ms = SortedMultiset([(1, 2)])

    with ms.transaction() as tx:
        tx.add(2, 4)
        tx.subtract(1, 1)
        tx.adjust(3, -0)

    assert list(ms.items()) == [(1, 1), (2, 4)]
    ms._check()

    before = list(ms.items())

    with pytest.raises(NegativeCountError):
        with ms.transaction() as tx:
            tx.subtract(1, 5)
            tx.add(3, 1)

    assert list(ms.items()) == before

    with pytest.raises(RuntimeError):
        with ms.transaction() as tx:
            tx.add(9, 1)
            raise RuntimeError('abort')

    assert 9 not in ms
    assert list(ms.items()) == before
    ms._check()


def test_snapshot_is_immutable_view():
    ms = SortedMultiset([(1, 2), (2, 3)])
    view = ms.snapshot()

    assert isinstance(view, SortedMultisetView)
    assert view.snapshot() is view

    ms.add(1, 10)
    ms.subtract(2, 3)
    ms.add(5, 1)

    assert len(view) == 5
    assert view.distinct_count() == 2
    assert view.rank_range(1) == (0, 2)
    assert view.rank_range(2) == (2, 5)
    assert list(view.elements()) == [1, 1, 2, 2, 2]
    assert view == [1, 1, 2, 2, 2]

    assert len(ms) == 13
    assert ms.rank_range(1) == (0, 12)
    assert 2 not in ms
    ms._check()
    view._check()


def test_expanded_cross_check_random():
    random.seed(20261113)

    for _ in range(20):
        values = [random.randint(0, 30) for _ in range(random.randint(0, 300))]
        ms = SortedMultiset.from_values(values)
        expanded = sorted(values)
        ms._check()

        assert len(ms) == len(expanded)
        assert list(ms.elements()) == expanded
        assert ms == expanded

        for rank in range(len(expanded)):
            assert ms[rank] == expanded[rank]
            assert ms[rank - len(expanded)] == expanded[rank - len(expanded)]

        for start in range(0, len(expanded), 7):
            for stop in range(start, len(expanded) + 1, 11):
                assert ms[start:stop] == expanded[start:stop]

        for value in set(values):
            lo = expanded.index(value)
            hi = lo + expanded.count(value)
            assert ms.bisect_left(value) == lo
            assert ms.bisect_right(value) == hi
            assert ms.rank_range(value) == (lo, hi)
            assert ms.multiplicity(value) == expanded.count(value)

        absent = 101
        assert ms.bisect_left(absent) == len(expanded)
        assert ms.bisect_right(absent) == len(expanded)


def test_randomized_mutations_cross_check():
    random.seed(7)
    ms = SortedMultiset()
    reference = Counter()

    for step in range(2000):
        value = random.randint(0, 12)
        choice = random.random()

        if choice < 0.45:
            count = random.randint(1, 5)
            ms.add(value, count)
            reference[value] += count
        elif choice < 0.8:
            count = random.randint(1, 5)

            if count > reference[value]:
                with pytest.raises(NegativeCountError):
                    ms.subtract(value, count)
            else:
                ms.subtract(value, count)
                reference[value] -= count

                if reference[value] == 0:
                    del reference[value]
        else:
            count = random.randint(0, 6)
            ms.set_count(value, count)

            if count:
                reference[value] = count
            else:
                reference.pop(value, None)

        if step % 50 == 0:
            ms._check()
            expanded = sorted(reference.elements())
            assert list(ms.elements()) == expanded
            assert len(ms) == sum(reference.values())

            if expanded:
                rank = random.randrange(len(expanded))
                assert ms[rank] == expanded[rank]

    ms._check()


def test_point_update_is_logarithmic_and_no_rebuild():
    small = SortedMultiset((value, value + 10) for value in range(1024))
    large = SortedMultiset((value, value + 10) for value in range(4096))

    for ms, distinct in ((small, 1024), (large, 4096)):
        bound = 2 * math.ceil(math.log2(distinct)) + 2

        ms._tree.reset_accesses()
        ms[distinct // 2]
        assert ms._tree.accesses <= bound

        ms._tree.reset_accesses()
        ms.bisect_left(distinct // 3)
        assert ms._tree.accesses <= bound

        ms._tree.reset_accesses()
        ms.rank_range(distinct // 3)
        assert ms._tree.accesses <= 2 * bound

        tree_before = ms._tree
        ms._tree.reset_accesses()
        ms.add(distinct // 2, 999)
        assert ms._tree is tree_before
        assert ms._tree.accesses <= bound

        ms._tree.reset_accesses()
        ms.subtract(distinct // 2, 999)
        assert ms._tree is tree_before

    small._tree.reset_accesses()
    small[512]
    small_cost = small._tree.accesses
    large._tree.reset_accesses()
    large[2048]
    large_cost = large._tree.accesses

    assert large_cost < 2 * small_cost

    tree_before = large._tree
    large.add(1_000_000, 1)
    assert large._tree is not tree_before
    large._check()


def test_merge_withdraw_partitions_cross_check():
    random.seed(99)
    primary = SortedMultiset()
    reference = Counter()

    for partition in range(6):
        partition_ms = SortedMultiset()
        partition_counts = {}

        for _ in range(100):
            value = random.randint(0, 20)
            count = random.randint(1, 4)
            partition_ms.add(value, count)
            partition_counts[value] = partition_counts.get(value, 0) + count

        primary.merge(partition_ms)

        for value, count in partition_counts.items():
            reference[value] += count

        primary._check()
        assert list(primary.elements()) == sorted(reference.elements())

    snapshot = primary.snapshot()

    for partition in range(3):
        withdrawal = {}

        for _ in range(40):
            value = random.randint(0, 20)
            count = random.randint(1, 2)
            withdrawal[value] = withdrawal.get(value, 0) + count

        if any(count > reference.get(value, 0) for value, count in withdrawal.items()):
            before = list(primary.items())

            with pytest.raises(NegativeCountError):
                primary.withdraw(withdrawal)

            assert list(primary.items()) == before
        else:
            primary.withdraw(withdrawal)

            for value, count in withdrawal.items():
                reference[value] -= count

                if reference[value] == 0:
                    del reference[value]

        primary._check()
        assert list(primary.elements()) == sorted(reference.elements())

    assert list(snapshot.elements()) != list(primary.elements())
    snapshot._check()
    primary._check()


def test_repr_and_equality():
    ms = SortedMultiset([(1, 2), (2, 1)])
    assert repr(ms) == 'SortedMultiset({1: 2, 2: 1})'
    assert repr(ms.snapshot()) == 'SortedMultisetView({1: 2, 2: 1})'

    assert ms == [1, 1, 2]
    assert ms != [1, 1]
    assert ms == ms.snapshot()
    assert ms != 42
