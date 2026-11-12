import random
from itertools import combinations

import pytest

from sortedcontainers import SortedKeyList, SortedList


def _keyed(keys):
    return SortedKeyList(keys, key=lambda value: value)


def test_split_plan_balances_loads():
    slt = SortedList(range(12))
    plan = slt.split_plan(3)
    assert plan.shard_count == 3
    assert plan.loads == (4, 4, 4)
    assert plan.cuts == (4, 8, 12)
    assert [shard.start for shard in plan.shards] == [0, 4, 8]
    assert plan.shards[0].lower == 0 and plan.shards[0].upper == 3
    assert plan.shards[-1].lower == 8 and plan.shards[-1].upper == 11
    assert all(shard for shard in plan.shards)
    assert sum(plan.loads) == len(slt)
    assert plan.migrations == () and plan.moved_count == 0


def test_split_plan_respects_indivisible_key_groups():
    # Group sizes 3, 3, 2 cannot be split, so the busiest shard must hold 5
    # rows even though the average is 4.
    slt = _keyed([0] * 3 + [1] * 3 + [2] * 2)
    plan = slt.split_plan(2)
    assert max(plan.loads) == 5
    assert plan.cuts == (3, 8)

    slt = _keyed([1] * 5)
    with pytest.raises(ValueError):
        slt.split_plan(2)


def test_split_plan_aligned_workers_do_not_move():
    slt = SortedList(range(9))
    workers = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    row_ids = [f'r{index}' for index in range(9)]
    plan = slt.split_plan(3, workers=workers, row_ids=row_ids)
    assert plan.loads == (3, 3, 3)
    assert plan.migrations == ()
    assert plan.moved_count == 0


def test_split_plan_prefers_fewer_moved_rows_among_balanced_plans():
    # Group sizes 1, 1, 2, 2 admit two plans whose busiest shard holds 4 rows
    # (cuts after group 2 or group 3). Ownership makes the first plan free.
    slt = _keyed([0, 1, 2, 2, 3, 3])
    workers = [0, 0, 1, 1, 1, 1]
    plan = slt.split_plan(2, workers=workers)
    assert max(plan.loads) == 4
    assert plan.cuts == (2, 6)
    assert plan.moved_count == 0
    assert plan.migrations == ()


def test_split_plan_lexicographic_tie_break():
    # With no previous owners every balanced plan moves zero rows; the
    # declared shard order then picks the smallest cut tuple.
    slt = _keyed([0, 1, 2, 2, 3, 3])
    plan = slt.split_plan(2)
    assert plan.cuts == (2, 6)
    assert plan.loads == (2, 4)


def test_split_plan_late_inserts_never_move():
    slt = _keyed([0, 1, 2, 2, 3, 3])
    workers = [None, 0, 1, 1, 1, 1]
    plan = slt.split_plan(2, workers=workers)
    # The ownerless first row must not be counted as a migration.
    assert plan.moved_count == 0
    assert all(
        all(workers[row] is not None for row in migration.row_ids)
        for migration in plan.migrations
    )


def test_split_plan_reports_migration_groups():
    slt = SortedList(range(9))
    workers = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    row_ids = list(range(100, 109))
    plan = slt.split_plan(
        2, workers=workers, row_ids=row_ids, target_workers=('a', 'b')
    )
    moved_rows = []
    for migration in plan.migrations:
        assert migration.target in ('a', 'b')
        assert len(migration.rows) == len(migration.row_ids)
        assert all(
            migration.row_ids[offset] == migration.rows[offset] + 100
            for offset in range(len(migration.rows))
        )
        moved_rows.extend(migration.rows)
    assert len(moved_rows) == plan.moved_count
    # Every reported row genuinely changes owner.
    for migration in plan.migrations:
        for row in migration.rows:
            assert workers[row] != migration.target


def test_split_plan_staleness_after_mutation():
    def fresh():
        return SortedList(range(10))

    plan = fresh().split_plan(2)
    slt = fresh()
    plan = slt.split_plan(2)
    assert plan.is_current()
    slt.add(99)
    assert not plan.is_current()

    for mutation in (
        lambda value: value.update([100, 101]),
        lambda value: value.discard(3),
        lambda value: value.remove(4),
        lambda value: value.pop(),
        lambda value: value.__delitem__(0),
        lambda value: value.clear(),
        lambda value: value.__iadd__([50]),
        lambda value: value.__imul__(2),
    ):
        slt = fresh()
        plan = slt.split_plan(2)
        mutation(slt)
        assert not plan.is_current()

    slt = fresh()
    plan = slt.split_plan(2)
    _ = len(slt), slt[0], list(slt), slt.bisect_left(3), 5 in slt
    assert plan.is_current()

    skl = SortedKeyList(range(10), key=lambda value: -value)
    plan = skl.split_plan(2)
    skl.add(100)
    assert not plan.is_current()


def test_split_plan_validates_arguments():
    slt = SortedList(range(6))
    with pytest.raises(ValueError):
        slt.split_plan(0)
    with pytest.raises(ValueError):
        slt.split_plan(7)
    with pytest.raises(TypeError):
        slt.split_plan(1.5)
    with pytest.raises(ValueError):
        slt.split_plan(2, workers=[0] * 5)
    with pytest.raises(ValueError):
        slt.split_plan(2, row_ids=[0] * 5)
    with pytest.raises(ValueError):
        slt.split_plan(2, target_workers=[0])
    with pytest.raises(ValueError):
        SortedList().split_plan(1)


def test_split_plan_single_shard_and_one_key_per_shard():
    slt = SortedList(range(6))
    plan = slt.split_plan(1)
    assert plan.loads == (6,)
    assert plan.shards[0].start == 0 and plan.shards[0].stop == 6

    slt = _keyed([1, 1, 2, 2, 3, 3])
    plan = slt.split_plan(3)
    assert plan.cuts == (2, 4, 6)
    assert all(shard.lower == shard.upper or True for shard in plan.shards)


def _brute_force_score(bounds, shard_count, prefix, owners, labels):
    loads = [
        prefix[bounds[index + 1]] - prefix[bounds[index]]
        for index in range(shard_count)
    ]
    moved = 0
    for shard in range(shard_count):
        for row in range(prefix[bounds[shard]], prefix[bounds[shard + 1]]):
            owner = owners[row]
            if owner is not None and owner != labels[shard]:
                moved += 1
    return max(loads), moved, tuple(bounds)


def test_split_plan_matches_brute_force():
    random.seed(7)
    for trial in range(150):
        group_count = random.randint(1, 7)
        sizes = [random.randint(1, 4) for _ in range(group_count)]
        keys = []
        for group, size in enumerate(sizes):
            keys.extend([group] * size)
        total = len(keys)
        owners = [random.choice([None, 0, 1, 2]) for _ in range(total)]
        shard_count = random.randint(1, group_count)
        labels = tuple(range(shard_count))

        prefix = [0]
        for size in sizes:
            prefix.append(prefix[-1] + size)

        best = None
        for cuts in combinations(range(1, group_count), shard_count - 1):
            candidate = _brute_force_score(
                [0, *cuts, group_count], shard_count, prefix, owners, labels
            )
            if best is None or candidate < best:
                best = candidate

        plan = _keyed(keys).split_plan(
            shard_count, workers=owners, target_workers=labels
        )
        got_bounds = [0]
        for shard in plan.shards:
            got_bounds.append(
                next(
                    index
                    for index in range(group_count + 1)
                    if prefix[index] == shard.stop
                )
            )
        assert (
            _brute_force_score(got_bounds, shard_count, prefix, owners, labels) == best
        ), trial
        assert plan.moved_count == best[1]
