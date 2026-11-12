import json
import random

import pytest

from sortedcontainers import (
    InvalidCursorError,
    Migration,
    Snapshot,
    SortedKeyList,
    SortedList,
    SortedSet,
    SortPolicyRegistry,
    migrate,
    register_policy,
)
from sortedcontainers.sortedmigration import DEFAULT_REGISTRY


def identity(value):
    return value


def negate(value):
    return -value


def modulo(value):
    return value % 5


def modulo_then_value(value):
    return (value % 5, value)


def make_registry():
    registry = SortPolicyRegistry()
    registry.register('identity', identity)
    registry.register('negate', negate)
    registry.register('modulo', modulo)
    registry.register('modulo-then-value', modulo_then_value)
    return registry


def test_registry_register():
    registry = SortPolicyRegistry()
    assert registry.register('negate', negate) is negate
    assert 'negate' in registry
    assert registry['negate'] is negate
    assert len(registry) == 1
    assert list(registry) == ['negate']
    assert repr(registry) == "SortPolicyRegistry(['negate'])"


def test_registry_register_idempotent():
    registry = SortPolicyRegistry()
    registry.register('negate', negate)
    registry.register('negate', negate)
    assert len(registry) == 1


def test_registry_register_errors():
    registry = SortPolicyRegistry()
    with pytest.raises(TypeError):
        registry.register(0, negate)
    with pytest.raises(TypeError):
        registry.register('negate', 0)
    registry.register('negate', negate)
    with pytest.raises(ValueError):
        registry.register('negate', identity)


def test_registry_lookup_errors():
    registry = SortPolicyRegistry()
    with pytest.raises(KeyError):
        registry['missing']
    with pytest.raises(ValueError):
        registry.name_of(negate)
    registry.register('negate', negate)
    assert registry.name_of(negate) == 'negate'


def test_default_registry():
    assert 'identity' in DEFAULT_REGISTRY
    register_policy('test-default-negate', negate)
    assert DEFAULT_REGISTRY['test-default-negate'] is negate


def test_capture_sorted_list():
    values = SortedList([3, 1, 2])
    snapshot = Snapshot.capture(values)
    assert snapshot.policy == 'identity'
    assert snapshot.rows == [1, 2, 3]
    assert len(snapshot) == 3


def test_capture_infers_key():
    values = SortedKeyList([1, 2, 3], key=negate)
    registry = make_registry()
    snapshot = Snapshot.capture(values, registry=registry)
    assert snapshot.policy == 'negate'
    assert snapshot.rows == [3, 2, 1]


def test_capture_sorted_set():
    values = SortedSet([1, 2, 3], key=negate)
    registry = make_registry()
    snapshot = Snapshot.capture(values, registry=registry)
    assert snapshot.policy == 'negate'
    assert snapshot.rows == [3, 2, 1]


def test_capture_explicit_policy():
    values = SortedKeyList([1, 2, 3], key=negate)
    registry = make_registry()
    snapshot = Snapshot.capture(values, 'negate', registry)
    assert snapshot.policy == 'negate'
    snapshot = Snapshot.capture(values, negate, registry)
    assert snapshot.policy == 'negate'


def test_capture_unregistered_policy():
    values = SortedList([1, 2, 3])
    registry = make_registry()
    with pytest.raises(ValueError):
        Snapshot.capture(values, 'missing', registry)
    with pytest.raises(ValueError):
        Snapshot.capture(values, lambda value: value, registry)
    keyed = SortedKeyList([1, 2, 3], key=lambda value: value)
    with pytest.raises(ValueError):
        Snapshot.capture(keyed, registry=registry)


def test_snapshot_round_trip():
    registry = make_registry()
    values = SortedKeyList([3, 1, 2], key=negate)
    snapshot = Snapshot.capture(values, 'negate', registry)
    data = snapshot.to_dict()
    assert set(data) == {'policy', 'rows'}
    assert data == {'policy': 'negate', 'rows': [3, 2, 1]}
    assert Snapshot.from_dict(data, registry) == snapshot
    text = snapshot.to_json()
    assert Snapshot.from_json(text, registry) == snapshot
    assert Snapshot.from_dict(json.loads(text), registry) == snapshot


def test_snapshot_serializes_no_functions():
    snapshot = Snapshot.capture(SortedList([2, 1]))
    data = snapshot.to_dict()
    assert isinstance(data['policy'], str)
    assert all(not callable(value) for value in data.values())
    json.dumps(data)


def test_snapshot_from_dict_unregistered():
    registry = make_registry()
    with pytest.raises(KeyError):
        Snapshot.from_dict({'policy': 'missing', 'rows': []}, registry)


def test_snapshot_equality():
    first = Snapshot([1, 2], 'identity')
    assert first == Snapshot([1, 2], 'identity')
    assert first != Snapshot([2, 1], 'identity')
    assert first != Snapshot([1, 2], 'negate')
    assert first != [1, 2]
    assert repr(first) == "Snapshot(policy='identity', rows=[1, 2])"


def test_migrate_refinement():
    registry = make_registry()
    values = SortedKeyList(range(20), key=modulo)
    snapshot = Snapshot.capture(values, 'modulo', registry)
    migration = snapshot.migrate('modulo-then-value', registry)
    assert migration.kind == 'refinement'
    assert migration.affected_rows == ()
    assert list(migration.values) == sorted(range(20), key=modulo_then_value)
    assert list(migration.sorted_list) == list(migration.values)
    assert len(migration) == 20
    for cursor in range(21):
        plan = migration.cursor(cursor)
        assert plan.continuable
        assert plan.status in ('exact', 'ranged')
        assert plan.stop == 20
        tail = list(plan.tail_rows())
        assert sorted(row for row, _ in tail) == list(range(cursor, 20))
        assert [value for _, value in tail] == [snapshot.rows[row] for row, _ in tail]


def test_migrate_refinement_intervals():
    registry = SortPolicyRegistry()
    registry.register('length', len)
    registry.register('length-then-text', lambda value: (len(value), value))
    words = SortedKeyList(['cherry', 'fig', 'banana', 'apple', 'kiwi'], key=len)
    snapshot = Snapshot.capture(words, 'length', registry)
    assert snapshot.rows == ['fig', 'kiwi', 'apple', 'cherry', 'banana']
    migration = snapshot.migrate('length-then-text', registry)
    assert migration.kind == 'refinement'
    assert list(migration.values) == ['fig', 'kiwi', 'apple', 'banana', 'cherry']
    # Group boundaries transfer exactly at the same index.
    for cursor in (0, 1, 2, 3, 5):
        plan = migration.cursor(cursor)
        assert plan.status == 'exact'
        assert (plan.start, plan.stop) == (cursor, 5)
    # Mid-group cursor resumes at the start of its equal-key group.
    plan = migration.cursor(4)
    assert plan.status == 'ranged'
    assert (plan.start, plan.stop) == (3, 5)
    assert list(plan.tail()) == ['banana']
    assert list(plan.tail_rows()) == [(4, 'banana')]


def test_migrate_reorder():
    registry = make_registry()
    values = SortedList([1, 2, 3, 4])
    snapshot = Snapshot.capture(values, 'identity', registry)
    migration = snapshot.migrate('negate', registry)
    assert migration.kind == 'reorder'
    assert migration.affected_rows == (0, 1, 2, 3)
    assert list(migration.values) == [4, 3, 2, 1]
    for cursor in (1, 2, 3):
        plan = migration.cursor(cursor)
        assert plan.status == 'invalid'
        assert not plan.continuable
        assert plan.start is None and plan.stop is None
        with pytest.raises(InvalidCursorError):
            list(plan.tail())
        with pytest.raises(InvalidCursorError):
            list(plan.tail_rows())
    for cursor in (0, 4):
        plan = migration.cursor(cursor)
        assert plan.status == 'exact'
        assert plan.continuable


def test_migrate_reorder_partial():
    registry = make_registry()
    values = SortedKeyList([0, 1, 2, 3], key=identity)
    snapshot = Snapshot.capture(values, 'identity', registry)
    # Swap the first two rows only; later cuts stay order-separated.
    registry.register('swap-first-two', lambda value: (1, 0, 2, 3)[value])
    migration = snapshot.migrate('swap-first-two', registry)
    assert migration.kind == 'reorder'
    assert migration.affected_rows == (0, 1)
    assert list(migration.values) == [1, 0, 2, 3]
    assert migration.cursor(1).status == 'invalid'
    assert migration.cursor(2).status == 'exact'
    assert migration.cursor(3).status == 'exact'
    assert list(migration.cursor(2).tail()) == [2, 3]


def test_migrate_duplicates_are_independent_rows():
    registry = make_registry()
    values = SortedList([2, 1, 2, 1, 3])
    snapshot = Snapshot.capture(values)
    assert snapshot.rows == [1, 1, 2, 2, 3]
    migration = snapshot.migrate('negate', registry)
    assert migration.kind == 'reorder'
    assert len(migration.order) == 5
    assert len(set(migration.order)) == 5
    plan = migration.cursor(0)
    rows = list(plan.tail_rows())
    assert sorted(row for row, _ in rows) == [0, 1, 2, 3, 4]
    assert [value for _, value in rows] == [3, 2, 2, 1, 1]


def test_migrate_duplicates_refinement():
    registry = make_registry()
    values = SortedKeyList([5, 10, 15, 6, 11, 16], key=modulo)
    snapshot = Snapshot.capture(values, 'modulo', registry)
    assert snapshot.rows == [5, 10, 15, 6, 11, 16]
    migration = snapshot.migrate('modulo-then-value', registry)
    assert migration.kind == 'refinement'
    assert list(migration.values) == [5, 10, 15, 6, 11, 16]
    plan = migration.cursor(2)
    assert plan.status == 'exact'
    assert list(plan.tail()) == [15, 6, 11, 16]


def test_migrate_empty_snapshot():
    registry = make_registry()
    snapshot = Snapshot.capture(SortedList(), registry=registry)
    migration = snapshot.migrate('negate', registry)
    assert migration.kind == 'refinement'
    assert migration.affected_rows == ()
    assert len(migration) == 0
    plan = migration.cursor(0)
    assert plan.status == 'exact'
    assert (plan.start, plan.stop) == (0, 0)
    assert list(plan.tail()) == []


def test_migrate_callable_policy():
    registry = make_registry()
    snapshot = Snapshot.capture(SortedList([1, 2, 3]), registry=registry)
    migration = snapshot.migrate(negate, registry)
    assert migration.policy == 'negate'
    assert migration.kind == 'reorder'


def test_migrate_unregistered_policy():
    registry = make_registry()
    snapshot = Snapshot.capture(SortedList([1, 2, 3]), registry=registry)
    with pytest.raises(KeyError):
        snapshot.migrate('missing', registry)
    with pytest.raises(ValueError):
        snapshot.migrate(lambda value: value, registry)
    orphan = Snapshot([1], 'missing')
    with pytest.raises(KeyError):
        orphan.migrate('negate', registry)


def test_migrate_module_function():
    snapshot = Snapshot.capture(SortedList([1, 2, 3]))
    migration = migrate(snapshot, 'identity')
    assert isinstance(migration, Migration)
    assert migration.kind == 'refinement'


def test_cursor_out_of_range():
    snapshot = Snapshot.capture(SortedList([1, 2, 3]))
    migration = snapshot.migrate('identity')
    for bad in (-1, 4, 1.5, '1', None):
        with pytest.raises(ValueError):
            migration.cursor(bad)


def test_cursor_repr():
    snapshot = Snapshot.capture(SortedList([1, 2, 3]))
    migration = snapshot.migrate('identity')
    plan = migration.cursor(1)
    assert repr(plan) == "CursorPlan(cursor=1, status='exact', start=1, stop=3)"
    assert repr(migration) == "Migration(policy='identity', kind='refinement', rows=3)"


def brute_force_affected(rows, old_key, new_key):
    count = len(rows)
    old_keys = [old_key(value) for value in rows]
    new_keys = [new_key(value) for value in rows]
    affected = set()
    refinement = True
    for first in range(count):
        for second in range(count):
            if (
                old_keys[first] < old_keys[second]
                and new_keys[first] > new_keys[second]
            ):
                refinement = False
                affected.add(first)
                affected.add(second)
    return refinement, affected


def test_migrate_randomized_exactly_once():
    registry = make_registry()
    policies = ['identity', 'negate', 'modulo', 'modulo-then-value']
    rng = random.Random(20261113)
    for _ in range(200):
        values = [rng.randrange(20) for _ in range(rng.randrange(30))]
        old_name = rng.choice(policies)
        new_name = rng.choice(policies)
        old_key = registry[old_name]
        new_key = registry[new_name]
        container = SortedKeyList(values, key=old_key)
        snapshot = Snapshot.capture(container, old_name, registry)
        migration = snapshot.migrate(new_name, registry)

        refinement, affected = brute_force_affected(snapshot.rows, old_key, new_key)
        assert (migration.kind == 'refinement') == refinement
        assert set(migration.affected_rows) == affected

        count = len(snapshot.rows)
        assert sorted(migration.order) == list(range(count))
        assert list(migration.sorted_list) == list(migration.values)
        for position, row in enumerate(migration.order):
            assert migration.values[position] == snapshot.rows[row]

        for cursor in range(count + 1):
            plan = migration.cursor(cursor)
            if refinement:
                assert plan.continuable
            if not plan.continuable:
                continue
            tail_rows = list(plan.tail_rows())
            tail_ids = [row for row, _ in tail_rows]
            # Old page prefix plus migrated tail covers each row exactly once.
            assert len(tail_ids) == len(set(tail_ids))
            assert sorted(list(range(cursor)) + tail_ids) == list(range(count))
            assert [value for _, value in tail_rows] == [
                snapshot.rows[row] for row in tail_ids
            ]
            # Tail rows lie within the reported interval.
            positions = sorted(migration.order.index(row) for row in tail_ids)
            assert all(plan.start <= pos < plan.stop for pos in positions)
            if plan.status == 'exact':
                assert (plan.start, plan.stop) == (cursor, count)
                assert sorted(tail_ids) == list(range(cursor, count))
