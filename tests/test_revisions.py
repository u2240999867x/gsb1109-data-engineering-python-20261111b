"Tests for atomic revision orders on the sorted dictionary recipes."

import copy
import pickle

import pytest

from sortedcollections import (
    ItemSortedDict,
    REMOVE,
    RevisionError,
    ValueSortedDict,
)


class ExplodingValues:
    "Key function that records every value it is asked to compare."

    def __init__(self, bad=()):
        self.bad = set(bad)
        self.calls = []

    def __call__(self, value):
        self.calls.append(value)
        if value in self.bad:
            raise ValueError('bad sort value {0!r}'.format(value))
        return value


def identity(value):
    return value


def item_key(key, value):
    return value


def make_value_dict(bad=()):
    func = ExplodingValues(bad)
    mapping = ValueSortedDict(func, {0: 0, 1: 1, 2: 2, 3: 3})
    mapping._list._reset(4)
    return mapping, func


def make_item_dict(bad=()):
    func = ExplodingValues(bad)

    def key_func(key, value):
        return func(value)

    mapping = ItemSortedDict(key_func, {0: 0, 1: 1, 2: 2, 3: 3})
    mapping._list._reset(4)
    return mapping, func


@pytest.mark.parametrize('factory', [make_value_dict, make_item_dict])
def test_revision_commutes_change_add_and_withdraw_together(factory):
    mapping, _ = factory()
    with mapping.revision([(1, 10), (4, -1), (2, REMOVE)]):
        pass

    assert dict(mapping) == {0: 0, 1: 10, 3: 3, 4: -1}
    assert list(mapping) == [4, 0, 3, 1]
    mapping._check()


@pytest.mark.parametrize('factory', [make_value_dict, make_item_dict])
def test_revision_accepts_mapping_and_explicit_commit(factory):
    mapping, _ = factory()
    order = mapping.revision({4: 40})
    order.commit()
    assert mapping[4] == 40
    mapping._check()


def test_revision_empty_order_is_noop():
    mapping, _ = make_value_dict()
    before = dict(mapping)
    with mapping.revision():
        pass
    assert dict(mapping) == before
    mapping._check()


def test_revision_order_cannot_commit_twice():
    mapping, _ = make_value_dict()
    order = mapping.revision([(4, 40)])
    order.commit()
    with pytest.raises(RuntimeError):
        order.commit()
    with pytest.raises(RuntimeError):
        with order:
            pass


def test_revision_exception_in_body_suppresses_commit():
    mapping, _ = make_value_dict()
    before = dict(mapping)

    class Abort(Exception):
        pass

    with pytest.raises(Abort):
        with mapping.revision([(4, 40)]):
            raise Abort()

    assert dict(mapping) == before
    mapping._check()


@pytest.mark.parametrize('factory', [make_value_dict, make_item_dict])
def test_key_function_failure_is_located_by_source_row(factory):
    mapping, func = factory(bad={'boom'})
    before = dict(mapping)
    func.calls.clear()

    with pytest.raises(RevisionError) as exc_info:
        with mapping.revision([(0, 9), (4, 8), (5, 'boom'), (1, 7)]):
            pass

    error = exc_info.value
    assert error.row == 2
    assert error.key == 5
    assert isinstance(error.__cause__, ValueError)
    assert func.calls == [9, 8, 'boom']
    assert dict(mapping) == before
    assert list(mapping) == [0, 1, 2, 3]
    mapping._check()


def test_each_new_value_is_keyed_exactly_once_on_success():
    mapping, func = make_value_dict()
    func.calls.clear()
    with mapping.revision([(0, 10), (4, 20), (1, 30), (2, REMOVE)]):
        pass
    assert sorted(func.calls) == [10, 20, 30]


def test_comparison_failure_is_located_by_source_row():
    mapping, _ = make_value_dict()
    before = dict(mapping)
    blocks = len(mapping._list._lists)
    view = mapping.keys()

    with pytest.raises(RevisionError) as exc_info:
        with mapping.revision([(0, 10), (4, 'nope'), (5, 11)]):
            pass

    error = exc_info.value
    assert error.row == 1
    assert error.key == 4
    assert isinstance(error.__cause__, TypeError)
    assert dict(mapping) == before
    assert len(mapping._list._lists) == blocks
    assert list(view) == [0, 1, 2, 3]
    mapping._check()


def test_legal_revision_still_commits_after_failure():
    mapping, _ = make_value_dict()
    with pytest.raises(RevisionError):
        with mapping.revision([(4, 'nope')]):
            pass

    with mapping.revision([(4, -4), (1, 10)]):
        pass

    assert dict(mapping) == {0: 0, 1: 10, 2: 2, 3: 3, 4: -4}
    assert list(mapping) == [4, 0, 2, 3, 1]
    mapping._check()


def test_setitem_failure_keeps_old_key_value_and_rank():
    mapping, _ = make_value_dict()
    before = dict(mapping)
    view = mapping.keys()

    with pytest.raises(TypeError):
        mapping[1] = 'nope'

    assert dict(mapping) == before
    assert list(view) == [0, 1, 2, 3]
    assert mapping.bisect_key_left(2) == 2
    assert mapping.keys()[1:3] == [1, 2]
    mapping._check()

    mapping[1] = 10
    assert mapping[1] == 10
    mapping._check()


def test_setitem_new_incompatible_value_raises_raw_exception():
    mapping, _ = make_value_dict()
    before = dict(mapping)
    with pytest.raises(TypeError):
        mapping[9] = 'nope'
    assert dict(mapping) == before
    mapping._check()


def test_setitem_key_function_error_is_not_wrapped():
    mapping, _ = make_value_dict(bad={'boom'})
    before = dict(mapping)
    with pytest.raises(ValueError):
        mapping[9] = 'boom'
    assert dict(mapping) == before
    mapping._check()


def test_non_iterable_pair_is_rejected_without_touching_state():
    mapping, _ = make_value_dict()
    before = dict(mapping)
    with pytest.raises(TypeError):
        with mapping.revision([42]):
            pass
    assert dict(mapping) == before
    mapping._check()


def test_delitem_missing_key_raises_keyerror_and_keeps_state():
    mapping, _ = make_value_dict()
    before = dict(mapping)
    with pytest.raises(KeyError):
        del mapping[9]
    assert dict(mapping) == before
    mapping._check()


def test_prepare_revision_missing_remove_without_wrapping():
    mapping, _ = make_value_dict()
    with pytest.raises(KeyError):
        mapping._prepare_revision([(9, REMOVE)], wrap_errors=False)


def test_revision_withdraw_missing_key_is_row_located():
    mapping, _ = make_value_dict()
    before = dict(mapping)
    with pytest.raises(RevisionError) as exc_info:
        with mapping.revision([(9, REMOVE)]):
            pass
    assert exc_info.value.row == 0
    assert exc_info.value.key == 9
    assert isinstance(exc_info.value.__cause__, KeyError)
    assert dict(mapping) == before


def test_duplicate_key_in_order_is_rejected():
    mapping, _ = make_value_dict()
    before = dict(mapping)
    with pytest.raises(ValueError):
        with mapping.revision([(0, 1), (0, 2)]):
            pass
    assert dict(mapping) == before
    mapping._check()


def test_malformed_pair_is_rejected_with_row():
    mapping, _ = make_value_dict()
    before = dict(mapping)
    with pytest.raises(TypeError):
        with mapping.revision([(0, 1), (42,)]):
            pass
    assert dict(mapping) == before
    mapping._check()


def test_update_is_atomic_and_supports_all_dict_forms():
    mapping, _ = make_value_dict()
    before = dict(mapping)

    with pytest.raises(RevisionError) as exc_info:
        mapping.update([(0, 9), (4, 'nope')])
    assert exc_info.value.row == 1
    assert dict(mapping) == before
    mapping._check()

    mapping.update({0: 9}, c=12)
    assert mapping[0] == 9 and mapping['c'] == 12
    mapping._check()


def test_setdefault_pop_and_clear_still_work():
    mapping, _ = make_value_dict()
    assert mapping.setdefault(9, -9) == -9
    assert mapping[9] == -9
    assert mapping.pop(9) == -9
    assert mapping.pop(9, 7) == 7
    mapping.clear()
    assert not mapping
    mapping._check()


def test_views_remain_consistent_after_rollback():
    mapping, _ = make_value_dict()
    keys_view = mapping.keys()
    items_view = mapping.items()
    values_view = mapping.values()

    with pytest.raises(RevisionError):
        with mapping.revision([(4, 'nope')]):
            pass

    assert list(keys_view) == [0, 1, 2, 3]
    assert list(items_view) == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert list(values_view) == [0, 1, 2, 3]


def test_large_revision_matches_stepwise_legacy_ordering():
    legacy = ValueSortedDict({i: i for i in range(200)})
    legacy._list._reset(8)
    mapping = ValueSortedDict({i: i for i in range(200)})
    mapping._list._reset(8)

    rows = (
        [(i, i * 2) for i in range(0, 100, 2)]
        + [(i, REMOVE) for i in range(1, 50, 2)]
        + [(1000 + i, -i) for i in range(20)]
    )

    with mapping.revision(rows):
        pass

    for key, value in rows:
        if value is REMOVE:
            del legacy[key]
        else:
            legacy[key] = value

    mapping._check()
    legacy._check()
    assert list(mapping) == list(legacy)
    assert dict(mapping) == dict(legacy)


def test_copy_deepcopy_and_pickle_round_trip():
    mapping = ValueSortedDict(identity, {0: 0, 1: 1})
    with mapping.revision([(2, 2), (0, REMOVE)]):
        pass

    assert mapping.copy() == mapping
    assert copy.deepcopy(mapping) == mapping

    restored = pickle.loads(pickle.dumps(mapping))
    assert restored == mapping
    assert list(restored) == list(mapping)


def test_base_sort_key_helper_is_abstract():
    mapping, _ = make_value_dict()
    with pytest.raises(NotImplementedError):
        type(mapping).__mro__[1]._make_sort_key(mapping, 0, 0)
