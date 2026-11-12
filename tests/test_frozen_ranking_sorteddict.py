import pytest

from sortedcontainers import (
    FrozenViewRanking,
    RankConflictError,
    SortedDict,
    StaleRankError,
)


def test_view_freeze_captures_rows():
    sd = SortedDict({'a': 1, 'b': 2, 'c': 3})

    keys = sd.keys().freeze(token='k')
    items = sd.items().freeze()
    values = sd.values().freeze()

    assert isinstance(keys, FrozenViewRanking)
    assert keys.token == 'k'
    assert keys.version == sd._list.version
    assert list(keys) == ['a', 'b', 'c']
    assert list(items) == [('a', 1), ('b', 2), ('c', 3)]
    assert list(values) == [1, 2, 3]
    assert items[-1] == ('c', 3)
    assert values[-2] == 2


def test_insertions_do_not_drift_view_target():
    sd = SortedDict({letter: index for index, letter in enumerate('abcde')})
    ranking = sd.items().freeze()

    sd['aa'] = 99
    sd['z'] = 99

    current_rank, row = ranking.locate(2)

    assert current_rank == 3
    assert row == ('c', 2)
    assert sd.keys()[current_rank] == 'c'


def test_negative_rank_relative_to_frozen_view():
    sd = SortedDict({str(index): index for index in range(6)})
    ranking = sd.keys().freeze()

    assert ranking[-1] == '5'

    current_rank, key = ranking.locate(-3)
    assert key == '3'
    assert current_rank == 3

    with pytest.raises(IndexError):
        ranking.locate(-7)


@pytest.mark.parametrize(
    'view_name, expected_row',
    [('keys', 'c'), ('items', ('c', 3)), ('values', 3)],
)
def test_locate_on_each_view(view_name, expected_row):
    sd = SortedDict({'a': 1, 'b': 2, 'c': 3})
    view = getattr(sd, view_name)()
    ranking = view.freeze()

    sd['a0'] = 0

    current_rank, row = ranking.locate(2)
    assert row == expected_row
    assert current_rank == 3


def test_quarantine_via_keys_view_removes_mapping_entry():
    sd = SortedDict({letter: index for index, letter in enumerate('abcde')})
    ranking = sd.keys().freeze(token='inspection')

    sd['aa'] = 99
    receipt = ranking.quarantine(1)

    assert receipt['action'] == 'quarantined'
    assert receipt['old_rank'] == 1
    assert receipt['current_rank'] == 2
    assert receipt['key'] == 'b'
    assert receipt['row'] == 'b'
    assert receipt['token'] == 'inspection'
    assert 'b' not in sd
    assert len(sd) == 5
    sd._check()


def test_quarantine_via_values_view_receipt_carries_row():
    sd = SortedDict({'a': 1, 'b': 2, 'c': 3})
    ranking = sd.values().freeze()

    receipt = ranking.quarantine(0)

    assert receipt['row'] == 1
    assert receipt['key'] == 'a'
    assert 'a' not in sd


def test_duplicate_request_quarantines_once_with_receipt():
    sd = SortedDict({'a': 1, 'b': 2, 'c': 3})
    ranking = sd.keys().freeze()

    first = ranking.quarantine(1)
    size = len(sd)
    second = ranking.quarantine(1)

    assert first['action'] == 'quarantined'
    assert second['action'] == 'duplicate'
    assert second['key'] == 'b'
    assert second['token'] == first['token']
    assert len(sd) == size
    assert ranking.quarantined == {1: first}


def test_value_change_conflicts_on_items_and_values_views():
    sd = SortedDict({'a': 1, 'b': 2, 'c': 3})
    items = sd.items().freeze()
    values = sd.values().freeze()

    sd['b'] = 20

    for ranking in (items, values):
        with pytest.raises(RankConflictError):
            ranking.quarantine(1)

    assert sd['b'] == 20
    assert len(sd) == 3


def test_in_place_mutable_value_change_conflicts():
    sd = SortedDict()
    payload = {'score': 1}
    sd['row'] = payload
    ranking = sd.values().freeze()

    payload['score'] = 2

    with pytest.raises(RankConflictError):
        ranking.quarantine(0)

    assert 'row' in sd


def test_removed_key_is_stale():
    sd = SortedDict({'a': 1, 'b': 2, 'c': 3})
    ranking = sd.items().freeze()

    del sd['b']

    with pytest.raises(StaleRankError):
        ranking.quarantine(1)

    assert len(sd) == 2


def test_quarantine_all_validates_before_any_deletion():
    sd = SortedDict({str(index): index for index in range(6)})
    ranking = sd.items().freeze()

    sd['x'] = -1
    sd['3'] = 300
    snapshot = dict(sd)

    with pytest.raises(RankConflictError):
        ranking.quarantine_all([0, 3, 5])

    assert dict(sd) == snapshot
    assert len(sd) == 7
    sd._check()


def test_quarantine_all_stale_key_leaves_mapping_intact():
    sd = SortedDict({str(index): index for index in range(6)})
    ranking = sd.keys().freeze()

    del sd['2']

    with pytest.raises(StaleRankError):
        ranking.quarantine_all([0, 2, 4])

    assert set(sd) == {'0', '1', '3', '4', '5'}
    sd._check()


def test_quarantine_all_one_receipt_per_request():
    sd = SortedDict({str(index): index for index in range(8)})
    ranking = sd.values().freeze()

    sd['m'] = -1
    receipts = ranking.quarantine_all([0, 2, 2, -1])

    assert [receipt['action'] for receipt in receipts] == [
        'quarantined',
        'quarantined',
        'duplicate',
        'quarantined',
    ]
    assert [receipt['old_rank'] for receipt in receipts] == [0, 2, 2, 7]
    assert [receipt['key'] for receipt in receipts] == ['0', '2', '2', '7']
    assert set(sd) == {'1', '3', '4', '5', '6', 'm'}
    sd._check()


def test_repr_contains_token_and_version():
    ranking = SortedDict({'a': 1}).keys().freeze(token='xyz')
    text = repr(ranking)

    assert 'FrozenViewRanking' in text
    assert 'xyz' in text
