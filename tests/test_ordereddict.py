"Test sortedcollections.OrderedDict"

import pickle

import pytest

from sortedcollections import OrderedDict

pairs = dict(enumerate(range(10)))


def test_init():
    od = OrderedDict()
    assert len(od) == 0
    od._check()
    od = OrderedDict(enumerate(range(10)))
    assert len(od) == 10
    od._check()
    od = OrderedDict(a=0, b=1, c=2)
    assert len(od) == 3
    od._check()
    od = OrderedDict(pairs)
    assert len(od) == 10
    od._check()


def test_setitem():
    od = OrderedDict()
    od['alice'] = 0
    od['bob'] = 1
    od['carol'] = 2
    assert len(od) == 3
    od['carol'] = 2
    assert len(od) == 3
    od._check()


def test_delitem():
    od = OrderedDict(pairs)
    assert len(od) == 10
    for value in range(10):
        del od[value]
    assert len(od) == 0
    od._check()


def test_iter_reversed():
    od = OrderedDict([('b', 0), ('a', 1), ('c', 2)])
    assert list(od) == ['b', 'a', 'c']
    assert list(reversed(od)) == ['c', 'a', 'b']
    od._check()


def test_clear():
    od = OrderedDict(pairs)
    assert len(od) == 10
    od.clear()
    assert len(od) == 0
    od._check()


def test_popitem():
    od = OrderedDict(enumerate(range(10)))
    for num in reversed(range(10)):
        key, value = od.popitem()
        assert num == key == value
        od._check()

    od = OrderedDict(enumerate(range(10)))
    for num in range(10):
        key, value = od.popitem(last=False)
        assert num == key == value
        od._check()


def test_keys():
    od = OrderedDict(enumerate(range(10)))
    assert list(reversed(od.keys())) == list(reversed(range(10)))
    assert od.keys()[:3] == [0, 1, 2]
    od._check()


def test_items():
    items = list(enumerate(range(10)))
    od = OrderedDict(enumerate(range(10)))
    assert list(reversed(od.items())) == list(reversed(items))
    assert od.items()[:3] == [(0, 0), (1, 1), (2, 2)]
    od._check()


def test_values():
    od = OrderedDict(enumerate(range(10)))
    assert list(reversed(od.values())) == list(reversed(range(10)))
    assert od.values()[:3] == [0, 1, 2]
    od._check()


def test_iloc():
    od = OrderedDict(enumerate(range(10)))
    iloc = od.keys()
    for num in range(10):
        assert iloc[num] == num
    iloc[-1] == 9
    assert len(iloc) == 10
    od._check()


def test_pop():
    od = OrderedDict(enumerate(range(10)))
    for num in range(10):
        assert od.pop(num) == num
        od._check()
    assert od.pop(0, 'thing') == 'thing'
    assert od.pop(1, default='thing') == 'thing'
    od._check()


def test_pop_error():
    od = OrderedDict()
    with pytest.raises(KeyError):
        od.pop(0)


def test_setdefault():
    od = OrderedDict()
    od.setdefault(0, False)
    assert od[0] is False
    od.setdefault(1, default=True)
    assert od[1] is True
    od.setdefault(2)
    assert od[2] is None
    assert od.setdefault(0) is False
    assert od.setdefault(1) is True


def test_repr():
    od = OrderedDict()
    assert repr(od) == 'OrderedDict([])'
    assert str(od) == 'OrderedDict([])'


def test_reduce():
    od = OrderedDict(enumerate(range(10)))
    data = pickle.dumps(od)
    copy = pickle.loads(data)
    assert od == copy


def test_copy():
    od = OrderedDict(enumerate(range(10)))
    copy = od.copy()
    assert od == copy


def test_fromkeys():
    od = OrderedDict.fromkeys('abc')
    assert od == {'a': None, 'b': None, 'c': None}
    od._check()


def test_equality():
    od = OrderedDict.fromkeys('abc')
    assert od == {'a': None, 'b': None, 'c': None}
    assert od != {}
    assert od != OrderedDict()
    od._check()


def test_live_views_track_mapping():
    od = OrderedDict(enumerate(range(3)))
    keys, values, items = od.keys(), od.values(), od.items()
    od[3] = 3
    assert list(keys) == [0, 1, 2, 3]
    assert list(values) == [0, 1, 2, 3]
    assert list(items) == [(0, 0), (1, 1), (2, 2), (3, 3)]
    del od[0]
    assert list(keys) == [1, 2, 3]
    assert values[0] == 1
    od[1] = 10
    assert values[0] == 10 and items[0] == (1, 10)
    assert 1 in keys and 0 not in keys
    assert 10 in values and 0 not in values
    assert (1, 10) in items and (0, 0) not in items
    assert (9, 9) not in items
    od._check()


def test_view_sequence_fetch_hook():
    from sortedcollections.ordereddict import _ViewSequence
    view = _ViewSequence.__new__(_ViewSequence)
    with pytest.raises(NotImplementedError):
        view._fetch(0)


def test_batch_views_share_row_positions():
    od = OrderedDict([('a', 1), ('b', 2), ('c', 3), ('d', 4)])
    with od.export_batch():
        keys = od.keys()
        values = od.values()
        items = od.items()
        rows = [(keys[index], items[index], values[index])
                for index in range(len(keys))]
        assert rows == [
            ('a', ('a', 1), 1),
            ('b', ('b', 2), 2),
            ('c', ('c', 3), 3),
            ('d', ('d', 4), 4),
        ]
        assert keys[1:3] == ['b', 'c']
        assert values[1:3] == [2, 3]
        assert items[1:3] == [('b', 2), ('c', 3)]
        assert list(reversed(values)) == [4, 3, 2, 1]


def test_batch_value_reassignment_keeps_row():
    od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])
    with od.export_batch():
        keys, values, items = od.keys(), od.values(), od.items()
        od['b'] = 20
        assert keys[1] == 'b'
        assert values[1] == 20
        assert items[1] == ('b', 20)
        assert 20 in values and ('b', 20) in items
        assert len(keys) == 3


def test_batch_delete_and_reinsert_same_key():
    od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])
    with od.export_batch():
        keys, values, items = od.keys(), od.values(), od.items()
        first_key = keys[0]
        del od['a']
        od['a'] = 99
        assert first_key == 'a'
        assert list(keys) == ['b', 'c']
        assert list(values) == [2, 3]
        assert list(items) == [('b', 2), ('c', 3)]
        assert 'a' not in keys
        assert 99 not in values
        assert ('a', 99) not in items
    assert list(od.keys()) == ['b', 'c', 'a']
    assert od['a'] == 99
    od._check()


def test_batch_slice_pins_only_involved_rows():
    od = OrderedDict(enumerate(range(6)))
    with od.export_batch(slice(1, 4)):
        keys, values = od.keys(), od.values()
        assert len(keys) == 3
        assert list(keys) == [1, 2, 3]
        od[2] = 200
        assert values[1] == 200
        del od[0]
        assert list(keys) == [1, 2, 3]
        od[4] = 40
        assert list(values) == [1, 200, 3]


def test_batch_deleted_pinned_row_compacts_views():
    od = OrderedDict(enumerate(range(5)))
    with od.export_batch():
        keys, values, items = od.keys(), od.values(), od.items()
        del od[1]
        del od[3]
        assert len(keys) == len(values) == len(items) == 3
        assert list(keys) == [0, 2, 4]
        assert list(values) == [0, 2, 4]
        assert list(items) == [(0, 0), (2, 2), (4, 4)]
        assert keys[2] == 4 and values[2] == 4
        with pytest.raises(IndexError):
            keys[3]


def test_batch_single_row_and_step_slice():
    od = OrderedDict(enumerate(range(6)))
    with od.export_batch(2) as batch:
        keys = od.keys()
        assert len(batch) == 1
        assert len(keys) == 1 and keys[0] == 2
        del od[2]
        assert len(batch) == 0
        assert list(keys) == []
    stepped = OrderedDict(enumerate(range(6)))
    with stepped.export_batch(slice(0, 6, 2)):
        assert list(stepped.keys()) == [0, 2, 4]
        assert list(stepped.values()) == [0, 2, 4]


def test_nested_batches_end_independently():
    od = OrderedDict(enumerate(range(5)))
    with od.export_batch():
        outer = od.keys()
        with od.export_batch(slice(0, 2)):
            inner = od.keys()
            assert list(inner) == [0, 1]
            assert len(outer) == 5
        assert len(inner) == 2
        assert list(od.keys()) == [0, 1, 2, 3, 4]
    assert list(od.keys()) == [0, 1, 2, 3, 4]


def test_batch_view_protocols():
    od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])
    with od.export_batch(slice(0, 2)):
        keys, values, items = od.keys(), od.values(), od.items()
        assert 'a' in keys and 'z' not in keys
        assert 1 in values and 9 not in values
        assert ('a', 1) in items and ('a', 2) not in items
        assert ('z', 1) not in items
        assert keys.index('b') == 1
        assert keys.count('a') == 1
        assert list(reversed(keys)) == ['b', 'a']
        assert list(keys[:1]) == ['a']


def test_batch_released_on_exception():
    od = OrderedDict(enumerate(range(3)))
    with pytest.raises(RuntimeError):
        with od.export_batch():
            raise RuntimeError('boom')
    assert od._batch is None
    assert list(od.keys()) == [0, 1, 2]


def test_batch_clear():
    od = OrderedDict(enumerate(range(4)))
    with od.export_batch(slice(1, 3)):
        keys, values = od.keys(), od.values()
        od.clear()
        assert len(keys) == 0
        assert list(keys) == [] and list(values) == []
    assert len(od) == 0
    od._check()
