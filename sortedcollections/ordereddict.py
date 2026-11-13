"""Ordered dictionary implementation.

"""

from contextlib import contextmanager
from itertools import count
from operator import eq

from sortedcontainers import SortedDict
from sortedcontainers.sortedlist import recursive_repr

from .recipes import abc

NONE = object()


class _Batch:
    """Shared row-position context for one export.

    A batch records the insertion numbers of the rows involved in one
    export so that every view opened inside the batch resolves column
    names, key/value pairs and column values through the same row
    identities. Reassigning a value keeps the row; deleting a key and
    re-inserting the same key allocates a new row the batch cannot see,
    so a column name fetched earlier can never be paired with it.

    Rows deleted while the batch is active drop out of every view,
    keeping length, indexing, slicing and iteration consistent. The
    surviving rows are cached and only recomputed after the mapping
    reports a structural change.

    """

    # pylint: disable=protected-access
    __slots__ = ('_mapping', '_pinned', '_alive', '_version')

    def __init__(self, mapping, nums):
        self._mapping = mapping
        self._pinned = tuple(nums)
        self._alive = list(self._pinned)
        self._version = -1

    def _refresh(self):
        "Recache the pinned rows that still exist in the mapping."
        mapping = self._mapping
        if mapping._version == self._version:
            return
        nums = mapping._nums
        self._alive = [num for num in self._pinned if num in nums]
        self._version = mapping._version

    def __len__(self):
        "Number of pinned rows still present in the mapping."
        self._refresh()
        return len(self._alive)

    def row_num(self, index):
        "Return a surviving pinned insertion number by position."
        self._refresh()
        return self._alive[index]

    def row_nums(self, index):
        "Resolve a slice to the surviving pinned insertion numbers."
        self._refresh()
        return self._alive[index]


class _ViewSequence(abc.Sequence):
    """Shared numeric row lookup used by the mapping views.

    Views without a batch stay live: positions are read straight from
    the mapping on every access. Views holding a batch resolve positions
    through the pinned insertion numbers so an export keeps a stable row
    identity for its whole lifetime.

    """

    # pylint: disable=protected-access
    _batch = None

    def __init__(self, mapping, batch=None):
        super().__init__(mapping)
        self._batch = batch

    def _row_num(self, index):
        "Resolve a positional index to a row insertion number."
        if self._batch is None:
            return self._mapping._nums._list[index]
        return self._batch.row_num(index)

    def _row_nums(self, index):
        "Resolve a slice to the row insertion numbers involved."
        if self._batch is None:
            return self._mapping._nums._list[index]
        return self._batch.row_nums(index)

    def _fetch(self, num):
        "Return the view-specific data held by row number ``num``."
        raise NotImplementedError

    def __getitem__(self, index):
        "``view[index]``"
        if isinstance(index, slice):
            return [self._fetch(num) for num in self._row_nums(index)]
        return self._fetch(self._row_num(index))

    def __len__(self):
        "``len(view)``"
        if self._batch is None:
            return len(self._mapping)
        return len(self._batch)

    def __iter__(self):
        "Iterate the visible rows in position order."
        for num in self._row_nums(slice(None)):
            yield self._fetch(num)

    def __reversed__(self):
        "Iterate the visible rows in reverse position order."
        for num in reversed(self._row_nums(slice(None))):
            yield self._fetch(num)


class KeysView(_ViewSequence, abc.KeysView):
    "Read-only view of mapping keys."
    # noqa pylint: disable=too-few-public-methods,protected-access,too-many-ancestors

    def _fetch(self, num):
        return self._mapping._nums[num]

    def __contains__(self, key):
        "``key in keys_view``"
        if self._batch is None:
            return key in self._mapping
        return any(key == candidate for candidate in self)


class ItemsView(_ViewSequence, abc.ItemsView):
    "Read-only view of mapping items."
    # noqa pylint: disable=too-few-public-methods,protected-access,too-many-ancestors

    def _fetch(self, num):
        key = self._mapping._nums[num]
        return key, self._mapping[key]

    def __contains__(self, item):
        "``(key, value) in items_view``"
        key, value = item
        if self._batch is None:
            try:
                candidate = self._mapping[key]
            except KeyError:
                return False
            return candidate is value or candidate == value
        return any((key, value) == pair for pair in self)


class ValuesView(_ViewSequence, abc.ValuesView):
    "Read-only view of mapping values."
    # noqa pylint: disable=too-few-public-methods,protected-access,too-many-ancestors

    def _fetch(self, num):
        return self._mapping[self._mapping._nums[num]]

    def __contains__(self, value):
        "``value in values_view``"
        return any(value == candidate for candidate in self)


class OrderedDict(dict):
    """Dictionary that remembers insertion order and is numerically indexable.

    Keys are numerically indexable using dict views. For example::

        >>> ordered_dict = OrderedDict.fromkeys('abcde')
        >>> keys = ordered_dict.keys()
        >>> keys[0]
        'a'
        >>> keys[-2:]
        ['d', 'e']

    The dict views support the sequence abstract base class.

    An export batch pins row positions for the views opened inside it::

        >>> ordered_dict = OrderedDict([('a', 1), ('b', 2), ('c', 3)])
        >>> with ordered_dict.export_batch(slice(0, 2)):
        ...     keys = ordered_dict.keys()
        ...     values = ordered_dict.values()
        ...     ordered_dict['a'] = 10
        ...     keys[:], values[:]
        (['a', 'b'], [10, 2])

    """

    # pylint: disable=super-init-not-called
    def __init__(self, *args, **kwargs):
        self._keys = {}
        self._nums = SortedDict()
        self._keys_view = self._nums.keys()
        self._count = count()
        self._batches = []
        self._version = 0
        self.update(*args, **kwargs)

    @property
    def _batch(self):
        "Innermost active export batch, or ``None`` outside a batch."
        batches = self._batches
        return batches[-1] if batches else None

    @contextmanager
    def export_batch(self, index=slice(None)):
        """Open an export batch pinning row positions for the views.

        ``index`` may be a slice, in which case only the rows reached by
        the slice are pinned, or a single positional index pinning one
        row. Every view created while the batch is active reads the same
        row-position context. Batches may be nested and each batch ends
        independently when its ``with`` block exits.

        """
        if isinstance(index, slice):
            nums = self._nums._list[index]
        else:
            nums = (self._nums._list[index],)
        batch = _Batch(self, nums)
        self._batches.append(batch)
        try:
            yield batch
        finally:
            self._batches.remove(batch)

    def __setitem__(self, key, value, dict_setitem=dict.__setitem__):
        "``ordered_dict[key] = value``"
        if key not in self:
            num = next(self._count)
            self._keys[key] = num
            self._nums[num] = key
        dict_setitem(self, key, value)

    def __delitem__(self, key, dict_delitem=dict.__delitem__):
        "``del ordered_dict[key]``"
        dict_delitem(self, key)
        num = self._keys.pop(key)
        del self._nums[num]
        self._version += 1

    def __iter__(self):
        "``iter(ordered_dict)``"
        return iter(self._nums.values())

    def __reversed__(self):
        "``reversed(ordered_dict)``"
        nums = self._nums
        for key in reversed(nums):
            yield nums[key]

    def clear(self, dict_clear=dict.clear):
        "Remove all items from mapping."
        dict_clear(self)
        self._keys.clear()
        self._nums.clear()
        self._version += 1

    def popitem(self, last=True):
        """Remove and return (key, value) item pair.

        Pairs are returned in LIFO order if last is True or FIFO order if
        False.

        """
        index = -1 if last else 0
        num = self._keys_view[index]
        key = self._nums[num]
        value = self.pop(key)
        return key, value

    update = __update = abc.MutableMapping.update

    def keys(self):
        "Return set-like and sequence-like view of mapping keys."
        return KeysView(self, self._batch)

    def items(self):
        "Return set-like and sequence-like view of mapping items."
        return ItemsView(self, self._batch)

    def values(self):
        "Return set-like and sequence-like view of mapping values."
        return ValuesView(self, self._batch)

    def pop(self, key, default=NONE):
        """Remove given key and return corresponding value.

        If key is not found, default is returned if given, otherwise raise
        KeyError.

        """
        if key in self:
            value = self[key]
            del self[key]
            return value
        if default is NONE:
            raise KeyError(key)
        return default

    def setdefault(self, key, default=None):
        """Return ``mapping.get(key, default)``, also set ``mapping[key] = default`` if
        key not in mapping.

        """
        if key in self:
            return self[key]
        self[key] = default
        return default

    @recursive_repr()
    def __repr__(self):
        "Text representation of mapping."
        return f'{self.__class__.__name__}({list(self.items())!r})'

    __str__ = __repr__

    def __reduce__(self):
        "Support for pickling serialization."
        return (self.__class__, (list(self.items()),))

    def copy(self):
        "Return shallow copy of mapping."
        return self.__class__(self)

    @classmethod
    def fromkeys(cls, iterable, value=None):
        """Return new mapping with keys from iterable.

        If not specified, value defaults to None.

        """
        return cls((key, value) for key in iterable)

    def __eq__(self, other):
        "Test self and other mapping for equality."
        if isinstance(other, OrderedDict):
            return dict.__eq__(self, other) and all(map(eq, self, other))
        return dict.__eq__(self, other)

    __ne__ = abc.MutableMapping.__ne__

    def _check(self):
        "Check consistency of internal member variables."
        # pylint: disable=protected-access
        keys = self._keys
        nums = self._nums

        for key, value in keys.items():
            assert nums[value] == key

        nums._check()
