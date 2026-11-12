"""Sorted collections recipes implementations.

"""

from collections import abc
from copy import deepcopy
from itertools import count

from sortedcontainers import SortedDict, SortedKeyList, SortedSet
from sortedcontainers.sortedlist import recursive_repr


class IndexableDict(SortedDict):
    """Dictionary that supports numerical indexing.

    Keys are numerically indexable using dict views. For example::

        >>> indexable_dict = IndexableDict.fromkeys('abcde')
        >>> keys = indexable_dict.keys()
        >>> sorted(keys[:]) == ['a', 'b', 'c', 'd', 'e']
        True

    The dict views support the sequence abstract base class.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(hash, *args, **kwargs)


class IndexableSet(SortedSet):
    """Set that supports numerical indexing.

    Values are numerically indexable. For example::

        >>> indexable_set = IndexableSet('abcde')
        >>> sorted(indexable_set[:]) == ['a', 'b', 'c', 'd', 'e']
        True

    `IndexableSet` implements the sequence abstract base class.

    """

    # pylint: disable=too-many-ancestors
    def __init__(self, *args, **kwargs):
        super().__init__(*args, key=hash, **kwargs)

    def __reduce__(self):
        return self.__class__, (set(self),)


class ItemSortedDict(SortedDict):
    """Sorted dictionary with key-function support for item pairs.

    Requires key function callable specified as the first argument. The
    callable must accept two arguments, key and value, and return a value used
    to determine the sort order. For example::

        def multiply(key, value):
            return key * value
        mapping = ItemSortedDict(multiply, [(3, 2), (4, 1), (2, 5)])
        list(mapping) == [4, 3, 2]

    Above, the key/value item pairs are ordered by ``key * value`` according to
    the callable given as the first argument.

    """

    def __init__(self, *args, **kwargs):
        assert args and callable(args[0])
        args = list(args)
        func = self._func = args[0]

        def key_func(key):
            "Apply key function to (key, value) item pair."
            return func(key, self[key])

        args[0] = key_func
        super().__init__(*args, **kwargs)

    def __delitem__(self, key):
        "``del mapping[key]``"
        if key not in self:
            raise KeyError(key)
        self._list_remove(key)
        dict.__delitem__(self, key)

    def __setitem__(self, key, value):
        "``mapping[key] = value``"
        if key in self:
            self._list_remove(key)
            dict.__delitem__(self, key)
        dict.__setitem__(self, key, value)
        self._list_add(key)

    _setitem = __setitem__

    def copy(self):
        "Return shallow copy of the mapping."
        return self.__class__(self._func, iter(self.items()))

    __copy__ = copy

    def __deepcopy__(self, memo):
        items = (deepcopy(item, memo) for item in self.items())
        return self.__class__(self._func, items)


class ValueSortedDict(SortedDict):
    """Sorted dictionary that maintains (key, value) item pairs sorted by value.

    - ``ValueSortedDict()`` -> new empty dictionary.

    - ``ValueSortedDict(mapping)`` -> new dictionary initialized from a mapping
      object's (key, value) pairs.

    - ``ValueSortedDict(iterable)`` -> new dictionary initialized as if via::

        d = ValueSortedDict()
        for k, v in iterable:
            d[k] = v

    - ``ValueSortedDict(**kwargs)`` -> new dictionary initialized with the
      name=value pairs in the keyword argument list.  For example::

        ValueSortedDict(one=1, two=2)

    An optional key function callable may be specified as the first
    argument. When so, the callable will be applied to the value of each item
    pair to determine the comparable for sort order as with Python's builtin
    ``sorted`` function.

    """

    def __init__(self, *args, **kwargs):
        args = list(args)
        if args and callable(args[0]):
            func = self._func = args[0]

            def key_func(key):
                "Apply key function to ``mapping[value]``."
                return func(self[key])

            args[0] = key_func
        else:
            self._func = None

            def key_func(key):
                "Return mapping value for key."
                return self[key]

            if args and args[0] is None:
                args[0] = key_func
            else:
                args.insert(0, key_func)
        super().__init__(*args, **kwargs)

    def __delitem__(self, key):
        "``del mapping[key]``"
        if key not in self:
            raise KeyError(key)
        self._list_remove(key)
        dict.__delitem__(self, key)

    def __setitem__(self, key, value):
        "``mapping[key] = value``"
        if key in self:
            self._list_remove(key)
            dict.__delitem__(self, key)
        dict.__setitem__(self, key, value)
        self._list_add(key)

    _setitem = __setitem__

    def copy(self):
        "Return shallow copy of the mapping."
        return self.__class__(self._func, iter(self.items()))

    __copy__ = copy

    def __reduce__(self):
        items = [(key, self[key]) for key in self._list]
        args = (self._func, items)
        return (self.__class__, args)

    @recursive_repr()
    def __repr__(self):
        items = ', '.join(f'{key!r}: {self[key]!r}' for key in self._list)
        return f'{self.__class__.__name__}({self._func!r}, {{{items}}})'


class OrderedSet(abc.MutableSet, abc.Sequence):
    """Like OrderedDict, OrderedSet maintains the insertion order of elements.

    For example::

        >>> ordered_set = OrderedSet('abcde')
        >>> list(ordered_set) == list('abcde')
        True
        >>> ordered_set = OrderedSet('edcba')
        >>> list(ordered_set) == list('edcba')
        True

    OrderedSet also implements the collections.Sequence interface.

    """

    # pylint: disable=too-many-ancestors
    def __init__(self, iterable=()):
        # pylint: disable=super-init-not-called
        self._keys = {}
        self._nums = SortedDict()
        self._keys_view = self._nums.keys()
        self._count = count()
        self |= iterable

    def __contains__(self, key):
        "``key in ordered_set``"
        return key in self._keys

    count = __contains__

    def __iter__(self):
        "``iter(ordered_set)``"
        return iter(self._nums.values())

    def __reversed__(self):
        "``reversed(ordered_set)``"
        _nums = self._nums
        for key in reversed(_nums):
            yield _nums[key]

    def __getitem__(self, index):
        "``ordered_set[index]`` -> element; lookup element at index."
        num = self._keys_view[index]
        return self._nums[num]

    def __len__(self):
        "``len(ordered_set)``"
        return len(self._keys)

    def index(self, value):
        "Return index of value."
        # pylint: disable=arguments-differ
        try:
            return self._keys[value]
        except KeyError:
            raise ValueError(f'{value!r} is not in {type(self).__name__}')

    def add(self, value):
        "Add element, value, to set."
        if value not in self._keys:
            num = next(self._count)
            self._keys[value] = num
            self._nums[num] = value

    def discard(self, value):
        "Remove element, value, from set if it is a member."
        num = self._keys.pop(value, None)
        if num is not None:
            del self._nums[num]

    def __repr__(self):
        "Text representation of set."
        return f'{type(self).__name__}({list(self)!r})'

    __str__ = __repr__


class SegmentList(SortedKeyList):
    """List that supports fast random insertion and deletion of elements.

    SegmentList is a special case of a SortedList initialized with a key
    function that always returns 0. As such, several SortedList methods are not
    implemented for SegmentList.

    """

    # pylint: disable=too-many-ancestors
    def __init__(self, iterable=()):
        super().__init__(iterable, self.zero)

    @staticmethod
    def zero(_):
        "Return 0."
        return 0

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            raise NotImplementedError
        pos, idx = self._pos(index)
        self._lists[pos][idx] = value

    def append(self, value):
        if self._len:
            pos = len(self._lists) - 1
            self._lists[pos].append(value)
            self._keys[pos].append(0)
            self._expand(pos)
        else:
            self._lists.append([value])
            self._keys.append([0])
            self._maxes.append(0)
        self._len += 1

    def extend(self, values):
        for value in values:
            self.append(value)

    def insert(self, index, value):
        if index == self._len:
            self.append(value)
            return
        pos, idx = self._pos(index)
        self._lists[pos].insert(idx, value)
        self._keys[pos].insert(idx, 0)
        self._expand(pos)
        self._len += 1

    def _replace_range(self, start, stop, values):
        "Replace rows ``range(start, stop)`` with the list ``values``."
        # Only the segments overlapping the range are staged and rebuilt;
        # segments outside the range keep their identity and are untouched.
        _lists = self._lists
        length = self._len
        count = len(values)

        if start == stop and not count:
            return

        if _lists:
            if start == length:
                pos_lo = len(_lists) - 1
                idx_lo = len(_lists[pos_lo])
            else:
                pos_lo, idx_lo = self._pos(start)
            if stop == length:
                pos_hi = len(_lists) - 1
                idx_hi = len(_lists[pos_hi])
            else:
                pos_hi, idx_hi = self._pos(stop)
            run = _lists[pos_lo][:idx_lo] + values + _lists[pos_hi][idx_hi:]
        else:
            pos_lo = pos_hi = 0
            run = list(values)

        load = self._load
        half = load >> 1

        # Widen the staged window by a neighboring segment when the run
        # alone would leave a segment smaller than half the load factor in
        # a non-last position.
        if 0 < len(run) < half and _lists:
            if pos_lo > 0:
                pos_lo -= 1
                run = _lists[pos_lo] + run
            if len(run) < half and pos_hi < len(_lists) - 1:
                pos_hi += 1
                run = run + _lists[pos_hi]

        # Split the run into balanced segments so every segment but the
        # last holds at least half the load factor.
        size = len(run)
        if size:
            num = -(-size // load)
            base, rem = divmod(size, num)
            new_lists = []
            offset = 0
            for _ in range(rem):
                new_lists.append(run[offset:offset + base + 1])
                offset += base + 1
            while offset < size:
                new_lists.append(run[offset:offset + base])
                offset += base
        else:
            new_lists = []

        _lists[pos_lo:pos_hi + 1] = new_lists
        self._keys[pos_lo:pos_hi + 1] = [[0] * len(sub) for sub in new_lists]
        self._maxes[pos_lo:pos_hi + 1] = [0] * len(new_lists)
        self._len = length - (stop - start) + count

        # Invalidate the positional index like ``_expand`` does; it is
        # rebuilt lazily on the next positional lookup.
        del self._index[:]
        self._offset = 0

    def _replace_rows(self, rows, values):
        "Assign ``values`` at ``rows`` in order; row positions are unchanged."
        _lists = self._lists
        _pos = self._pos
        for row, value in zip(rows, values):
            pos, idx = _pos(row)
            _lists[pos][idx] = value

    def reverse(self):
        values = list(self)
        values.reverse()
        self.clear()
        self.extend(values)

    def sort(self, key=None, reverse=False):
        "Stable sort in place."
        values = sorted(self, key=key, reverse=reverse)
        self.clear()
        self.extend(values)

    def _not_implemented(self, *args, **kwargs):
        "Not implemented."
        raise NotImplementedError

    add = _not_implemented
    bisect = _not_implemented
    bisect_left = _not_implemented
    bisect_right = _not_implemented
    bisect_key = _not_implemented
    bisect_key_left = _not_implemented
    bisect_key_right = _not_implemented
    irange = _not_implemented
    irange_key = _not_implemented
    update = _not_implemented


def replace_column_slices(columns, selection, sources):
    """Replace a frozen row selection across row-aligned segment-list columns.

    A batch of columnar samples stored in :class:`SegmentList` columns is
    aligned by physical row number: every column has the same length and row
    *r* refers to the same record in each column. This function replaces the
    rows selected by the slice ``selection`` in every column at once, using
    one source per column from ``sources``.

    Each source may reference another column of the same table (including a
    column being replaced) or be a one-shot generator. All sources are fully
    read before any column is written, so overlapping selections, negative
    strides and self-references behave as if the complete old selection were
    read first and only then written.

    The edit is atomic across columns: if the per-column target row counts
    differ, or a source (such as a generator) fails while being read, no
    column is modified and the alignment between columns is preserved. For
    extended slices (step not equal to 1) the common source count must match
    the number of selected rows exactly, as with list slice assignment.

    Only the row segments overlapping the selection are staged and rebuilt;
    the rest of each column is left in place rather than expanding the whole
    table.

    Returns a dict mapping each original row number to its new position.
    When the replacement changes the row count, rows removed by the
    selection map to ``None`` and later rows shift accordingly; otherwise
    (including all extended slices) the mapping is the identity.

    >>> first = SegmentList(range(5))
    >>> second = SegmentList(range(100, 105))
    >>> mapping = replace_column_slices([first, second], slice(None), [second, first])
    >>> list(first)
    [100, 101, 102, 103, 104]
    >>> list(second)
    [0, 1, 2, 3, 4]
    >>> mapping == {row: row for row in range(5)}
    True

    :param columns: sequence of row-aligned :class:`SegmentList` columns
    :param selection: slice selecting the frozen rows to replace
    :param sources: sequence of replacement iterables, one per column
    :return: dict mapping original row numbers to new positions

    """
    columns = list(columns)
    sources = list(sources)

    if len(columns) != len(sources):
        raise ValueError('columns and sources must pair up one-to-one')

    for column in columns:
        if not isinstance(column, SegmentList):
            raise TypeError('columns must be SegmentList instances')

    if not columns:
        return {}

    if not isinstance(selection, slice):
        raise TypeError('selection must be a slice')

    length = len(columns[0])
    if any(len(column) != length for column in columns):
        raise ValueError('columns are not aligned by physical row number')

    start, stop, step = selection.indices(length)
    rows = range(start, stop, step)
    target = len(rows)

    # Snapshot every source before any column is written. Reads observe the
    # pre-change table; a failing source aborts the edit before any mutation.
    staged = [list(source) for source in sources]

    counts = set(map(len, staged))
    if len(counts) > 1:
        raise ValueError('target row counts differ between columns')
    count = counts.pop()

    if step != 1 and count != target:
        raise ValueError(
            f'attempt to assign sequence of size {count} '
            f'to extended slice of size {target}'
        )

    if step == 1:
        for column, values in zip(columns, staged):
            column._replace_range(start, stop, values)
        delta = count - target
        if delta == 0:
            mapping = {row: row for row in range(length)}
        else:
            mapping = {}
            for row in range(length):
                if row < start:
                    mapping[row] = row
                elif row < stop:
                    mapping[row] = None
                else:
                    mapping[row] = row + delta
    else:
        for column, values in zip(columns, staged):
            column._replace_rows(rows, values)
        mapping = {row: row for row in range(length)}

    return mapping
