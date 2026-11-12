"""Sorted collections recipes implementations.

"""

from collections import abc
from copy import deepcopy
from itertools import chain, count

from sortedcontainers import SortedDict, SortedKeyList, SortedSet
from sortedcontainers.sortedlist import recursive_repr


# Sentinel marking a row that withdraws an existing key from a revision.
REMOVE = object()


class RevisionError(Exception):
    """Raised when a row of a revision order cannot be prepared.

    Sort keys are evaluated while the revision order is built. When the key
    function raises for a row, the original exception is preserved as the
    :attr:`__cause__` and the offending row can be located through
    :attr:`row` and :attr:`key`.

    """

    def __init__(self, row, key, cause):
        self.row = row
        self.key = key
        self.__cause__ = cause
        message = 'revision row {0} for key {1!r} failed: {2}'.format(
            row, key, cause
        )
        super().__init__(message)


class _RevisionOrder:
    """Deferred revision order committed as a single transaction."""

    def __init__(self, mapping, new_items, new_list):
        self._mapping = mapping
        self._new_items = new_items
        self._new_list = new_list
        self._closed = False

    def __enter__(self):
        if self._closed:
            raise RuntimeError('revision order may only be committed once')
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        if _exc_type is None:
            self.commit()
        self._closed = True
        return False

    def commit(self):
        "Swap in the prepared post-revision state."
        if self._closed:
            raise RuntimeError('revision order may only be committed once')
        self._closed = True
        self._mapping._install(self._new_items, self._new_list)


class _TransactionalSortedDict(SortedDict):
    """Base for sorted dicts whose revisions commit atomically.

    A revision may change existing values, add new keys or withdraw existing
    keys. Sort keys are evaluated exactly once, while the revision order is
    built, and each one is tagged with its source row. The order is applied
    to a private copy of the sorted list first, so a failure when sort keys
    are compared leaves the live dictionary, its rank ranges and any views
    already handed out untouched.

    """

    def _make_sort_key(self, key, value):
        "Return the comparable for the ``(key, value)`` item pair."
        raise NotImplementedError

    @staticmethod
    def _unpack_pair(row, pair):
        try:
            key, value = pair
        except (TypeError, ValueError) as cause:
            raise TypeError(
                'cannot interpret revision row {0} as a (key, value) '
                'pair'.format(row)
            ) from cause
        return key, value

    def _eval_sort_key(self, row, key, value, wrap_errors):
        try:
            return self._make_sort_key(key, value)
        except Exception as cause:
            if wrap_errors:
                raise RevisionError(row, key, cause) from cause
            raise

    def _prepare_rows(self, pairs, wrap_errors):
        "Validate every row and evaluate its new sort key exactly once."
        if isinstance(pairs, abc.Mapping):
            pairs = pairs.items()

        rows = []
        seen = set()

        for row, pair in enumerate(pairs):
            key, value = self._unpack_pair(row, pair)

            if key in seen:
                raise ValueError(
                    'key {0!r} appears more than once in the revision; '
                    'duplicate at row {1}'.format(key, row)
                )
            seen.add(key)

            if value is REMOVE:
                if key not in self:
                    cause = KeyError(key)
                    if wrap_errors:
                        raise RevisionError(row, key, cause) from cause
                    raise cause
                rows.append((row, 'remove', key, REMOVE, None))
            else:
                sort_key = self._eval_sort_key(row, key, value, wrap_errors)
                operation = 'change' if key in self else 'add'
                rows.append((row, operation, key, value, sort_key))

        return rows

    def _prepare_revision(self, pairs, wrap_errors):
        """Build the post-revision state without touching the live mapping.

        Rows are validated and every new sort key is evaluated once in
        :meth:`_prepare_rows`. A private sorted list then replays the rows
        in order using the precomputed sort keys, so a comparison failure is
        attributed to the row being replayed and leaves the live dictionary
        and its blocks untouched.

        """
        rows = self._prepare_rows(pairs, wrap_errors)
        if not rows:
            return dict.copy(self), self._list

        old_list = self._list
        old_key_values = chain.from_iterable(old_list._keys)
        sort_keys = dict(zip(old_list, old_key_values))
        new_items = dict.copy(self)

        def lookup(key):
            return sort_keys[key]

        new_list = SortedKeyList(key=lookup)
        new_list._load = old_list._load
        new_list._update(old_list)

        for row, operation, key, value, sort_key in rows:
            if operation in ('remove', 'change'):
                new_list.remove(key)

            if operation == 'remove':
                del new_items[key]
                del sort_keys[key]
                continue

            new_items[key] = value
            sort_keys[key] = sort_key
            try:
                new_list.add(key)
            except Exception as cause:
                if wrap_errors:
                    raise RevisionError(row, key, cause) from cause
                raise

        new_list._key = self._key
        return new_items, new_list

    def _install(self, new_items, new_list):
        "Swap in a fully prepared post-revision state."
        dict.clear(self)
        dict.update(self, new_items)
        self._list = new_list
        self._bind_list_methods()

    def _bind_list_methods(self):
        "Rebind cached sorted list methods after swapping the sorted list."
        _list = self._list
        self._list_add = _list.add
        self._list_clear = _list.clear
        self._list_iter = _list.__iter__
        self._list_reversed = _list.__reversed__
        self._list_pop = _list.pop
        self._list_remove = _list.remove
        self._list_update = _list.update

        self.bisect_left = _list.bisect_left
        self.bisect = _list.bisect_right
        self.bisect_right = _list.bisect_right
        self.index = _list.index
        self.irange = _list.irange
        self.islice = _list.islice
        self._reset = _list._reset

        self.bisect_key_left = _list.bisect_key_left
        self.bisect_key_right = _list.bisect_key_right
        self.bisect_key = _list.bisect_key
        self.irange_key = _list.irange_key

    def revision(self, changes=()):
        """Build a revision order from ``(key, value)`` rows.

        Rows may add new keys, change existing keys or, when the value is
        the :data:`REMOVE` sentinel, withdraw a key. Each key may appear at
        most once in the order. The order is committed atomically, either as
        a context manager or through :meth:`commit`::

            with mapping.revision([('a', 1), ('b', REMOVE)]):
                pass

        Sort keys are evaluated while the order is built, so a failing row
        raises :class:`RevisionError` tagged with its source row before the
        dictionary is touched. The order captures the mapping state at build
        time and is committed by swapping in the prepared state.

        """
        new_items, new_list = self._prepare_revision(changes, wrap_errors=True)
        return _RevisionOrder(self, new_items, new_list)

    def __setitem__(self, key, value):
        "``mapping[key] = value``"
        new_items, new_list = self._prepare_revision(
            [(key, value)], wrap_errors=False
        )
        self._install(new_items, new_list)

    _setitem = __setitem__

    def __delitem__(self, key):
        "``del mapping[key]``"
        if key not in self:
            raise KeyError(key)
        new_items, new_list = self._prepare_revision(
            [(key, REMOVE)], wrap_errors=False
        )
        self._install(new_items, new_list)

    def update(self, *args, **kwargs):
        """Update the mapping as one atomic revision.

        Overwrites existing items. Arguments follow the same mapping,
        iterable of pairs and keyword argument rules as :class:`dict`.

        """
        pairs = dict(*args, **kwargs).items()
        new_items, new_list = self._prepare_revision(pairs, wrap_errors=True)
        self._install(new_items, new_list)

    _update = update

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        self[key] = default
        return default


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


class ItemSortedDict(_TransactionalSortedDict):
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

    def _make_sort_key(self, key, value):
        "Return ``func(key, value)`` for the item pair."
        return self._func(key, value)

    def copy(self):
        "Return shallow copy of the mapping."
        return self.__class__(self._func, iter(self.items()))

    __copy__ = copy

    def __deepcopy__(self, memo):
        items = (deepcopy(item, memo) for item in self.items())
        return self.__class__(self._func, items)


class ValueSortedDict(_TransactionalSortedDict):
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

    def _make_sort_key(self, key, value):
        "Return the value, or the key function applied to it."
        if self._func is None:
            return value
        return self._func(value)

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
