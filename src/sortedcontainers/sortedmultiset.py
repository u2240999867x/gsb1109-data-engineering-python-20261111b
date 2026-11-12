"""Sorted Multiset
==================

Sorted multiset with weighted multiplicities.

A :class:`SortedMultiset` stores each distinct value once together with its
multiplicity instead of expanding repeated values into individual entries. A
Fenwick (binary indexed) tree built over the multiplicities answers weighted
rank queries -- the value at rank *k* and the rank interval of a status -- in
`O(log m)` time where *m* is the number of distinct values, independent of the
total number of stored occurrences. Point adjustments touch only `O(log m)`
tree nodes; the prefix-sum tree is never rebuilt for a multiplicity-only
change.

Example::

    >>> ms = SortedMultiset([(200, 5_000_000), (404, 12), (500, 3)])
    >>> len(ms)
    5000015
    >>> ms[0]
    200
    >>> ms.rank_range(404)
    (5000000, 5000012)
    >>> ms.percentile(99.9)
    200

"""

from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from itertools import repeat
from math import ceil


class NegativeCountError(ValueError):
    """Raised when an adjustment would drive a multiplicity below zero.

    The :attr:`offending` mapping reports each value together with the
    negative multiplicity that would have resulted.

    """

    def __init__(self, message='multiplicity would be negative', offending=None):
        super().__init__(message)
        self.offending = dict(offending or {})


class _FenwickTree:
    """Fenwick (binary indexed) prefix-sum tree over bucket weights.

    Point updates, prefix sums and weighted rank selection all run in
    `O(log n)` time and touch only `O(log n)` nodes. The :attr:`accesses`
    counter records how many tree nodes were visited since the last reset;
    it is a diagnostic hook for logarithmic-complexity checks.

    """

    __slots__ = ('tree', 'accesses')

    def __init__(self, size=0):
        self.tree = [0] * (size + 1)
        self.accesses = 0

    @classmethod
    def from_counts(cls, counts):
        """Build a tree in linear time from an aligned count array."""
        tree = [0] * (len(counts) + 1)

        for index, count in enumerate(counts, 1):
            tree[index] = count

        for index in range(1, len(tree)):
            parent = index + (index & -index)
            if parent < len(tree):
                tree[parent] += tree[index]

        instance = cls.__new__(cls)
        instance.tree = tree
        instance.accesses = 0
        return instance

    def add(self, index, delta):
        """Add `delta` to the weight at 0-based `index` in `O(log n)`."""
        position = index + 1
        tree = self.tree
        bound = len(tree)

        while position < bound:
            tree[position] += delta
            position += position & -position
            self.accesses += 1

    def prefix_sum(self, count):
        """Sum the weights of the first `count` buckets in `O(log n)`."""
        position = count
        tree = self.tree
        total = 0

        while position:
            total += tree[position]
            position -= position & -position
            self.accesses += 1

        return total

    def select(self, rank):
        """Return the 0-based bucket index containing weighted `rank`.

        Returns the smallest index *i* for which the prefix sum through
        bucket *i* is greater than `rank`. Caller must ensure
        `0 <= rank < total weight`.

        """
        size = len(self.tree) - 1
        position = 0
        step = 1 << (size.bit_length() - 1) if size else 0
        tree = self.tree

        while step:
            nxt = position + step

            if nxt <= size and tree[nxt] <= rank:
                rank -= tree[nxt]
                position = nxt

            step >>= 1
            self.accesses += 1

        return position

    def reset_accesses(self):
        """Reset the diagnostic node-access counter."""
        self.accesses = 0


class SortedMultisetView:
    """Read-only weighted distribution, typically captured by :meth:`snapshot`.

    A view owns private copies of the distinct values, multiplicities and
    prefix-sum tree taken at capture time. Later adjustments to the source
    :class:`SortedMultiset` never affect the view, so ranks captured before a
    modification keep answering according to the original distribution.

    Views support the full read-only query surface but none of the mutating
    methods.

    """

    def __init__(self, values, counts):
        self._values = list(values)
        self._counts = list(counts)
        self._tree = _FenwickTree.from_counts(self._counts)
        self._total = sum(self._counts)

    def __len__(self):
        """Total number of stored occurrences (sum of multiplicities)."""
        return self._total

    def distinct_count(self):
        """Number of distinct values (number of weighted buckets)."""
        return len(self._values)

    def __contains__(self, value):
        if not self._values:
            return False

        index = bisect_left(self._values, value)
        return index < len(self._values) and self._values[index] == value

    def multiplicity(self, value):
        """Multiplicity of `value`, zero when it is absent."""
        if not self._values:
            return 0

        index = bisect_left(self._values, value)

        if index < len(self._values) and self._values[index] == value:
            return self._counts[index]

        return 0

    count = multiplicity

    def __iter__(self):
        """Iterate distinct values in sorted order (use :meth:`elements` to expand)."""
        return iter(self._values)

    def __reversed__(self):
        return reversed(self._values)

    def items(self):
        """Iterate ``(value, multiplicity)`` pairs in sorted order."""
        return zip(self._values, self._counts, strict=True)

    def elements(self):
        """Iterate every occurrence, repeating each value by its multiplicity."""
        for value, multiplicity in zip(self._values, self._counts, strict=True):
            yield from repeat(value, multiplicity)

    def bisect_left(self, value):
        """Weighted rank immediately before any occurrence of `value`.

        Equivalent to :meth:`bisect_left` on the fully expanded sorted
        sequence, computed in `O(log m)` time.

        """
        index = bisect_left(self._values, value)
        return self._tree.prefix_sum(index)

    def bisect_right(self, value):
        """Weighted rank immediately after every occurrence of `value`.

        Equivalent to :meth:`bisect_right` on the fully expanded sorted
        sequence, computed in `O(log m)` time.

        """
        index = bisect_right(self._values, value)
        return self._tree.prefix_sum(index)

    def rank_range(self, value):
        """Return the half-open weighted rank interval ``(lo, hi)`` of `value`.

        Ranks `lo` through `hi - 1` are occupied by `value`. Raises
        :exc:`KeyError` when `value` is absent.

        >>> ms = SortedMultiset([(1, 2), (3, 4)])
        >>> ms.rank_range(3)
        (2, 6)

        """
        lo = self.bisect_left(value)
        hi = self.bisect_right(value)

        if lo == hi:
            raise KeyError(value)

        return lo, hi

    def __getitem__(self, index):
        """Value at weighted rank `index`; also supports slices.

        >>> ms = SortedMultiset([('a', 3), ('b', 2)])
        >>> ms[3]
        'b'
        >>> ms[-1]
        'b'

        """
        if isinstance(index, slice):
            start, stop, step = index.indices(self._total)
            select = self._tree.select
            values = self._values
            return [values[select(rank)] for rank in range(start, stop, step)]

        if index < 0:
            index += self._total

        if not 0 <= index < self._total:
            raise IndexError('multiset index out of range')

        return self._values[self._tree.select(index)]

    def quantile(self, q):
        """Nearest-rank quantile for `q` in the closed interval ``[0, 1]``.

        The returned value is the one at rank ``ceil(q * n) - 1`` (with
        ``q = 0`` mapping to the first rank), matching nearest-rank
        percentile conventions.

        """
        if not 0 <= q <= 1:
            raise ValueError('quantile must be between 0 and 1')

        if not self._total:
            raise IndexError('quantile of empty multiset')

        rank = 0 if q == 0 else min(self._total - 1, ceil(q * self._total) - 1)
        return self[rank]

    def percentile(self, p):
        """Nearest-rank percentile for `p` in ``[0, 100]``."""
        if not 0 <= p <= 100:
            raise ValueError('percentile must be between 0 and 100')

        return self.quantile(p / 100)

    def snapshot(self):
        """Return a read-only view; a view is already immutable."""
        return self

    def __eq__(self, other):
        if isinstance(other, SortedMultisetView):
            return self._values == other._values and self._counts == other._counts

        if isinstance(other, Sequence):
            if len(other) != self._total:
                return False

            return all(
                left == right
                for left, right in zip(self.elements(), other, strict=True)
            )

        return NotImplemented

    def __repr__(self):
        body = ', '.join(f'{value!r}: {count}' for value, count in self.items())
        return f'{type(self).__name__}({{{body}}})'

    def _check(self):
        """Check invariants of the weighted distribution.

        Runtime complexity: `O(m)`.

        """
        values = self._values
        counts = self._counts

        assert len(values) == len(counts)
        assert all(isinstance(count, int) and count > 0 for count in counts)

        for index in range(1, len(values)):
            assert values[index - 1] < values[index]

        assert self._total == sum(counts)
        assert self._tree.prefix_sum(len(counts)) == self._total

        rebuilt = _FenwickTree.from_counts(counts)
        assert rebuilt.tree == self._tree.tree

        cumulative = 0

        for index, count in enumerate(counts):
            assert self._tree.prefix_sum(index + 1) == cumulative + count
            assert self._tree.select(cumulative) == index
            cumulative += count


class SortedMultiset(SortedMultisetView):
    """Sorted multiset storing multiplicities instead of expanded entries.

    Initialize from an iterable of ``(value, multiplicity)`` pairs or a
    mapping::

        >>> ms = SortedMultiset([(200, 5), (404, 2)])
        >>> ms = SortedMultiset({200: 5, 404: 2})

    Use :meth:`from_values` to build one from an ordinary iterable of values.

    Weighted queries run in `O(log m)` where *m* is the number of distinct
    values, regardless of the total multiplicity. Multiplicity adjustments
    touch only `O(log m)` prefix-sum nodes; :meth:`snapshot` captures an
    immutable view that keeps answering at the old ranks after the live
    multiset changes.

    """

    def __init__(self, pairs=None):
        self._values = []
        self._counts = []
        self._tree = _FenwickTree(0)
        self._total = 0

        if pairs is not None:
            aggregated = {}

            for value, count in self._pairs_of(pairs):
                self._require_positive_count(count)
                aggregated[value] = aggregated.get(value, 0) + count

            self._values = sorted(aggregated)
            self._counts = [aggregated[value] for value in self._values]
            self._rebuild_tree()
            self._total = sum(self._counts)

    @classmethod
    def from_values(cls, iterable):
        """Build a multiset from an iterable of individual values."""
        aggregated = {}

        for value in iterable:
            aggregated[value] = aggregated.get(value, 0) + 1

        return cls(aggregated.items())

    @staticmethod
    def _pairs_of(other):
        if isinstance(other, SortedMultisetView):
            return list(zip(other._values, other._counts, strict=True))

        if isinstance(other, Mapping):
            return list(other.items())

        return list(other)

    @staticmethod
    def _require_positive_count(count):
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError('multiplicity must be a positive integer')

    @staticmethod
    def _require_non_negative_count(count):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError('multiplicity must be a non-negative integer')

    def _rebuild_tree(self):
        self._tree = _FenwickTree.from_counts(self._counts)

    def _add_at(self, index, delta):
        self._counts[index] += delta
        self._tree.add(index, delta)
        self._total += delta

    def _insert_bucket(self, index, value, count):
        self._values.insert(index, value)
        self._counts.insert(index, count)
        self._rebuild_tree()
        self._total += count

    def _delete_bucket(self, index):
        count = self._counts[index]
        del self._values[index]
        del self._counts[index]
        self._rebuild_tree()
        self._total -= count

    def add(self, value, count=1):
        """Increase the multiplicity of `value` by `count`.

        A new distinct value inserts a bucket in `O(m)`; adjusting an
        existing value is a `O(log m)` point update with no prefix-sum
        rebuild.

        >>> ms = SortedMultiset()
        >>> ms.add(200, 5_000_000)
        >>> ms.add(200, 12)
        >>> ms.multiplicity(200)
        5000012

        """
        self._require_positive_count(count)

        index = bisect_left(self._values, value)

        if index < len(self._values) and self._values[index] == value:
            self._add_at(index, count)
        else:
            self._insert_bucket(index, value, count)

    def subtract(self, value, count=1):
        """Decrease the multiplicity of `value` by `count`.

        Raises :class:`NegativeCountError` without changing the multiset if
        the adjustment would make the bucket negative. A bucket reaching zero
        is removed.

        >>> ms = SortedMultiset([(200, 3)])
        >>> ms.subtract(200, 2)
        >>> ms.multiplicity(200)
        1

        """
        self._require_positive_count(count)

        index = bisect_left(self._values, value)
        current = (
            self._counts[index]
            if index < len(self._values) and self._values[index] == value
            else 0
        )

        if count > current:
            raise NegativeCountError(
                f'cannot subtract {count} from multiplicity {current} of {value!r}',
                offending={value: current - count},
            )

        if count == current:
            self._delete_bucket(index)
        else:
            self._add_at(index, -count)

    def adjust(self, value, delta):
        """Apply a single signed multiplicity delta atomically."""
        if not isinstance(delta, int) or isinstance(delta, bool):
            raise ValueError('delta must be an integer')

        if delta > 0:
            self.add(value, delta)
        elif delta < 0:
            self.subtract(value, -delta)

    def set_count(self, value, count):
        """Assign the multiplicity of `value`; zero removes the bucket."""
        self._require_non_negative_count(count)

        if count == 0:
            self.discard(value)
            return

        index = bisect_left(self._values, value)

        if index < len(self._values) and self._values[index] == value:
            self._add_at(index, count - self._counts[index])
        else:
            self._insert_bucket(index, value, count)

    def discard(self, value):
        """Remove the whole bucket of `value` if present."""
        index = bisect_left(self._values, value)

        if index < len(self._values) and self._values[index] == value:
            self._delete_bucket(index)

    def remove(self, value):
        """Remove the whole bucket of `value` or raise :exc:`KeyError`."""
        index = bisect_left(self._values, value)

        if index == len(self._values) or self._values[index] != value:
            raise KeyError(value)

        self._delete_bucket(index)

    def clear(self):
        """Remove every bucket."""
        self._values = []
        self._counts = []
        self._tree = _FenwickTree(0)
        self._total = 0

    def apply_deltas(self, deltas):
        """Apply signed ``value -> delta`` adjustments as one transaction.

        `deltas` is a mapping or iterable of ``(value, delta)`` pairs. Every
        bucket is previewed first: if any resulting multiplicity would be
        negative, :class:`NegativeCountError` is raised and nothing changes.
        On success all deltas take effect together.

        >>> ms = SortedMultiset([(1, 3), (2, 5)])
        >>> ms.apply_deltas({1: 2, 2: -4})
        >>> ms.multiplicity(1), ms.multiplicity(2)
        (5, 1)

        """
        normalized = dict(deltas)

        for value, delta in normalized.items():
            if not isinstance(delta, int) or isinstance(delta, bool):
                raise ValueError(f'delta for {value!r} must be an integer')

        offending = {}

        for value, delta in normalized.items():
            if delta >= 0:
                continue

            index = bisect_left(self._values, value)
            current = (
                self._counts[index]
                if index < len(self._values) and self._values[index] == value
                else 0
            )

            if current + delta < 0:
                offending[value] = current + delta

        if offending:
            raise NegativeCountError(offending=offending)

        for value in sorted(normalized):
            delta = normalized[value]

            if delta > 0:
                self.add(value, delta)
            elif delta < 0:
                self.subtract(value, -delta)

    def merge(self, other):
        """Incrementally fold another partition's multiplicities into self.

        `other` is a :class:`SortedMultisetView`, mapping or iterable of
        ``(value, multiplicity)`` pairs. Existing buckets get a
        `O(log m)` point update; new values insert a bucket.

        >>> left = SortedMultiset([(200, 5), (404, 1)])
        >>> right = SortedMultiset([(200, 2), (500, 7)])
        >>> left.merge(right)
        >>> left.multiplicity(200)
        7

        """
        aggregated = {}

        for value, count in self._pairs_of(other):
            self._require_positive_count(count)
            aggregated[value] = aggregated.get(value, 0) + count

        for value, count in aggregated.items():
            self.add(value, count)

    def withdraw(self, other):
        """Subtract another partition's multiplicities as one transaction.

        Negative buckets are previewed first: the whole withdrawal is
        rejected with :class:`NegativeCountError` and the multiset is left
        untouched if any bucket would go below zero. Only when every bucket
        stays non-negative do the subtractions take effect.

        >>> ms = SortedMultiset([(200, 5), (404, 2)])
        >>> ms.withdraw([(200, 4), (404, 2)])
        >>> list(ms.items())
        [(200, 1)]

        """
        aggregated = {}

        for value, count in self._pairs_of(other):
            self._require_positive_count(count)
            aggregated[value] = aggregated.get(value, 0) + count

        offending = {}

        for value, count in aggregated.items():
            current = self.multiplicity(value)

            if count > current:
                offending[value] = current - count

        if offending:
            raise NegativeCountError(offending=offending)

        for value, count in aggregated.items():
            self.subtract(value, count)

    @contextmanager
    def transaction(self):
        """Buffer adjustments and commit them as one atomic transaction.

        Yields a recorder exposing ``add``, ``subtract`` and ``adjust``
        methods. Commit runs the negative-bucket preview, so a failure raises
        :class:`NegativeCountError` with the multiset unchanged. An exception
        raised inside the block discards every buffered adjustment.

        >>> ms = SortedMultiset([(1, 2)])
        >>> with ms.transaction() as tx:
        ...     tx.add(2, 4)
        ...     tx.subtract(1, 1)
        >>> list(ms.items())
        [(1, 1), (2, 4)]

        """
        pending = {}
        require_positive = self._require_positive_count

        class _Transaction:
            def adjust(self_, value, delta):
                if not isinstance(delta, int) or isinstance(delta, bool):
                    raise ValueError('delta must be an integer')

                pending[value] = pending.get(value, 0) + delta

            def add(self_, value, count=1):
                require_positive(count)
                self_.adjust(value, count)

            def subtract(self_, value, count=1):
                require_positive(count)
                self_.adjust(value, -count)

        try:
            yield _Transaction()
        except Exception:
            raise
        else:
            if pending:
                self.apply_deltas(pending)

    def snapshot(self):
        """Capture a read-only view of the current distribution.

        The view copies the values, multiplicities and prefix-sum tree, so
        subsequent adjustments do not affect it and old ranks remain
        queryable.

        >>> ms = SortedMultiset([(1, 2), (2, 3)])
        >>> view = ms.snapshot()
        >>> ms.add(1, 10)
        >>> len(view), len(ms)
        (5, 15)

        """
        return SortedMultisetView(self._values, self._counts)
