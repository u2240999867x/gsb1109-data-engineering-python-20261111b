"""Sorted Migration -- snapshots and cursor migration across sort policies.

Offline readers paginate a sorted container with cursors: a cursor is the
number of rows already delivered, so the next page begins at that row. When
the sort policy changes, rows may move and naive cursors would skip or
repeat rows. This module provides a migration entry point built on three
pieces:

1. :class:`SortPolicyRegistry` maps policy names to key functions. Snapshots
   store only the registered name -- key functions are never serialized.

2. :class:`Snapshot` captures the rows of a sorted container with stable row
   numbers (``0`` to ``len - 1``) in the order of the old policy. Duplicate
   values are independent rows with distinct row numbers. Snapshots convert
   to and from plain dicts and JSON for transport to another process.

3. :func:`migrate` (or :meth:`Snapshot.migrate`) loads a snapshot under a new
   registered policy and returns a :class:`Migration` describing how every
   old cursor transfers to the new ordering.

When the new policy only refines the old one -- rows are reordered strictly
within old equal-key groups -- every old cursor remains continuable and the
migration reports the new-order interval in which its tail rows lie. When
the new policy reorders rows across old groups, the migration reports the
actually affected rows and invalidates each cursor whose already-delivered
prefix interleaves with its undelivered tail in the new ordering.

For every continuable cursor the old page prefix (rows before the cursor)
and the migrated tail read together cover every row exactly once.

::

    >>> from sortedcontainers import SortedKeyList, Snapshot, SortPolicyRegistry
    >>> registry = SortPolicyRegistry()
    >>> def length_then_text(value):
    ...     return (len(value), value)
    >>> registry.register('length', len) is len
    True
    >>> registry.register('length-then-text', length_then_text) is length_then_text
    True
    >>> words = SortedKeyList(['cherry', 'fig', 'banana', 'apple', 'kiwi'], key=len)
    >>> snapshot = Snapshot.capture(words, 'length', registry)
    >>> snapshot.rows
    ['fig', 'kiwi', 'apple', 'cherry', 'banana']

The snapshot serializes to plain data -- no functions:

    >>> import json
    >>> json.loads(snapshot.to_json()) == snapshot.to_dict()
    True

In a new process, register the policies and migrate:

    >>> migration = snapshot.migrate('length-then-text', registry)
    >>> migration.kind
    'refinement'
    >>> list(migration.values)
    ['fig', 'kiwi', 'apple', 'banana', 'cherry']

Cursor 4 already delivered ``['fig', 'kiwi', 'apple', 'cherry']``. Its tail
is confined to the interval of its old equal-key group:

    >>> plan = migration.cursor(4)
    >>> plan.status, plan.start, plan.stop
    ('ranged', 3, 5)
    >>> list(plan.tail())
    ['banana']

"""

import json as _json
from bisect import bisect_right as _bisect_right

from .sortedlist import SortedKeyList as _SortedKeyList


def _identity(value):
    return value


class InvalidCursorError(Exception):
    """Raised when reading the tail of an invalidated cursor."""


class SortPolicyRegistry:
    """Registry mapping sort policy names to key functions.

    Snapshots serialize policy names rather than key functions, so a process
    loading a snapshot resolves names through a registry. Registering the
    same name twice with the same function is a no-op; registering it with a
    different function raises :exc:`ValueError`.

    """

    __slots__ = ('_policies',)

    def __init__(self):
        self._policies = {}

    def register(self, name, key):
        """Register *key* under *name* and return *key*."""
        if not isinstance(name, str):
            raise TypeError(f'policy name must be a string: {name!r}')
        if not callable(key):
            raise TypeError(f'policy key must be callable: {key!r}')
        existing = self._policies.get(name)
        if existing is not None and existing is not key:
            raise ValueError(f'policy name already registered: {name!r}')
        self._policies[name] = key
        return key

    def name_of(self, key):
        """Return the registered name of *key*.

        Raises :exc:`ValueError` if *key* is not registered.

        """
        for name, registered in self._policies.items():
            if registered is key:
                return name
        raise ValueError(f'key function is not registered: {key!r}')

    def __getitem__(self, name):
        try:
            return self._policies[name]
        except KeyError:
            raise KeyError(f'unregistered sort policy: {name!r}') from None

    def __contains__(self, name):
        return name in self._policies

    def __len__(self):
        return len(self._policies)

    def __iter__(self):
        return iter(self._policies)

    def __repr__(self):
        return f'{type(self).__name__}({sorted(self._policies)!r})'


#: Default registry used when none is given. Pre-registers the ``identity``
#: policy matching the natural ordering of a plain :class:`SortedList`.
DEFAULT_REGISTRY = SortPolicyRegistry()
DEFAULT_REGISTRY.register('identity', _identity)


def register_policy(name, key):
    """Register *key* under *name* in the default registry."""
    return DEFAULT_REGISTRY.register(name, key)


class Snapshot:
    """Serializable snapshot of a sorted container with stable row numbers.

    Rows are stored in the container's order under the old sort policy and
    numbered ``0`` to ``len(snapshot) - 1``; those numbers are the stable row
    numbers cursors refer to. Only row data and the registered policy name
    are stored -- key functions are never serialized.

    """

    __slots__ = ('rows', 'policy')

    def __init__(self, rows, policy):
        self.rows = list(rows)
        self.policy = policy

    @classmethod
    def capture(cls, container, policy=None, registry=DEFAULT_REGISTRY):
        """Capture *container*'s rows in iteration order as a snapshot.

        *policy* is the registered name of the container's sort policy, or a
        registered key function. When omitted, the policy is inferred from
        the container's ``key`` attribute, defaulting to ``'identity'``.

        """
        if policy is None:
            key = getattr(container, 'key', None)
            policy = 'identity' if key is None else registry.name_of(key)
        elif callable(policy):
            policy = registry.name_of(policy)
        if policy not in registry:
            raise ValueError(f'unregistered sort policy: {policy!r}')
        return cls(list(container), policy)

    def to_dict(self):
        """Return the snapshot as plain data: policy name and rows."""
        return {'policy': self.policy, 'rows': list(self.rows)}

    @classmethod
    def from_dict(cls, mapping, registry=DEFAULT_REGISTRY):
        """Restore a snapshot from plain data produced by :meth:`to_dict`.

        The stored policy must be registered in *registry*.

        """
        policy = mapping['policy']
        if policy not in registry:
            raise KeyError(f'unregistered sort policy: {policy!r}')
        return cls(mapping['rows'], policy)

    def to_json(self, **kwargs):
        """Serialize the snapshot to JSON. Rows must be JSON-serializable."""
        return _json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, data, registry=DEFAULT_REGISTRY):
        """Restore a snapshot from JSON produced by :meth:`to_json`."""
        return cls.from_dict(_json.loads(data), registry)

    def migrate(self, policy, registry=DEFAULT_REGISTRY):
        """Load the snapshot under a new registered policy.

        Returns a :class:`Migration` describing how old cursors transfer.

        """
        return Migration(self, policy, registry)

    def __len__(self):
        return len(self.rows)

    def __eq__(self, other):
        if not isinstance(other, Snapshot):
            return NotImplemented
        return self.rows == other.rows and self.policy == other.policy

    def __repr__(self):
        return f'{type(self).__name__}(policy={self.policy!r}, rows={self.rows!r})'


def migrate(snapshot, policy, registry=DEFAULT_REGISTRY):
    """Migrate *snapshot* to the registered policy named *policy*."""
    return snapshot.migrate(policy, registry)


class Migration:
    """Result of migrating a snapshot to a new sort policy.

    Attributes:

    * ``kind`` -- ``'refinement'`` when the new policy only reorders rows
      within old equal-key groups, else ``'reorder'``.
    * ``affected_rows`` -- stable row numbers involved in cross-group
      reordering; empty for refinements.
    * ``order`` -- stable row numbers in the new ordering.
    * ``values`` -- row values in the new ordering.
    * ``sorted_list`` -- a :class:`SortedKeyList` of the rows under the new
      policy.

    """

    __slots__ = (
        'snapshot',
        'policy',
        'kind',
        'affected_rows',
        'order',
        'values',
        'sorted_list',
        '_count',
        '_groups',
        '_group_starts',
        '_prefix_max',
        '_suffix_min',
    )

    def __init__(self, snapshot, policy, registry=DEFAULT_REGISTRY):
        if callable(policy):
            policy = registry.name_of(policy)
        old_key = registry[snapshot.policy]
        new_key = registry[policy]
        rows = snapshot.rows
        count = len(rows)
        old_keys = [old_key(value) for value in rows]
        new_keys = [new_key(value) for value in rows]

        groups = []
        start = 0
        for index in range(1, count + 1):
            if index == count or old_keys[index] != old_keys[start]:
                groups.append((start, index))
                start = index

        group_max = [max(new_keys[start:stop]) for start, stop in groups]
        group_min = [min(new_keys[start:stop]) for start, stop in groups]
        refinement = all(
            earlier <= later
            for earlier, later in zip(group_max[:-1], group_min[1:], strict=True)
        )

        # Stable sort keeps old relative order within equal new keys.
        order = sorted(range(count), key=new_keys.__getitem__)
        positions = [0] * count
        for position, row in enumerate(order):
            positions[row] = position

        prefix_max = [0] * count
        running = -1
        for index in range(count):
            running = max(running, positions[index])
            prefix_max[index] = running
        suffix_min = [0] * count
        running = count
        for index in range(count - 1, -1, -1):
            running = min(running, positions[index])
            suffix_min[index] = running

        affected = []
        if not refinement:
            later_min = [None] * (len(groups) + 1)
            for group_index in range(len(groups) - 1, -1, -1):
                below = later_min[group_index + 1]
                own = group_min[group_index]
                later_min[group_index] = own if below is None else min(below, own)
            earlier_max = None
            for group_index, (start, stop) in enumerate(groups):
                below = later_min[group_index + 1]
                for index in range(start, stop):
                    key = new_keys[index]
                    if (earlier_max is not None and key < earlier_max) or (
                        below is not None and key > below
                    ):
                        affected.append(index)
                own = group_max[group_index]
                earlier_max = own if earlier_max is None else max(earlier_max, own)

        self.snapshot = snapshot
        self.policy = policy
        self.kind = 'refinement' if refinement else 'reorder'
        self.affected_rows = tuple(affected)
        self.order = tuple(order)
        self.values = tuple(rows[row] for row in order)
        self.sorted_list = _SortedKeyList(self.values, key=new_key)
        self._count = count
        self._groups = groups
        self._group_starts = [start for start, _ in groups]
        self._prefix_max = prefix_max
        self._suffix_min = suffix_min

    def cursor(self, row_number):
        """Return the :class:`CursorPlan` for an old cursor.

        *row_number* is the number of rows already delivered under the old
        policy, from ``0`` (nothing delivered) to ``len(snapshot)``.

        """
        count = self._count
        if not isinstance(row_number, int) or not 0 <= row_number <= count:
            raise ValueError(f'cursor out of range: {row_number!r}')
        if (
            row_number == 0
            or row_number == count
            or self._prefix_max[row_number - 1] < self._suffix_min[row_number]
        ):
            # Delivered prefix sorts entirely before the undelivered tail in
            # the new ordering: resume exactly at the same index.
            return CursorPlan(self, row_number, 'exact', row_number, count)
        if self.kind == 'refinement':
            # The cut splits an old equal-key group. Groups occupy the same
            # index blocks in the new ordering, so the tail lies in the
            # interval from the group's start to the end.
            group = _bisect_right(self._group_starts, row_number) - 1
            start = self._groups[group][0]
            return CursorPlan(self, row_number, 'ranged', start, count)
        return CursorPlan(self, row_number, 'invalid', None, None)

    def __len__(self):
        return self._count

    def __repr__(self):
        return (
            f'{type(self).__name__}(policy={self.policy!r}, kind={self.kind!r},'
            f' rows={self._count})'
        )


class CursorPlan:
    """Continuation plan for one old cursor under a migration.

    * ``status`` -- ``'exact'`` when the tail is precisely the new-order
      interval ``[start, stop)``; ``'ranged'`` when the tail lies in that
      interval but rows already delivered must be skipped; ``'invalid'``
      when the cursor cannot safely continue and is voided.
    * ``start``, ``stop`` -- the new-order interval of the tail, or ``None``
      for invalid cursors.

    """

    __slots__ = ('migration', 'cursor', 'status', 'start', 'stop')

    def __init__(self, migration, cursor, status, start, stop):
        self.migration = migration
        self.cursor = cursor
        self.status = status
        self.start = start
        self.stop = stop

    @property
    def continuable(self):
        """Whether the cursor can safely continue paging."""
        return self.status != 'invalid'

    def tail_rows(self):
        """Yield ``(row_number, value)`` pairs not yet delivered.

        Row numbers are the snapshot's stable row numbers. Raises
        :exc:`InvalidCursorError` if the cursor was invalidated.

        """
        if self.status == 'invalid':
            raise InvalidCursorError(
                f'cursor {self.cursor} cannot safely continue after migration'
                f' to policy {self.migration.policy!r}'
            )
        order = self.migration.order
        values = self.migration.values
        cursor = self.cursor
        for position in range(self.start, self.stop):
            row = order[position]
            if row >= cursor:
                yield row, values[position]

    def tail(self):
        """Yield the tail values in the new ordering."""
        for _, value in self.tail_rows():
            yield value

    def __repr__(self):
        return (
            f'{type(self).__name__}(cursor={self.cursor!r},'
            f' status={self.status!r}, start={self.start!r},'
            f' stop={self.stop!r})'
        )
