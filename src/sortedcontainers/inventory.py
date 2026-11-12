"""Inventory ranking with retroactive corrections.

:class:`InventoryRanking` keeps the quantities of stock-keeping units
(SKUs) ranked in a :class:`SortedKeyList`. Because quantities are traced
back to source events, the quantity of a SKU may be corrected
retroactively any number of times. Each correction arrives as an
:class:`InventoryEvent` carrying the source *batch* and *version*.

The :func:`InventoryRanking.rebuild` entry point replays source events up
to a given batch and reports the business keys whose rank truly changed,
together with their old and new ranks. Rows that moved only because
another SKU crossed them are reported as well. Stale re-deliveries
(versions at or below the recorded one) never change any rank. If taking
a new sort key or comparing keys raises, no half-updated ranking is
published: the previous ranking stays in place. Small corrections only
touch the changed item and the crossed rank interval, which callers can
verify with :attr:`InventoryRanking.access_counts`.

"""

from bisect import bisect_left, bisect_right
from typing import Any, NamedTuple

from .sortedlist import SortedKeyList, identity

__all__ = ['InventoryRanking', 'InventoryEvent', 'RankChange', 'AccessCounts']


class InventoryEvent(NamedTuple):
    """Source event reporting the quantity of a SKU.

    * `sku` -- business key of the stock-keeping unit.
    * `version` -- version of the report; an event is applied only when
      its version is newer than the version recorded for the SKU.
    * `quantity` -- quantity on hand reported by the event.
    * `batch` -- source batch of the event, defaults to zero.

    """

    sku: Any
    version: int
    quantity: Any
    batch: int = 0


class RankChange(NamedTuple):
    """Rank change of a business key returned by ``rebuild``.

    * `sku` -- business key whose rank truly changed.
    * `old_rank` -- rank before the rebuild, ``None`` for new SKUs.
    * `new_rank` -- rank after the rebuild.

    """

    sku: Any
    old_rank: int | None
    new_rank: int


class AccessCounts(NamedTuple):
    """Access counts of user-supplied callables.

    * `key_calls` -- number of sort-key computations.
    * `comparisons` -- number of sort-key comparisons.

    """

    key_calls: int
    comparisons: int


class _Counter:
    """Mutable access counter shared by a ranking and its keys."""

    __slots__ = ('key_calls', 'comparisons')

    def __init__(self):
        self.key_calls = 0
        self.comparisons = 0


class _RankKey:
    """Order entries by descending sort key, ties by ascending sequence."""

    __slots__ = ('sort_key', 'seq', '_counter')

    def __init__(self, sort_key, seq, counter):
        self.sort_key = sort_key
        self.seq = seq
        self._counter = counter

    def __lt__(self, other):
        self._counter.comparisons += 1
        sort_key = self.sort_key
        other_key = other.sort_key
        if other_key < sort_key:
            return True
        if sort_key < other_key:
            return False
        return self.seq < other.seq

    def __le__(self, other):
        self._counter.comparisons += 1
        sort_key = self.sort_key
        other_key = other.sort_key
        if sort_key < other_key:
            return False
        if other_key < sort_key:
            return True
        return self.seq <= other.seq

    def __gt__(self, other):
        self._counter.comparisons += 1
        sort_key = self.sort_key
        other_key = other.sort_key
        if sort_key < other_key:
            return True
        if other_key < sort_key:
            return False
        return self.seq > other.seq

    def __ge__(self, other):
        self._counter.comparisons += 1
        sort_key = self.sort_key
        other_key = other.sort_key
        if other_key < sort_key:
            return False
        if sort_key < other_key:
            return True
        return self.seq >= other.seq

    def __eq__(self, other):
        self._counter.comparisons += 1
        return self.sort_key == other.sort_key and self.seq == other.seq

    def __repr__(self):
        return f'_RankKey({self.sort_key!r}, {self.seq!r})'


class _Entry:
    """Immutable record of a SKU in the ranking."""

    __slots__ = ('sku', 'version', 'quantity', 'seq', 'rank_key')

    def __init__(self, sku, version, quantity, seq, rank_key):
        self.sku = sku
        self.version = version
        self.quantity = quantity
        self.seq = seq
        self.rank_key = rank_key

    def __repr__(self):
        return f'_Entry({self.sku!r}, {self.version!r}, {self.quantity!r})'


def _entry_rank_key(entry):
    "Sort key of an entry: its precomputed rank key."
    return entry.rank_key


class InventoryRanking:
    """Ranking of SKUs by quantity with retroactive corrections.

    SKUs are ranked by descending sort key of their quantity. Equal
    quantities are ordered by the sequence number assigned when the SKU
    first entered the ranking: first in, better rank. Ranks are
    zero-based, so rank zero is the top of the ranking.

    Optional `events` argument provides an iterable of
    :class:`InventoryEvent` applied as by :func:`rebuild`.

    Optional `key` argument maps a quantity to a sort key, like the
    `key` argument of the builtin ``sorted``. The default is the
    identity function, so quantities are ordered directly.

    >>> ranking = InventoryRanking()
    >>> ranking.rebuild([
    ...     InventoryEvent('sku-a', version=1, quantity=50, batch=1),
    ...     InventoryEvent('sku-b', version=1, quantity=70, batch=1),
    ...     InventoryEvent('sku-c', version=1, quantity=60, batch=1),
    ... ])
    [RankChange(sku='sku-b', old_rank=None, new_rank=0), RankChange(sku='sku-c', old_rank=None, new_rank=1), RankChange(sku='sku-a', old_rank=None, new_rank=2)]
    >>> list(ranking)
    ['sku-b', 'sku-c', 'sku-a']

    Correcting a quantity moves the SKU and every row it crosses:

    >>> changes = ranking.rebuild([InventoryEvent('sku-a', version=2, quantity=90, batch=2)])
    >>> list(ranking)
    ['sku-a', 'sku-b', 'sku-c']
    >>> changes
    [RankChange(sku='sku-a', old_rank=2, new_rank=0), RankChange(sku='sku-b', old_rank=0, new_rank=1), RankChange(sku='sku-c', old_rank=1, new_rank=2)]

    Stale re-deliveries, versions at or below the recorded one, change
    nothing:

    >>> ranking.rebuild([InventoryEvent('sku-a', version=1, quantity=5, batch=3)])
    []
    >>> list(ranking)
    ['sku-a', 'sku-b', 'sku-c']

    """

    def __init__(self, events=(), key=None):
        self._key = identity if key is None else key
        self._counter = _Counter()
        self._skl = SortedKeyList(key=_entry_rank_key)
        self._entries = {}
        self._seq = 0
        self.rebuild(events)

    @property
    def key(self):
        "Function mapping quantities to sort keys."
        return self._key

    @property
    def access_counts(self):
        "Access counts of user-supplied callables since the last reset."
        counter = self._counter
        return AccessCounts(counter.key_calls, counter.comparisons)

    def reset_access_counts(self):
        "Reset the access counts of user-supplied callables to zero."
        counter = self._counter
        counter.key_calls = 0
        counter.comparisons = 0

    def __len__(self):
        "Return the number of SKUs in the ranking."
        return len(self._skl)

    def __contains__(self, sku):
        "Return true if `sku` is in the ranking."
        return sku in self._entries

    def __iter__(self):
        "Iterate over SKUs in rank order."
        for entry in self._skl:
            yield entry.sku

    def __getitem__(self, index):
        "Return the SKU at `index`; slices return a list of SKUs."
        result = self._skl[index]
        if isinstance(index, slice):
            return [entry.sku for entry in result]
        return result.sku

    def __repr__(self):
        pairs = [(entry.sku, entry.quantity) for entry in self._skl]
        return f'{type(self).__name__}({pairs!r})'

    def items(self):
        "Iterate over ``(sku, quantity)`` pairs in rank order."
        for entry in self._skl:
            yield entry.sku, entry.quantity

    def rank(self, sku):
        """Return the current zero-based rank of `sku`.

        Runtime complexity: `O(log(n))` -- approximate.

        :param sku: business key to look up
        :return: zero-based rank of `sku`
        :raises KeyError: if `sku` is not in the ranking

        """
        entry = self._entries[sku]
        pos, idx = self._find(self._skl, entry)
        return self._skl._loc(pos, idx)

    def quantity(self, sku):
        """Return the recorded quantity of `sku`.

        :raises KeyError: if `sku` is not in the ranking

        """
        return self._entries[sku].quantity

    def version(self, sku):
        """Return the recorded version of `sku`.

        :raises KeyError: if `sku` is not in the ranking

        """
        return self._entries[sku].version

    def rebuild(self, events, batch=None):
        """Rebuild the ranking from source `events` and report rank changes.

        Events are applied in order. An event is applied only when its
        batch is at most `batch` (all batches when `batch` is None) and
        its version is newer than the version recorded for the SKU.
        Stale re-deliveries are ignored and never change any rank. An
        event for an unknown SKU enters the ranking and is assigned the
        next first-in sequence number.

        If taking a new sort key or comparing keys raises, the ranking
        is left untouched: no half-updated ranking is published.

        A small correction only touches the changed item and the crossed
        rank interval; use :attr:`access_counts` to verify.

        :param events: iterable of :class:`InventoryEvent`
        :param batch: apply events up to and including this batch
            (optional, defaults to all batches)
        :return: list of :class:`RankChange` sorted by new rank

        """
        skl = self._snapshot()
        entries = dict(self._entries)
        seq = self._seq
        touched = {}

        for event in events:
            if batch is not None and event.batch > batch:
                continue
            sku = event.sku
            entry = entries.get(sku)
            if entry is None:
                new_entry, new_rank = self._prepare_entry(
                    skl, sku, event.version, event.quantity, seq
                )
                self._shift(skl, touched, new_rank, skl._len, 1)
                self._insert_at(skl, new_rank, new_entry)
                entries[sku] = new_entry
                seq += 1
                touched[sku] = [None, new_rank]
            else:
                if event.version <= entry.version:
                    continue
                prepared = self._prepare_correction(
                    skl, entry, event.version, event.quantity
                )
                new_entry, pos, idx, old_rank, new_rank = prepared
                if new_rank < old_rank:
                    self._shift(skl, touched, new_rank, old_rank, 1)
                elif new_rank > old_rank:
                    self._shift(skl, touched, old_rank + 1, new_rank + 1, -1)
                skl._delete(pos, idx)
                self._insert_at(skl, new_rank, new_entry)
                entries[sku] = new_entry
                if new_rank != old_rank:
                    slot = touched.get(sku)
                    if slot is None:
                        touched[sku] = [old_rank, new_rank]
                    else:
                        slot[1] = new_rank

        self._skl = skl
        self._entries = entries
        self._seq = seq

        changes = [
            RankChange(sku, old_rank, new_rank)
            for sku, (old_rank, new_rank) in touched.items()
            if old_rank != new_rank
        ]
        changes.sort(key=lambda change: change.new_rank)
        return changes

    def _snapshot(self):
        """Copy the sorted-key list structure.

        Only references are copied: no sort key is computed and no key
        comparison is made, so the copy cannot raise on user data.

        """
        skl = self._skl
        clone = SortedKeyList(key=skl.key)
        clone._len = skl._len
        clone._load = skl._load
        clone._lists = [sublist[:] for sublist in skl._lists]
        clone._keys = [sublist[:] for sublist in skl._keys]
        clone._maxes = skl._maxes[:]
        return clone

    def _take_sort_key(self, quantity):
        "Compute the sort key of `quantity`; counted as a key access."
        self._counter.key_calls += 1
        return self._key(quantity)

    @staticmethod
    def _find(skl, entry):
        """Locate `entry` in `skl` and return its ``(pos, idx)`` pair.

        Only key comparisons are made; the structure is not mutated.

        """
        rank_key = entry.rank_key
        pos = bisect_left(skl._maxes, rank_key)
        _lists = skl._lists
        _keys = skl._keys
        idx = bisect_left(_keys[pos], rank_key)
        len_keys = len(_keys)
        len_sublist = len(_keys[pos])
        while True:
            if _keys[pos][idx] != rank_key:
                raise ValueError(f'{entry.sku!r} not in ranking')
            if _lists[pos][idx] is entry:
                return pos, idx
            idx += 1
            if idx == len_sublist:
                pos += 1
                if pos == len_keys:
                    raise ValueError(f'{entry.sku!r} not in ranking')
                len_sublist = len(_keys[pos])
                idx = 0

    @staticmethod
    def _bisect_rank(skl, rank_key):
        """Return the global insertion rank of `rank_key` in `skl`.

        Only key comparisons are made; the structure is not mutated.

        """
        _maxes = skl._maxes
        if not _maxes:
            return 0
        pos = bisect_right(_maxes, rank_key)
        if pos == len(_maxes):
            return skl._len
        idx = bisect_right(skl._keys[pos], rank_key)
        return skl._loc(pos, idx)

    @staticmethod
    def _insert_at(skl, rank, entry):
        """Insert `entry` at global `rank` in `skl`.

        Runs no user code: no sort key is computed and no key comparison
        is made, so the insertion cannot raise on user data.

        """
        rank_key = entry.rank_key
        _lists = skl._lists
        if not _lists:
            _lists.append([entry])
            skl._keys.append([rank_key])
            skl._maxes.append(rank_key)
            skl._len = 1
            return
        if rank == skl._len:
            pos = len(_lists) - 1
            _lists[pos].append(entry)
            keys_pos = skl._keys[pos]
            keys_pos.append(rank_key)
            skl._maxes[pos] = rank_key
        else:
            pos, idx = skl._pos(rank)
            _lists[pos].insert(idx, entry)
            keys_pos = skl._keys[pos]
            keys_pos.insert(idx, rank_key)
            if idx == len(keys_pos) - 1:
                skl._maxes[pos] = rank_key
        skl._len += 1
        skl._expand(pos)

    def _prepare_entry(self, skl, sku, version, quantity, seq):
        """Prepare a new entry; may raise before anything is mutated."""
        sort_key = self._take_sort_key(quantity)
        rank_key = _RankKey(sort_key, seq, self._counter)
        entry = _Entry(sku, version, quantity, seq, rank_key)
        rank = self._bisect_rank(skl, rank_key)
        return entry, rank

    def _prepare_correction(self, skl, entry, version, quantity):
        """Prepare a corrected entry; may raise before anything is mutated.

        Return the new entry, the ``(pos, idx)`` pair of the old entry,
        and the old and new global ranks.

        """
        sort_key = self._take_sort_key(quantity)
        rank_key = _RankKey(sort_key, entry.seq, self._counter)
        new_entry = _Entry(entry.sku, version, quantity, entry.seq, rank_key)
        pos, idx = self._find(skl, entry)
        old_rank = skl._loc(pos, idx)
        new_rank = self._bisect_rank(skl, rank_key)
        if new_rank > old_rank:
            new_rank -= 1
        return new_entry, pos, idx, old_rank, new_rank

    @staticmethod
    def _shift(skl, touched, lo, hi, delta):
        """Record that entries at ranks in ``[lo, hi)`` move by `delta`."""
        getitem = skl.__getitem__
        for rank in range(lo, hi):
            sku = getitem(rank).sku
            slot = touched.get(sku)
            if slot is None:
                touched[sku] = [rank, rank + delta]
            else:
                slot[1] += delta
