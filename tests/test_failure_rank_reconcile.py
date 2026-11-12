"""Rank reconciliation for sorted-container failure paths.

A market-quote percentile index (SortedList of quotes ordered by price)
meets values it cannot compare only in the middle of a batch update. When
the batch raises, the container may be left in several states that look
alike from the outside. This module reconciles the rank view after each
failure against an independent model:

* Business row numbers (`Quote.rowno`) are assigned outside the container
  and never reused, so a missing row is distinguishable from a moved row.
* Ordering relations (ranks, duplicate-price ranges, price-window counts)
  are pre-recorded in `QuoteBook`, which is maintained with `bisect` on
  plain tuples and shares no code with sortedcontainers.

After every induced failure the reconciler checks each queryable quantile
(`slist[rank]` for every rank), each duplicate-key range
(`bisect_left`/`bisect_right` per distinct price) and range counts over
every price window (`irange`), then classifies the outcome:

* `ROWS_LOST` -- rows vanished from the physical blocks while the rank
  index stays valid (e.g. the documented ``del sl[i]`` + ``sl.add(v)``
  replace pattern when ``add`` raises).
* `RANK_INDEX_INVALID` -- rows are still physically present (or present in
  the dict view) but the rank index no longer reaches them (e.g. the bulk
  merge path of `SortedList.update` mutating ``_lists`` before the final
  sort raises).
* `KEY_SIDE_EFFECT` -- the container reconciles perfectly but the user's
  key/comparison left externally visible side effects (journals, counters).

All scenarios run at the minimum load factor (4) with just enough rows to
span multiple blocks. A failure that only reproduces across a structural
change (block split, batch merge) may not be shrunk to a single block and
declared excluded; `test_no_split_no_failure_control` pins the structural
trigger and `assert_multi_block` guards the scale of every scenario.
"""

import bisect
import enum

import pytest

from sortedcontainers import SortedDict, SortedKeyList, SortedList

LOAD = 4  # Minimum load factor accepted by SortedList._check.
HI = 1 << 62  # Sentinel rowno above every business row number.


class Quote:
    """Market quote. `rowno` is the independent business row number.

    Quotes from different generations (`gen`) are incomparable, mimicking a
    batch that mixes quote formats: comparisons across generations raise
    `TypeError`.
    """

    __slots__ = ('px', 'rowno', 'gen')

    def __init__(self, px, rowno, gen=0):
        self.px = px
        self.rowno = rowno
        self.gen = gen

    def _check_gen(self, other):
        if self.gen != other.gen:
            raise TypeError(f'incomparable quotes: {self!r} vs {other!r}')

    def __lt__(self, other):
        self._check_gen(other)
        return (self.px, self.rowno) < (other.px, other.rowno)

    def __le__(self, other):
        self._check_gen(other)
        return (self.px, self.rowno) <= (other.px, other.rowno)

    def __eq__(self, other):
        return isinstance(other, Quote) and self.rowno == other.rowno

    def __hash__(self):
        return hash(self.rowno)

    def __repr__(self):
        return f'Quote({self.px}, row={self.rowno}, gen={self.gen})'


class SplitTriggeredQuote(Quote):
    """Quote that becomes incomparable once the container grows a block.

    Every comparison is journaled externally (a user-key side effect) and
    raises `TypeError` once the container's block count exceeds the count
    observed at construction. This places the failure exactly after a block
    split without depending on timsort comparison counts.
    """

    __slots__ = ('_container', '_baseline', '_journal')

    def __init__(self, px, rowno, container, journal):
        super().__init__(px, rowno)
        self._container = container
        self._baseline = len(container._lists)
        self._journal = journal

    def _audit(self, other):
        self._journal.append((self.rowno, other.rowno))
        if len(self._container._lists) > self._baseline:
            raise TypeError('quote became incomparable after block split')

    def __lt__(self, other):
        self._audit(other)
        return super().__lt__(other)

    def __le__(self, other):
        self._audit(other)
        return super().__le__(other)


class QuoteBook:
    """Independent model pre-recording the ordering relations.

    Maintained with `bisect` on plain (price, rowno) tuples; shares no code
    with the container under test.
    """

    def __init__(self):
        self.rows = {}
        self.rank_keys = []

    def record(self, quote):
        self.rows[quote.rowno] = quote
        bisect.insort(self.rank_keys, (quote.px, quote.rowno))

    def expected_rownos(self):
        return [rowno for _, rowno in self.rank_keys]

    def distinct_prices(self):
        return sorted({px for px, _ in self.rank_keys})

    def dup_range(self, px):
        """Expected [lo, hi) ranks of rows priced exactly `px`."""
        lo = bisect.bisect_left(self.rank_keys, (px, -1))
        hi = bisect.bisect_right(self.rank_keys, (px, HI))
        return lo, hi

    def range_count(self, lo_px, hi_px):
        """Expected count of rows with lo_px <= price <= hi_px."""
        lo = bisect.bisect_left(self.rank_keys, (lo_px, -1))
        hi = bisect.bisect_right(self.rank_keys, (hi_px, HI))
        return hi - lo


class Outcome(enum.Enum):
    CLEAN = 'clean'
    ROWS_LOST = 'rows-lost'
    RANK_INDEX_INVALID = 'rank-index-invalid'
    KEY_SIDE_EFFECT = 'key-side-effect'


class ReconReport:
    def __init__(self):
        self.logical_len = None
        self.physical_rownos = None
        self.iterated_rownos = None
        self.index_valid = None
        self.quantile_errors = {}
        self.quantile_mismatches = {}
        self.dup_range_errors = {}
        self.dup_range_mismatches = {}
        self.range_count_errors = {}
        self.range_count_mismatches = {}

    def structurally_sound(self):
        return (
            self.index_valid
            and not self.quantile_errors
            and not self.quantile_mismatches
            and not self.dup_range_errors
            and not self.dup_range_mismatches
            and not self.range_count_errors
            and not self.range_count_mismatches
            and self.logical_len == len(self.physical_rownos)
            and self.iterated_rownos == self.physical_rownos
        )


def reconcile_list(slist, book):
    """Cross-check every queryable rank relation of `slist` against `book`."""
    report = ReconReport()
    report.logical_len = slist._len
    report.physical_rownos = [v.rowno for sub in slist._lists for v in sub]

    try:
        report.iterated_rownos = [v.rowno for v in slist]
    except Exception:
        report.iterated_rownos = None

    try:
        slist._check()
        report.index_valid = True
    except Exception:
        report.index_valid = False

    expected = book.expected_rownos()

    # Quantiles: every rank addressable through __getitem__.
    for rank in range(max(report.logical_len, len(expected))):
        try:
            actual = slist[rank].rowno
        except Exception as exc:
            report.quantile_errors[rank] = type(exc).__name__
            continue
        if rank >= len(expected) or actual != expected[rank]:
            want = expected[rank] if rank < len(expected) else None
            report.quantile_mismatches[rank] = (want, actual)

    # Duplicate-key ranges: [lo, hi) ranks per distinct price.
    for px in book.distinct_prices():
        try:
            actual = (
                slist.bisect_left(Quote(px, -1)),
                slist.bisect_right(Quote(px, HI)),
            )
        except Exception as exc:
            report.dup_range_errors[px] = type(exc).__name__
            continue
        want = book.dup_range(px)
        if actual != want:
            report.dup_range_mismatches[px] = (want, actual)

    # Range counts: every inclusive price window over distinct prices.
    prices = book.distinct_prices()
    for lo_pos, lo_px in enumerate(prices):
        for hi_px in prices[lo_pos:]:
            try:
                actual = len(list(slist.irange(Quote(lo_px, -1), Quote(hi_px, HI))))
            except Exception as exc:
                report.range_count_errors[(lo_px, hi_px)] = type(exc).__name__
                continue
            want = book.range_count(lo_px, hi_px)
            if actual != want:
                report.range_count_mismatches[(lo_px, hi_px)] = (want, actual)

    return report


def classify(report, book, side_effects=0):
    """Distinguish row loss, rank-index invalidation and key side effects."""
    expected = set(book.expected_rownos())
    missing = expected - set(report.physical_rownos)
    if missing:
        return Outcome.ROWS_LOST
    if not report.structurally_sound():
        return Outcome.RANK_INDEX_INVALID
    if side_effects:
        return Outcome.KEY_SIDE_EFFECT
    return Outcome.CLEAN


def build_book(quotes):
    book = QuoteBook()
    for quote in quotes:
        book.record(quote)
    return book


def build_list(quotes):
    slist = SortedList(quotes)
    slist._reset(LOAD)
    return slist


def assert_multi_block(slist):
    """Guard: scenario must not shrink to a single block.

    A failure that stops reproducing when the container fits in one block
    has not been excluded; the multi-block scale is part of the reproducer.
    """
    assert len(slist._lists) >= 2


def seed_quotes(count, gen=0):
    return [Quote(px, px, gen) for px in range(count)]


def test_add_failure_before_block_split():
    """Incomparable add raises in the `_maxes` bisect: nothing mutates."""
    quotes = seed_quotes(9)
    slist = build_list(quotes)
    book = build_book(quotes)
    blocks_before = [len(sub) for sub in slist._lists]

    with pytest.raises(TypeError):
        slist.add(Quote(2, 100, gen=1))

    assert_multi_block(slist)
    assert [len(sub) for sub in slist._lists] == blocks_before
    assert 100 not in {v.rowno for sub in slist._lists for v in sub}

    report = reconcile_list(slist, book)
    assert classify(report, book) is Outcome.CLEAN


def test_batch_failure_after_block_split():
    """Batch update commits a block split, then the poisoned quote raises.

    The failure is triggered by the split itself: the poisoned quote's
    comparisons raise only once the container has grown a new block.
    """
    quotes = seed_quotes(28)
    slist = build_list(quotes)
    blocks_before = len(slist._lists)

    journal = []
    good = [Quote(28 + i, 28 + i) for i in range(LOAD + 1)]
    poison = SplitTriggeredQuote(33, 33, slist, journal)
    batch = good + [poison]
    assert len(batch) * 4 < len(slist)  # Force the per-row merge path.

    with pytest.raises(TypeError):
        slist.update(batch)

    # The split and the good rows were committed before the failure.
    assert len(slist._lists) == blocks_before + 1
    assert_multi_block(slist)
    assert 33 not in {v.rowno for sub in slist._lists for v in sub}

    # Business expectation: committed prefix is present, poison is not.
    book = build_book(quotes + good)
    report = reconcile_list(slist, book)
    # Container reconciles, but the poisoned key journaled externally.
    assert journal
    assert classify(report, book, side_effects=len(journal)) is Outcome.KEY_SIDE_EFFECT


def test_batch_merge_failure_invalidates_rank_index():
    """Bulk merge path mutates `_lists` before the final sort raises."""
    quotes = seed_quotes(9)
    slist = build_list(quotes)
    batch = [Quote(10 + i, 100 + i, gen=1) for i in range(3)]
    assert len(batch) * 4 >= len(slist)  # Force the bulk merge path.

    with pytest.raises(TypeError):
        slist.update(batch)

    assert_multi_block(slist)
    book = build_book(quotes + batch)
    report = reconcile_list(slist, book)

    # Every batch row is still physically present in the blocks, but the
    # stale `_len`/`_maxes` make the tail rows unreachable by rank.
    assert set(book.expected_rownos()) <= set(report.physical_rownos)
    assert report.logical_len != len(book.expected_rownos())
    assert report.quantile_errors
    assert not report.index_valid
    assert classify(report, book) is Outcome.RANK_INDEX_INVALID


def test_sorteddict_bulk_update_failure_invalidates_rank_view():
    """SortedDict bulk update: dict view complete, rank view emptied."""

    class Key:
        def __init__(self, px, gen=0):
            self.px = px
            self.gen = gen

        def __lt__(self, other):
            if self.gen != other.gen:
                raise TypeError('incomparable keys')
            return self.px < other.px

        def __le__(self, other):
            if self.gen != other.gen:
                raise TypeError('incomparable keys')
            return self.px <= other.px

        def __eq__(self, other):
            return (
                isinstance(other, Key)
                and self.px == other.px
                and self.gen == other.gen
            )

        def __hash__(self):
            return hash(self.px)

    sdict = SortedDict((Key(px), px) for px in range(9))
    sdict._list._reset(LOAD)
    pairs = {Key(10 + i, gen=1): 100 + i for i in range(3)}
    assert 10 * len(pairs) > len(sdict)  # Force the bulk rebuild path.

    with pytest.raises(TypeError):
        sdict.update(pairs)

    # Mapping view keeps every row, including the incomparable new keys.
    assert dict.__len__(sdict) == 12
    assert sdict[Key(10, gen=1)] == 100

    # Rank view is invalidated: the key list was cleared and never rebuilt.
    # The keys view reports a mapping-derived length of 12 while its own
    # iteration and rank access go through the empty rank index.
    assert len(sdict._list) == 0
    assert len(sdict.keys()) == 12
    assert list(sdict.keys()) == []
    assert len(list(dict.__iter__(sdict))) == 12
    for rank in range(12):
        with pytest.raises(IndexError):
            sdict.keys()[rank]
    with pytest.raises(AssertionError):
        sdict._check()


def test_existing_key_update_partial_commit():
    """Small-path SortedDict.update: existing-key updates commit, then the
    incomparable new key raises before touching dict or rank index."""

    class Key:
        def __init__(self, px, gen=0):
            self.px = px
            self.gen = gen

        def __lt__(self, other):
            if self.gen != other.gen:
                raise TypeError('incomparable keys')
            return self.px < other.px

        def __le__(self, other):
            if self.gen != other.gen:
                raise TypeError('incomparable keys')
            return self.px <= other.px

        def __eq__(self, other):
            return (
                isinstance(other, Key)
                and self.px == other.px
                and self.gen == other.gen
            )

        def __hash__(self):
            return hash(self.px)

    sdict = SortedDict((Key(px), px) for px in range(30))
    sdict._list._reset(LOAD)
    pairs = {Key(0): 'u0', Key(1): 'u1', Key(99, gen=1): 'bad'}
    assert 10 * len(pairs) <= len(sdict)  # Force the per-key path.

    with pytest.raises(TypeError):
        sdict.update(pairs)

    # Existing-key updates committed before the failure.
    assert sdict[Key(0)] == 'u0'
    assert sdict[Key(1)] == 'u1'
    # The incomparable new key is rejected atomically: neither view moved.
    assert Key(99, gen=1) not in sdict
    assert dict.__len__(sdict) == 30
    assert len(sdict._list) == 30
    sdict._check()

    # Rank reconciliation: every key still sits at its pre-recorded rank.
    for rank in range(30):
        assert sdict.keys()[rank].px == rank
    assert_multi_block(sdict._list)


def test_replace_pattern_loses_row():
    """Documented replace pattern (`del` then `add`) loses the row when the
    `add` raises: rank index stays valid but the row is physically gone."""
    quotes = seed_quotes(9)
    slist = build_list(quotes)
    book = build_book(quotes)

    rank = slist.bisect_left(Quote(4, 4))
    del slist[rank]
    with pytest.raises(TypeError):
        slist.add(Quote(40, 40, gen=1))

    assert_multi_block(slist)
    report = reconcile_list(slist, book)

    # Index is valid and every remaining rank reconciles; row 4 is gone.
    assert report.index_valid
    assert 4 not in report.physical_rownos
    assert report.logical_len == 8
    assert classify(report, book) is Outcome.ROWS_LOST


def test_key_function_side_effect_only():
    """Key function raising leaves the container clean but journals externally."""
    journal = []

    def feed_key(value):
        journal.append(value)
        if value == 'halt':
            raise ValueError('key function failed')
        return value

    sklist = SortedKeyList((float(i) for i in range(9)), key=feed_key)
    sklist._reset(LOAD)
    journal.clear()

    with pytest.raises(ValueError):
        sklist.add('halt')

    assert_multi_block(sklist)
    # Container reconciles exactly; only the external journal moved.
    assert list(sklist) == [float(i) for i in range(9)]
    assert sklist[4] == 4.0
    assert sklist.bisect_key_left(4.0) == 4
    assert journal == ['halt']
    sklist._check()


def test_no_split_no_failure_control():
    """Control: the same poisoned quote is harmless when no split occurs.

    The split scenario's failure is structurally tied to the block split;
    shrinking the data so no split happens makes it stop reproducing, which
    is exactly why the multi-block scale may not be treated as excluded.
    """
    quotes = seed_quotes(25)  # Last block has 1 row: 5 adds will not split it.
    slist = build_list(quotes)
    blocks_before = len(slist._lists)

    journal = []
    good = [Quote(25 + i, 25 + i) for i in range(LOAD + 1)]
    poison = SplitTriggeredQuote(30, 30, slist, journal)
    batch = good + [poison]
    assert len(batch) * 4 < len(slist)

    slist.update(batch)  # No split -> poison never becomes incomparable.

    assert len(slist._lists) == blocks_before
    assert 30 in {v.rowno for sub in slist._lists for v in sub}
    book = build_book(quotes + batch)
    report = reconcile_list(slist, book)
    assert classify(report, book, side_effects=len(journal)) is Outcome.KEY_SIDE_EFFECT
