"""
Memo audit controls.

When a DBF table is opened with memo limits two kinds of caps are
enforced while reading the companion memo file (FPT/DBT):

* ``max_memo_size`` limits the size of a single memo value.
* ``memo_budget`` limits the total number of *distinct* bytes read from
  the memo file. Several fields or records may point at the same memo
  block; the byte range covered by that block is paid for only once.

The :class:`ReadBudget` is the bookkeeper. It records every byte range
that has actually been read as a merged set of intervals, tracks memo
blocks that have been fully cached or quarantined, and records declared
ranges that overlap with conflicting boundaries.

Memo values that cannot be delivered under the limits are returned as
:class:`QuarantinedMemo` objects instead of raising, so that the rest of
the record can still be streamed.
"""


# Reason codes attached to QuarantinedMemo objects.
MEMO_INDEX_INVALID = 'memo_index_invalid'
VALUE_EXCEEDS_LIMIT = 'value_exceeds_limit'
BUDGET_EXHAUSTED = 'budget_exhausted'
MEMO_TRUNCATED = 'memo_truncated'


class QuarantinedMemo(object):
    """A memo value that could not be delivered under the audit limits.

    The object carries a machine readable ``reason`` code and enough
    context (memo block index, declared file range, record number and
    field name) to explain why the value is isolated.
    """

    def __init__(self, reason, index=None, detail=None, range=None,
                 record=None, field=None):
        self.reason = reason
        self.index = index
        self.detail = detail
        self.range = range
        self.record = record
        self.field = field

    def at(self, field, record):
        """Return a copy annotated with the referencing field and record."""
        return QuarantinedMemo(
            self.reason, index=self.index, detail=self.detail,
            range=self.range, record=record, field=field)

    def __repr__(self):
        parts = ['reason={!r}'.format(self.reason)]
        if self.index is not None:
            parts.append('index={!r}'.format(self.index))
        if self.field is not None:
            parts.append('field={!r}'.format(self.field))
        if self.record is not None:
            parts.append('record={!r}'.format(self.record))
        if self.range is not None:
            parts.append('range={!r}'.format(self.range))
        if self.detail is not None:
            parts.append('detail={!r}'.format(self.detail))
        return 'QuarantinedMemo({})'.format(', '.join(parts))

    def __eq__(self, other):
        if not isinstance(other, QuarantinedMemo):
            return NotImplemented
        own = (self.reason, self.index, self.detail,
               self.range, self.record, self.field)
        their = (other.reason, other.index, other.detail,
                 other.range, other.record, other.field)
        return own == their

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    __hash__ = None


class RangeConflict(object):
    """Two declared memo ranges that overlap with different boundaries."""

    def __init__(self, first_range, first_refs, second_range, second_refs):
        self.first_range = first_range
        self.first_refs = list(first_refs)
        self.second_range = second_range
        self.second_refs = list(second_refs)

    @property
    def refs(self):
        """All (record number, field name) references involved."""
        return self.first_refs + self.second_refs

    def __repr__(self):
        return ('RangeConflict(first_range={!r}, second_range={!r}, '
                'refs={!r})').format(self.first_range, self.second_range,
                                     self.refs)

    def __eq__(self, other):
        if not isinstance(other, RangeConflict):
            return NotImplemented
        own = (self.first_range, self.second_range, self.refs)
        their = (other.first_range, other.second_range, other.refs)
        return own == their

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    __hash__ = None


class ReadBudget(object):
    """Budget and interval bookkeeping for one memo file.

    A single instance lives on the :class:`dbfread.DBF` table and is
    shared by every record pass, so totals and cached memos survive the
    memo file being closed and reopened between iterations.
    """

    def __init__(self, max_bytes=None, max_memo_size=None):
        self.max_bytes = max_bytes
        self.max_memo_size = max_memo_size

        # Number of distinct file bytes currently accounted for.
        self.used = 0

        # Merged, sorted list of (start, end) intervals already read.
        self._ranges = []

        # Declared memo extents: (start, end) -> list of (record, field).
        self._declared = {}

        # Conflicting declarations in the order they were discovered.
        self.conflicts = []

        # Memo blocks that have been read in full. Index -> memo object.
        self.cache = {}

        # Memo blocks that could not be delivered. Index -> QuarantinedMemo.
        self.quarantined = {}

    @property
    def remaining(self):
        """Bytes still available in the global budget."""
        if self.max_bytes is None:
            return None
        return max(0, self.max_bytes - self.used)

    @property
    def exhausted(self):
        return self.max_bytes is not None and self.used >= self.max_bytes

    @property
    def ranges(self):
        """Snapshot of the merged intervals that have been read."""
        return list(self._ranges)

    def value_is_allowed(self, size):
        """Check a single memo value against the per-value limit."""
        return self.max_memo_size is None or size <= self.max_memo_size

    def unreserved_count(self, start, end):
        """How many new bytes reserving ``[start, end)`` would add.

        Parts already covered by an earlier reservation are excluded, so
        callers can tell whether a physical read stays inside the
        remaining budget before issuing it.
        """
        if end <= start:
            return 0
        return self._measure_with(start, end) - self.used

    def reserve(self, start, end):
        """Account for the byte range ``[start, end)``.

        Overlapping parts of ranges already reserved are counted once.
        Returns ``True`` if the whole range fits in the budget and was
        recorded, ``False`` (without changing the budget) otherwise.
        """
        if end <= start:
            return True

        new_bytes = self._measure_with(start, end) - self.used
        if self.max_bytes is None:
            fits = True
        else:
            fits = new_bytes <= self.max_bytes - self.used
        if not fits:
            return False

        self._ranges = self._merge(self._ranges, (start, end))
        self.used += new_bytes
        return True

    def declare(self, start, end, ref=None):
        """Record the declared extent of a memo block.

        ``ref`` is a ``(record number, field name)`` tuple identifying
        the record that references the block. If another declared range
        overlaps this one with different boundaries a
        :class:`RangeConflict` naming every involved reference is
        recorded and returned.
        """
        declared = (start, end)
        conflict = None
        for other_range, refs in self._declared.items():
            if self._overlaps(declared, other_range) \
                    and declared != other_range:
                conflict = RangeConflict(other_range, refs,
                                         declared, self._as_refs(ref))
                self.conflicts.append(conflict)

        self._declared.setdefault(declared, []).extend(self._as_refs(ref))
        return conflict

    def quarantine(self, index, reason, range=None, detail=None):
        """Record and return a shared quarantine for a memo block."""
        memo = QuarantinedMemo(reason, index=index, detail=detail,
                               range=range)
        self.quarantined[index] = memo
        return memo

    def _measure_with(self, start, end):
        return self._measure(self._merge(self._ranges, (start, end)))

    @staticmethod
    def _measure(ranges):
        return sum(end - start for start, end in ranges)

    @staticmethod
    def _merge(ranges, new_range):
        merged = []
        start, end = new_range
        for other_start, other_end in ranges:
            if other_end < start:
                merged.append((other_start, other_end))
            elif other_start > end:
                merged.append((start, end))
                start, end = other_start, other_end
            else:
                start = min(start, other_start)
                end = max(end, other_end)
        merged.append((start, end))
        return merged

    @staticmethod
    def _overlaps(first, second):
        # A None end means the boundary was never discovered (DBT), in
        # which case the open range overlaps anything sharing its start
        # side that extends beyond it.
        first_start, first_end = first
        second_start, second_end = second
        first_inf = first_end is None
        second_inf = second_end is None
        if first_inf and second_inf:
            return first_start == second_start
        if first_inf:
            return second_end > first_start
        if second_inf:
            return first_end > second_start
        return first_start < second_end and second_start < first_end

    @staticmethod
    def _as_refs(ref):
        if ref is None:
            return []
        return [ref]
