"""
Sparse reads of DBF records.

A bad-row report lists physical record numbers (1-based and counted in
file order, so deleted rows still occupy their slot).  When the storage
layer only serves byte ranges and bills per request plus per byte
transferred it is cheaper to read a handful of contiguous ranges than
to issue one request per row.

:func:`plan_sparse_reads` turns a list of physical record numbers into
the cheapest set of byte ranges that covers every requested row given
the fixed cost of starting a read, the cost per byte and the maximum
length of a single read.  A range may span rows that were not
requested; those rows are transferred but never decoded or returned.

:func:`read_sparse` executes such a plan against an open :class:`DBF`
table.  If a read comes back shorter than planned only the rows fully
contained in the returned bytes are decoded.  Rows with insufficient
evidence are reported as :class:`ShortRead` objects together with the
exact byte range that must be read to continue.
"""
from collections import namedtuple


# Captured at import time so tests can replace the file opener without
# touching the memo module (which has its own reference to ``open``).
_open = open


class ReadCost(namedtuple('ReadCost', ['startup', 'per_byte'])):
    """Pricing of the storage layer.

    ``startup`` is charged once per read request and ``per_byte`` once
    per transferred byte.  Both may be ints or floats.
    """
    __slots__ = ()

    def of(self, length):
        """Cost of a single read that transfers ``length`` bytes."""
        return self.startup + self.per_byte * length


class ReadRange(namedtuple('ReadRange', ['start', 'end', 'records'])):
    """A byte range to read and the physical records it covers.

    The range is half-open, so bytes ``start`` ... ``end - 1`` are read
    and ``length == end - start``.  ``records`` is the tuple of physical
    record numbers whose fixed-length slots are fully contained in the
    range, in ascending order.
    """
    __slots__ = ()

    @property
    def length(self):
        return self.end - self.start

    def cost(self, pricing):
        """Cost of reading this range under ``pricing``."""
        return pricing.of(self.length)


class SparseReadPlan(namedtuple('SparseReadPlan',
                                ['ranges', 'cost', 'record_order',
                                 'record_ranges'])):
    """The cheapest set of reads covering a list of record numbers.

    ``ranges`` is the tuple of :class:`ReadRange` objects to read in
    file order.  ``cost`` is their combined price.  ``record_order`` is
    the requested record numbers with the original order and any
    duplicates preserved, and ``record_ranges`` is a parallel tuple with
    the :class:`ReadRange` that covers each entry.
    """
    __slots__ = ()

    def range_for(self, record_number):
        """Return the :class:`ReadRange` that covers ``record_number``."""
        for read_range in self.ranges:
            if record_number in read_range.records:
                return read_range
        raise KeyError(record_number)


class ShortRead(namedtuple('ShortRead',
                           ['record', 'have_start', 'have_end', 'missing'])):
    """A requested record that a short read could not supply.

    The read starting at ``have_start`` only returned the bytes up to
    ``have_end`` (half-open).  ``missing`` is the :class:`ReadRange`
    that must be read to continue: it contains the tail of this record
    and every other still-missing record from the same planned read.
    """
    __slots__ = ()


class SparseReadResult(namedtuple('SparseReadResult',
                                  ['records', 'short_reads',
                                   'missing_ranges', 'bytes_read',
                                   'requests', 'plan'])):
    """Outcome of executing a :class:`SparseReadPlan`.

    ``records`` is a list of ``(record_number, record)`` pairs in
    request order (duplicates included).  ``short_reads`` is a list of
    :class:`ShortRead` objects, also in request order.  ``missing_ranges``
    is the tuple of distinct continuation ranges needed to finish the
    short reads.  ``bytes_read`` and ``requests`` report what was
    actually transferred and issued.
    """
    __slots__ = ()


def record_offsets(table, record_number):
    """Return the ``(start, end)`` byte offsets of a record slot.

    ``record_number`` is 1-based and counts deleted rows as well.
    """
    header = table.header
    start = header.headerlen + (record_number - 1) * header.recordlen
    return start, start + header.recordlen


def _validate_record_numbers(record_numbers, numrecords):
    for record_number in record_numbers:
        if isinstance(record_number, bool) \
                or not isinstance(record_number, int):
            raise TypeError(
                'record numbers must be ints, got {!r}'.format(record_number))
        if record_number < 1 or record_number > numrecords:
            raise ValueError(
                'record number {} is outside the table '
                '(1..{})'.format(record_number, numrecords))


def _make_pricing(startup_cost, byte_cost, max_length, recordlen):
    if startup_cost < 0:
        raise ValueError('startup_cost must be non-negative')
    if byte_cost < 0:
        raise ValueError('byte_cost must be non-negative')
    if max_length is None:
        limit = None
    else:
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise TypeError('max_length must be an int or None')
        if max_length <= 0:
            raise ValueError('max_length must be positive')
        if max_length < recordlen:
            raise ValueError(
                'max_length ({}) is smaller than one record '
                '({} bytes)'.format(max_length, recordlen))
        limit = max_length
    return ReadCost(startup_cost, byte_cost), limit


def _make_range(headerlen, recordlen, requested):
    # The read spans every physical slot between the endpoints, so gap
    # rows that are transferred but never decoded are listed too.
    first = requested[0]
    last = requested[-1]
    start = headerlen + (first - 1) * recordlen
    end = headerlen + last * recordlen
    covered = tuple(range(first, last + 1))
    return ReadRange(start, end, covered)


def _optimal_segments(unique, recordlen, max_length, pricing):
    # Dynamic program over the sorted, unique record numbers.  A
    # segment unique[start:end + 1] becomes one read that spans every
    # physical record between its endpoints, requested or not.
    count = len(unique)
    dp = [None] * count
    parent = [None] * count

    for end in range(count):
        best = None
        for start in range(end, -1, -1):
            length = (unique[end] - unique[start] + 1) * recordlen
            if max_length is not None and length > max_length:
                # Widening the segment further only makes it longer.
                break

            segment_cost = pricing.of(length)
            if start == 0:
                prefix_cost = 0
                prefix_segments = 0
            else:
                prefix_cost, prefix_segments = dp[start - 1]

            candidate = (prefix_cost + segment_cost,
                         prefix_segments + 1,
                         start)
            if best is None \
                    or candidate[0] < best[0] \
                    or (candidate[0] == best[0]
                        and candidate[1] < best[1]):
                best = candidate

        dp[end] = (best[0], best[1])
        parent[end] = best[2]

    segments = []
    end = count - 1
    while end >= 0:
        start = parent[end]
        segments.append(unique[start:end + 1])
        end = start - 1
    segments.reverse()
    return segments


def plan_sparse_reads(table, record_numbers, startup_cost=0.0,
                      byte_cost=0.0, max_length=None):
    """Plan the cheapest byte ranges covering ``record_numbers``.

    ``table`` is an open :class:`DBF` (or any object exposing a header
    with ``headerlen``, ``recordlen`` and ``numrecords``).
    ``record_numbers`` are 1-based physical record numbers; the original
    order and any duplicates are preserved in the returned plan.

    ``startup_cost`` is charged once per read, ``byte_cost`` once per
    transferred byte and ``max_length`` caps the length of a single read
    (``None`` means no cap).  Returns a :class:`SparseReadPlan`.
    """
    record_numbers = tuple(record_numbers)
    header = table.header
    headerlen = header.headerlen
    recordlen = header.recordlen

    if not record_numbers:
        return SparseReadPlan((), 0, (), ())

    _validate_record_numbers(record_numbers, header.numrecords)
    pricing, limit = _make_pricing(startup_cost, byte_cost,
                                   max_length, recordlen)

    unique = sorted(set(record_numbers))
    segments = _optimal_segments(unique, recordlen, limit, pricing)
    ranges = tuple(_make_range(headerlen, recordlen, segment)
                   for segment in segments)
    total_cost = sum(read_range.cost(pricing) for read_range in ranges)

    range_by_record = {}
    for read_range in ranges:
        for record_number in read_range.records:
            range_by_record[record_number] = read_range
    record_ranges = tuple(range_by_record[number]
                          for number in record_numbers)

    return SparseReadPlan(ranges, total_cost, record_numbers,
                          record_ranges)


def _build_record(table, parse, slot):
    # In raw mode the whole fixed-length slot is returned, marker and
    # all, mirroring DBF._iter_records().
    if parse is None:
        return slot

    # The first byte is the deletion marker; the rest are the fields.
    position = 1
    items = []
    for field in table.fields:
        raw = slot[position:position + field.length]
        items.append((field.name, parse(field, raw)))
        position += field.length
    return table.recfactory(items)


def execute_sparse_plan(table, plan):
    """Execute ``plan`` against ``table`` and return a result.

    Each planned range is issued as a single read.  Reads that return
    fewer bytes than planned do not raise: the records that are fully
    present are decoded while the rest are reported as short reads with
    their continuation ranges.
    """
    recordlen = table.header.recordlen
    headerlen = table.header.headerlen

    with _open(table.filename, 'rb') as infile, \
            table._open_memofile() as memofile:
        if table.raw:
            parse = None
        else:
            parse = table.parserclass(table, memofile).parse

        chunks = {}
        requests = 0
        bytes_read = 0
        for read_range in plan.ranges:
            infile.seek(read_range.start)
            data = infile.read(read_range.length)
            requests += 1
            bytes_read += len(data)
            chunks[read_range] = data

        records = []
        short_reads = []
        shortfalls = {}

        for record_number in plan.record_order:
            read_range = plan.range_for(record_number)
            slot_start, slot_end = record_offsets(table, record_number)
            data = chunks[read_range]
            have_end = read_range.start + len(data)

            if have_end >= slot_end:
                local_start = slot_start - read_range.start
                slot = data[local_start:local_start + recordlen]
                records.append((record_number,
                                _build_record(table, parse, slot)))
                continue

            if read_range not in shortfalls:
                missing_records = tuple(
                    number for number in read_range.records
                    if headerlen + number * recordlen > have_end)
                shortfalls[read_range] = ReadRange(
                    have_end, read_range.end, missing_records)

            short_reads.append(ShortRead(
                record=record_number,
                have_start=read_range.start,
                have_end=have_end,
                missing=shortfalls[read_range]))

    missing_ranges = tuple(shortfalls[read_range]
                           for read_range in plan.ranges
                           if read_range in shortfalls)

    return SparseReadResult(
        records=records,
        short_reads=short_reads,
        missing_ranges=missing_ranges,
        bytes_read=bytes_read,
        requests=requests,
        plan=plan)


def read_sparse(table, record_numbers, startup_cost=0.0, byte_cost=0.0,
                max_length=None):
    """Plan and immediately execute a sparse read.

    Equivalent to ``execute_sparse_plan(table, plan_sparse_reads(...))``.
    """
    plan = plan_sparse_reads(table, record_numbers, startup_cost,
                             byte_cost, max_length)
    return execute_sparse_plan(table, plan)
