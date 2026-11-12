"""
Align two versions of a DBF table for auditing.

When a table is packed (deleted records physically removed) the
physical record numbers of the surviving records shift, so the old
and the new file can no longer be compared record by record.  This
module scans both files in physical record order (including deleted
records) and aligns them using only two assumptions:

* the relative order of the surviving records is unchanged
* the values of a set of caller declared witness columns are
  unchanged

The alignment maximizes the number of continued (matched) records.
All alignments that are equally optimal are taken into account, and
the records are classified as follows:

* ``continued`` -- record pairs matched in every optimal alignment.
* ``deleted`` / ``added`` -- records matched in no optimal alignment.
* ``ambiguous`` -- segments where duplicate witness values allow
  several optimal alignments.  Every possible destination of each
  old record is listed; no candidate is preferred over the others.

Corrupted or truncated tails of either file are reported separately
as ``unknown`` segments.
"""
from collections import namedtuple, OrderedDict

from .dbf import DBF

__all__ = ['align', 'scan_physical', 'PhysicalRecord', 'RowRef', 'Match',
           'CandidateSet', 'AmbiguousSegment', 'UnknownSegment',
           'AlignmentReport']


class PhysicalRecord(namedtuple('PhysicalRecord',
                                ['number', 'offset', 'length', 'deleted',
                                 'record'])):
    """A record slot in a DBF file, kept or deleted.

    ``number`` is the 0-based physical record number, ``offset`` and
    ``length`` the byte range of the record in the file (including the
    deletion flag byte), ``deleted`` the deletion flag and ``record``
    the parsed record.
    """
    __slots__ = ()

    @property
    def byte_range(self):
        """Byte range of the record as a (start, end) pair."""
        return (self.offset, self.offset + self.length)


class RowRef(namedtuple('RowRef', ['number', 'offset', 'length', 'deleted',
                                   'values'])):
    """Reference to a physical record in one of the aligned tables.

    ``number`` is the 0-based physical record number, ``offset`` and
    ``length`` the byte range of the record in the file (including the
    deletion flag byte), ``deleted`` the deletion flag and ``values``
    the parsed values of the witness columns.
    """
    __slots__ = ()

    @property
    def byte_range(self):
        """Byte range of the record as a (start, end) pair."""
        return (self.offset, self.offset + self.length)


class Match(namedtuple('Match', ['old', 'new'])):
    """A pair of records matched in every optimal alignment."""
    __slots__ = ()

    @property
    def old_byte_range(self):
        """Byte range of the record in the old file."""
        return self.old.byte_range

    @property
    def new_byte_range(self):
        """Byte range of the record in the new file."""
        return self.new.byte_range


class CandidateSet(namedtuple('CandidateSet', ['old', 'new_rows'])):
    """An old record and every new record it may continue as.

    All candidates come from equally optimal alignments, so no
    candidate (for example the one nearest the original record
    number) is preferred over the others.
    """
    __slots__ = ()


class AmbiguousSegment(namedtuple('AmbiguousSegment',
                                  ['old_rows', 'new_rows', 'candidates'])):
    """A segment that cannot be decided due to duplicate witness values.

    ``old_rows`` and ``new_rows`` are the undecidable records on each
    side and ``candidates`` a ``CandidateSet`` for each old record.
    """
    __slots__ = ()

    @property
    def old_byte_range(self):
        """Byte range of the segment in the old file."""
        first = self.old_rows[0]
        last = self.old_rows[-1]
        return (first.offset, last.offset + last.length)

    @property
    def new_byte_range(self):
        """Byte range of the segment in the new file."""
        first = self.new_rows[0]
        last = self.new_rows[-1]
        return (first.offset, last.offset + last.length)


class UnknownSegment(namedtuple('UnknownSegment',
                                ['offset', 'length', 'reason'])):
    """A corrupted or truncated part of a file."""
    __slots__ = ()

    @property
    def byte_range(self):
        """Byte range of the segment as a (start, end) pair."""
        return (self.offset, self.offset + self.length)


class AlignmentReport(namedtuple('AlignmentReport',
                                 ['key_fields', 'continued', 'deleted',
                                  'added', 'ambiguous', 'unknown_old',
                                  'unknown_new'])):
    """Result of aligning two versions of a table.

    ``continued``, ``deleted``, ``added`` and ``ambiguous`` are tuples
    of ``Match``, ``RowRef`` and ``AmbiguousSegment`` objects, and
    ``unknown_old`` / ``unknown_new`` tuples of ``UnknownSegment``
    for corrupted or truncated tails of the two files.
    """
    __slots__ = ()

    @property
    def num_continued(self):
        return len(self.continued)

    @property
    def num_deleted(self):
        return len(self.deleted)

    @property
    def num_added(self):
        return len(self.added)

    @property
    def num_ambiguous(self):
        return len(self.ambiguous)


def _scan_slots(table):
    """Scan a table in physical record order.

    Returns ``(slots, unknown)`` where ``slots`` is a list of
    ``(number, offset, deleted, data)`` tuples (``data`` is the raw
    record data without the deletion flag byte) and ``unknown`` a
    list of ``UnknownSegment`` for corrupted or truncated parts of
    the file.
    """
    slots = []
    unknown = []
    recordlen = table.header.recordlen

    with open(table.filename, 'rb') as infile:
        infile.seek(table.header.headerlen, 0)
        number = 0

        while True:
            offset = infile.tell()
            sep = infile.read(1)

            if sep == b'\x1a':
                # End of records. Anything after the marker is junk.
                rest = infile.read()
                if rest:
                    unknown.append(UnknownSegment(
                        offset + 1, len(rest),
                        'data after end-of-file marker'))
                break
            elif sep == b'':
                # End of file.
                break

            data = infile.read(recordlen - 1)
            if len(data) < recordlen - 1:
                unknown.append(UnknownSegment(
                    offset, 1 + len(data), 'truncated record'))
                break

            if sep in (b' ', b'*'):
                slots.append((number, offset, sep == b'*', data))
            else:
                unknown.append(UnknownSegment(
                    offset, recordlen,
                    'invalid deletion flag 0x{:02x}'.format(ord(sep))))
            number += 1

    return slots, unknown


def scan_physical(table):
    """Scan a table in physical record order, including deleted records.

    ``table`` is a ``DBF`` object or a file name.  Returns
    ``(records, unknown)`` where ``records`` is a list of
    ``PhysicalRecord`` and ``unknown`` a list of ``UnknownSegment``
    for corrupted or truncated parts of the file.
    """
    if not isinstance(table, DBF):
        table = DBF(table)

    slots, unknown = _scan_slots(table)
    recordlen = table.header.recordlen

    if table.raw:
        records = [PhysicalRecord(number, offset, recordlen, deleted, data)
                   for number, offset, deleted, data in slots]
    else:
        with table._open_memofile() as memofile:
            field_parser = table.parserclass(table, memofile)
            parse = field_parser.parse

            records = []
            for number, offset, deleted, data in slots:
                items = []
                pos = 0
                for field in table.fields:
                    items.append((field.name,
                                  parse(field, data[pos:pos + field.length])))
                    pos += field.length
                records.append(PhysicalRecord(number, offset, recordlen,
                                              deleted,
                                              table.recfactory(items)))

    return records, unknown


def _scan_rows(table, key_fields):
    """Scan a table and build RowRefs and witness keys for alignment."""
    slots, unknown = _scan_slots(table)
    recordlen = table.header.recordlen

    # Byte offset of each field within the record data.
    offsets = {}
    pos = 0
    for field in table.fields:
        offsets[field.name] = pos
        pos += field.length

    field_by_name = dict((field.name, field) for field in table.fields)
    witness = [(name, field_by_name[name]) for name in key_fields]

    if table.raw:
        parse = None
    else:
        parse = table.parserclass(table).parse

    rows = []
    keys = []
    for number, offset, deleted, data in slots:
        key = []
        values = OrderedDict()
        for name, field in witness:
            start = offsets[name]
            raw = data[start:start + field.length]
            # Witness columns are compared by their raw bytes, which
            # is the strictest interpretation of "unchanged".
            key.append(raw)
            if parse is None:
                values[name] = raw
            else:
                try:
                    values[name] = parse(field, raw)
                except Exception:
                    values[name] = raw
        keys.append(tuple(key))
        rows.append(RowRef(number, offset, recordlen, deleted, values))

    return rows, keys, unknown


def _lcs_tables(a, b):
    """Longest common subsequence tables for two key sequences.

    Returns ``(pref, suff)`` where ``pref[i][j]`` is the LCS length of
    ``a[:i]`` and ``b[:j]`` and ``suff[i][j]`` the LCS length of
    ``a[i:]`` and ``b[j:]``.
    """
    m, n = len(a), len(b)

    pref = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        ai = a[i]
        row = pref[i]
        next_row = pref[i + 1]
        for j in range(n):
            if ai == b[j]:
                next_row[j + 1] = row[j] + 1
            elif row[j + 1] >= next_row[j]:
                next_row[j + 1] = row[j + 1]
            else:
                next_row[j + 1] = next_row[j]

    suff = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        ai = a[i]
        row = suff[i]
        next_row = suff[i + 1]
        for j in range(n - 1, -1, -1):
            if ai == b[j]:
                row[j] = next_row[j + 1] + 1
            elif next_row[j] >= row[j + 1]:
                row[j] = next_row[j]
            else:
                row[j] = row[j + 1]

    return pref, suff


def _align_keys(a, b):
    """Align two sequences of witness keys.

    Returns ``(continued, deleted, added, ambiguous_old,
    ambiguous_new, candidates)`` where ``continued`` is a list of
    ``(old_index, new_index)`` pairs, ``deleted`` and ``added`` lists
    of indexes, ``ambiguous_old`` and ``ambiguous_new`` lists of
    indexes and ``candidates`` a dict mapping an ambiguous old index
    to the tuple of new indexes it may continue as in some optimal
    alignment.
    """
    m, n = len(a), len(b)
    pref, suff = _lcs_tables(a, b)
    lcs = pref[m][n]

    # A record is essential if removing it shortens the longest common
    # subsequence, which means it is matched in every optimal
    # alignment.  LCS(a without record i, b) can be computed as
    # max_j(pref[i][j] + suff[i+1][j]) by splitting b at j.
    essential_old = []
    for i in range(m):
        row = pref[i]
        next_row = suff[i + 1]
        best = 0
        for j in range(n + 1):
            best = max(best, row[j] + next_row[j])
        essential_old.append(best < lcs)

    essential_new = []
    for j in range(n):
        best = 0
        for i in range(m + 1):
            best = max(best, pref[i][j] + suff[i][j + 1])
        essential_new.append(best < lcs)

    # Pair (i, j) is feasible -- part of some optimal alignment -- if
    # the keys match and the best alignment through it is optimal.
    feasible_old = [[] for _ in range(m)]
    feasible_new = [[] for _ in range(n)]
    for i in range(m):
        ai = a[i]
        row = pref[i]
        next_row = suff[i + 1]
        for j in range(n):
            if ai == b[j] and row[j] + 1 + next_row[j + 1] == lcs:
                feasible_old[i].append(j)
                feasible_new[j].append(i)

    continued = []
    deleted = []
    ambiguous_old = []
    candidates = {}
    for i in range(m):
        cands = feasible_old[i]
        if not cands:
            # Matched in no optimal alignment.
            deleted.append(i)
            continue
        if essential_old[i]:
            partners = [j for j in cands if essential_new[j]]
            if partners:
                # Matched to the same record in every optimal
                # alignment.
                continued.append((i, partners[0]))
                continue
        ambiguous_old.append(i)
        candidates[i] = tuple(cands)

    continued_new = set(j for _, j in continued)
    added = []
    ambiguous_new = []
    for j in range(n):
        if not feasible_new[j]:
            added.append(j)
        elif j not in continued_new:
            ambiguous_new.append(j)

    return (continued, deleted, added, ambiguous_old, ambiguous_new,
            candidates)


def align(old, new, key_fields):
    """Align two versions of a DBF table for auditing.

    ``old`` and ``new`` are ``DBF`` objects or file names and
    ``key_fields`` the names of the witness columns.  Returns an
    ``AlignmentReport``.
    """
    if not isinstance(old, DBF):
        old = DBF(old)
    if not isinstance(new, DBF):
        new = DBF(new)

    if isinstance(key_fields, str):
        key_fields = [key_fields]
    key_fields = list(key_fields)

    for table, side in ((old, 'old'), (new, 'new')):
        missing = [name for name in key_fields
                   if name not in table.field_names]
        if missing:
            raise ValueError(
                'witness columns not found in {} table: {}'.format(
                    side, ', '.join(missing)))

    old_rows, old_keys, unknown_old = _scan_rows(old, key_fields)
    new_rows, new_keys, unknown_new = _scan_rows(new, key_fields)

    (continued, deleted, added, ambiguous_old, ambiguous_new,
     candidates) = _align_keys(old_keys, new_keys)

    # Continued matches are anchors that split both files into
    # independent blocks.  Ambiguous records inside one block can
    # only refer to each other, so each block with ambiguous records
    # becomes one segment.
    segments = []
    prev_old, prev_new = -1, -1
    for anchor_old, anchor_new in continued + [(len(old_rows),
                                                len(new_rows))]:
        seg_old = [i for i in ambiguous_old if prev_old < i < anchor_old]
        seg_new = [j for j in ambiguous_new if prev_new < j < anchor_new]
        if seg_old or seg_new:
            segments.append(AmbiguousSegment(
                old_rows=tuple(old_rows[i] for i in seg_old),
                new_rows=tuple(new_rows[j] for j in seg_new),
                candidates=tuple(
                    CandidateSet(old_rows[i],
                                 tuple(new_rows[j] for j in candidates[i]))
                    for i in seg_old)))
        prev_old, prev_new = anchor_old, anchor_new

    return AlignmentReport(
        key_fields=tuple(key_fields),
        continued=tuple(Match(old_rows[i], new_rows[j])
                        for i, j in continued),
        deleted=tuple(old_rows[i] for i in deleted),
        added=tuple(new_rows[j] for j in added),
        ambiguous=tuple(segments),
        unknown_old=tuple(unknown_old),
        unknown_new=tuple(unknown_new))
