"""
Tests for dbfread.align (physical order scan and alignment).
"""
import struct

from pytest import fixture, raises

from dbfread import DBF, align, scan_physical
from dbfread.align import (AlignmentReport, AmbiguousSegment, CandidateSet,
                           Match, PhysicalRecord, RowRef, UnknownSegment)


def make_dbf(path, fields, rows, eof_marker=True, trailer=b''):
    """Write a minimal dBase III file.

    ``fields`` is a list of (name, type, length) tuples (only 'C' and
    'N' are supported) and ``rows`` a list of (deleted, values) tuples
    where values are strings.
    """
    headerlen = 32 + 32 * len(fields) + 1
    recordlen = 1 + sum(length for _, _, length in fields)

    header = struct.pack(
        '<BBBBLHHHBBLLLBBH',
        0x03,          # dbversion
        126, 11, 13,   # date
        len(rows),
        headerlen,
        recordlen,
        0, 0, 0, 0, 0, 0, 0, 0, 0)

    for name, type_, length in fields:
        header += struct.pack(
            '<11scLBBHBBBB7sB',
            name.encode('ascii'),
            type_.encode('ascii'),
            0, length, 0, 0, 0, 0, 0, 0, b'\0' * 7, 0)
    header += b'\r'

    data = b''
    for deleted, values in rows:
        data += b'*' if deleted else b' '
        for (name, type_, length), value in zip(fields, values):
            if type_ == 'C':
                data += value.ljust(length).encode('ascii')[:length]
            else:
                data += value.rjust(length).encode('ascii')[-length:]

    if eof_marker:
        data += b'\x1a'
    data += trailer

    with open(str(path), 'wb') as outfile:
        outfile.write(header + data)

    return headerlen, recordlen


FIELDS = [('NAME', 'C', 10), ('AGE', 'N', 3)]


@fixture
def paths(tmpdir):
    return str(tmpdir.join('old.dbf')), str(tmpdir.join('new.dbf'))


def test_pack_shift(paths):
    old_path, new_path = paths
    old_headerlen, recordlen = make_dbf(old_path, FIELDS, [
        (False, ['Alice', '30']),
        (True, ['Bob', '40']),
        (False, ['Carol', '50']),
        (False, ['Dave', '60']),
    ])
    make_dbf(new_path, FIELDS, [
        (False, ['Alice', '30']),
        (False, ['Carol', '50']),
        (False, ['Dave', '60']),
    ])

    report = align(old_path, new_path, ['NAME'])

    assert isinstance(report, AlignmentReport)
    assert report.key_fields == ('NAME',)
    assert report.num_continued == 3
    assert report.num_deleted == 1
    assert report.num_added == 0
    assert report.ambiguous == ()
    assert report.unknown_old == ()
    assert report.unknown_new == ()

    # Physical record numbers shift after the packed record.
    pairs = [(match.old.number, match.new.number)
             for match in report.continued]
    assert pairs == [(0, 0), (2, 1), (3, 2)]

    # Byte ranges are reported on both sides.
    match = report.continued[2]
    assert isinstance(match, Match)
    assert match.old.byte_range == (old_headerlen + 3 * recordlen,
                                    old_headerlen + 4 * recordlen)
    assert match.new_byte_range == (old_headerlen + 2 * recordlen,
                                    old_headerlen + 3 * recordlen)
    assert match.old_byte_range == (match.old.offset,
                                    match.old.offset + match.old.length)

    deleted = report.deleted[0]
    assert isinstance(deleted, RowRef)
    assert deleted.number == 1
    assert deleted.deleted is True
    assert deleted.values['NAME'] == 'Bob'


def test_added_and_deleted(paths):
    old_path, new_path = paths
    make_dbf(old_path, FIELDS, [
        (False, ['Alice', '30']),
        (False, ['Bob', '40']),
    ])
    make_dbf(new_path, FIELDS, [
        (False, ['Alice', '30']),
        (False, ['Eve', '20']),
    ])

    report = align(old_path, new_path, ['NAME', 'AGE'])

    assert [(m.old.number, m.new.number) for m in report.continued] == [(0, 0)]
    assert [row.number for row in report.deleted] == [1]
    assert [row.number for row in report.added] == [1]
    assert report.deleted[0].values['NAME'] == 'Bob'
    assert report.added[0].values['NAME'] == 'Eve'
    assert report.ambiguous == ()


def test_ambiguous_duplicate_values(paths):
    old_path, new_path = paths
    old_headerlen, recordlen = make_dbf(old_path, FIELDS, [
        (False, ['Alice', '30']),
        (False, ['Alice', '30']),
    ])
    make_dbf(new_path, FIELDS, [
        (False, ['Alice', '30']),
        (False, ['Alice', '30']),
        (False, ['Alice', '30']),
    ])

    report = align(old_path, new_path, ['NAME', 'AGE'])

    # Any two of the three identical new records could be the
    # continuations, so no pair is matched in every optimal
    # alignment and the whole file is one ambiguous segment.
    assert report.continued == ()
    assert report.deleted == ()
    assert report.added == ()

    assert len(report.ambiguous) == 1
    segment = report.ambiguous[0]
    assert isinstance(segment, AmbiguousSegment)
    assert [row.number for row in segment.old_rows] == [0, 1]
    assert [row.number for row in segment.new_rows] == [0, 1, 2]
    assert segment.old_byte_range == (old_headerlen,
                                      old_headerlen + 2 * recordlen)

    # All candidates are listed, not only the nearest one.
    assert len(segment.candidates) == 2
    candidates = segment.candidates[0]
    assert isinstance(candidates, CandidateSet)
    assert candidates.old.number == 0
    assert [row.number for row in candidates.new_rows] == [0, 1]
    assert [row.number for row in segment.candidates[1].new_rows] == [1, 2]


def test_ambiguous_all_candidates_listed(paths):
    old_path, new_path = paths
    make_dbf(old_path, FIELDS, [(False, ['Alice', '30'])])
    make_dbf(new_path, FIELDS, [
        (False, ['Alice', '30']),
        (False, ['Alice', '30']),
    ])

    report = align(old_path, new_path, 'NAME')

    # Either of the two new records could be the continuation, so
    # nothing is certain and both candidates are listed.
    assert report.continued == ()
    assert report.deleted == ()
    assert report.added == ()
    assert len(report.ambiguous) == 1
    segment = report.ambiguous[0]
    assert [row.number for row in segment.old_rows] == [0]
    assert [row.number for row in segment.new_rows] == [0, 1]
    assert [row.number for row in segment.candidates[0].new_rows] == [0, 1]


def test_ambiguous_segments_split_by_anchors(paths):
    old_path, new_path = paths
    make_dbf(old_path, FIELDS, [
        (False, ['Ann', '1']),
        (False, ['Ann', '1']),
        (False, ['Bob', '2']),
        (False, ['Cid', '3']),
        (False, ['Cid', '3']),
    ])
    make_dbf(new_path, FIELDS, [
        (False, ['Ann', '1']),
        (False, ['Bob', '2']),
        (False, ['Cid', '3']),
    ])

    report = align(old_path, new_path, ['NAME', 'AGE'])

    # Only the unique Bob pair is certain.  The duplicated Ann and
    # Cid records form one ambiguous segment each, split by the
    # continued anchors.
    assert [(m.old.number, m.new.number) for m in report.continued] == \
        [(2, 1)]
    assert len(report.ambiguous) == 2
    assert [row.number for row in report.ambiguous[0].old_rows] == [0, 1]
    assert [row.number for row in report.ambiguous[0].new_rows] == [0]
    assert [row.number for row in report.ambiguous[1].old_rows] == [3, 4]
    assert [row.number for row in report.ambiguous[1].new_rows] == [2]
    assert [c.old.number for c in report.ambiguous[0].candidates] == [0, 1]
    assert [row.number
            for row in report.ambiguous[0].candidates[0].new_rows] == [0]


def test_truncated_tail_is_unknown(paths):
    old_path, new_path = paths
    make_dbf(old_path, FIELDS, [
        (False, ['Alice', '30']),
        (False, ['Bob', '40']),
    ])
    headerlen, recordlen = make_dbf(new_path, FIELDS, [
        (False, ['Alice', '30']),
        (False, ['Bob', '40']),
    ])

    # Truncate the new file in the middle of the second record.
    with open(new_path, 'rb') as infile:
        data = infile.read()
    cut = headerlen + recordlen + 5
    with open(new_path, 'wb') as outfile:
        outfile.write(data[:cut])

    report = align(old_path, new_path, ['NAME'])

    assert [(m.old.number, m.new.number) for m in report.continued] == [(0, 0)]
    assert [row.number for row in report.deleted] == [1]
    assert report.unknown_old == ()
    assert len(report.unknown_new) == 1
    unknown = report.unknown_new[0]
    assert isinstance(unknown, UnknownSegment)
    assert unknown.reason == 'truncated record'
    assert unknown.byte_range == (headerlen + recordlen, cut)


def test_data_after_eof_marker_is_unknown(paths):
    old_path, new_path = paths
    make_dbf(old_path, FIELDS, [(False, ['Alice', '30'])])
    headerlen, recordlen = make_dbf(new_path, FIELDS,
                                    [(False, ['Alice', '30'])],
                                    trailer=b'junk')

    report = align(old_path, new_path, ['NAME'])

    assert report.num_continued == 1
    assert len(report.unknown_new) == 1
    unknown = report.unknown_new[0]
    assert unknown.reason == 'data after end-of-file marker'
    assert unknown.byte_range == (headerlen + recordlen + 1,
                                  headerlen + recordlen + 5)


def test_missing_witness_column(paths):
    old_path, new_path = paths
    make_dbf(old_path, FIELDS, [(False, ['Alice', '30'])])
    make_dbf(new_path, FIELDS, [(False, ['Alice', '30'])])

    with raises(ValueError):
        align(old_path, new_path, ['NOSUCH'])


def test_scan_physical(paths):
    old_path, _ = paths
    headerlen, recordlen = make_dbf(old_path, FIELDS, [
        (False, ['Alice', '30']),
        (True, ['Bob', '40']),
        (False, ['Carol', '50']),
    ])

    records, unknown = scan_physical(old_path)

    assert unknown == []
    assert len(records) == 3
    assert all(isinstance(record, PhysicalRecord) for record in records)
    assert [record.number for record in records] == [0, 1, 2]
    assert [record.deleted for record in records] == [False, True, False]
    assert [record.record['NAME'] for record in records] == \
        ['Alice', 'Bob', 'Carol']
    assert records[1].byte_range == (headerlen + recordlen,
                                     headerlen + 2 * recordlen)


def test_scan_physical_dbf_object(paths):
    old_path, _ = paths
    make_dbf(old_path, FIELDS, [(False, ['Alice', '30'])])

    table = DBF(old_path)
    records, unknown = scan_physical(table)

    assert len(records) == 1
    assert records[0].record['NAME'] == 'Alice'
