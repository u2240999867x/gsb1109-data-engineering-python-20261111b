"""
Tests for the read-only header auditor.

The tests build DBF byte strings in memory, including headers where the
stored descriptor offsets and the offsets implied by cumulative field
widths disagree.
"""
import os
import struct
import pytest

from dbfread import DBF
from dbfread.header_audit import (
    HeaderAuditor,
    FormatVariant,
    AuditFailure,
    AmbiguousHeader,
    audit_header,
)

HEADER_SIZE = 32
FIELD_SIZE = 32
HEADER_STRUCT = struct.Struct('<BBBBLHHHBBLLLBBH')
FIELD_STRUCT = struct.Struct('<11scLBBHBBBB7sB')


def build_descriptor(name, field_type, length, address=0, decimal_count=0,
                     reserved1=0, workarea_id=0, reserved2=0, reserved3=0,
                     set_fields_flag=0, reserved4=b'\x00' * 7,
                     index_field_flag=0):
    return FIELD_STRUCT.pack(
        name.encode('ascii').ljust(11, b'\x00'),
        field_type.encode('ascii'),
        address,
        length,
        decimal_count,
        reserved1,
        workarea_id,
        reserved2,
        reserved3,
        set_fields_flag,
        reserved4,
        index_field_flag)


def build_dbf(descriptors, records, recordlen, version=0x30):
    numrecords = len(records)
    headerlen = HEADER_SIZE + FIELD_SIZE * len(descriptors) + 1
    header = HEADER_STRUCT.pack(
        version,        # dbversion
        122, 1, 1,      # year, month, day
        numrecords,
        headerlen,
        recordlen,
        0,              # reserved1
        0, 0,           # transaction / encryption flags
        0, 0, 0,        # free record thread, reserved2, reserved3
        0,              # mdx flag
        0,              # language driver -> ascii
        0)              # reserved4
    body = header + b''.join(descriptors) + b'\x0d'
    for marker, record_body in records:
        body += marker + record_body
    body += b'\x1a'
    return body


@pytest.fixture
def ambiguous_bytes():
    # Two char fields of width 4. Cumulative layout puts B at [5, 9),
    # the stored addresses put B at [6, 10). recordlen is 10, so the
    # cumulative layout has one byte of trailing padding.
    descriptors = [
        build_descriptor('A', 'C', 4, address=1),
        build_descriptor('B', 'C', 4, address=6),
    ]
    record_body = b'AAAA' + b'BBBB' + b'X'
    return build_dbf(descriptors,
                     [(b' ', record_body), (b' ', record_body)],
                     recordlen=10)


@pytest.fixture
def permissive_variant():
    return FormatVariant(
        'test_vfp', (0x30,),
        strategies=('cumulative', 'address'),
        require_contiguous=False,
        allow_trailing_padding=True)


def write(tmp_path, data, name='sample.dbf'):
    path = tmp_path.joinpath(name)
    path.write_bytes(data)
    return str(path)


def test_clean_header_has_unique_layout(tmp_path):
    descriptors = [
        build_descriptor('NAME', 'C', 10, address=1),
        build_descriptor('AGE', 'N', 3, address=11),
    ]
    records = [(b' ', b'Alice'.ljust(10, b' ') + b' 30'),
               (b' ', b'Bob'.ljust(10, b' ') + b' 42')]
    data = build_dbf(descriptors, records, recordlen=14)
    report = audit_header(write(tmp_path, data))

    assert report.unique
    assert report.layout.strategy == 'cumulative'
    assert [(f.name, f.offset, f.length) for f in report.layout.fields] == [
        ('NAME', 1, 10), ('AGE', 11, 3)]
    assert report.disputed_fields == []


def test_ambiguous_header_reports_consensus_and_dispute(
        tmp_path, ambiguous_bytes, permissive_variant):
    path = write(tmp_path, ambiguous_bytes)
    report = HeaderAuditor([permissive_variant]).audit(path)

    assert not report.unique
    assert len(report.valid_candidates) == 2
    consensus_names = [field.name for field in report.consensus_fields]
    assert consensus_names == ['A']
    # Deletion marker [0, 1) plus consensus field A [1, 5).
    assert report.agreed_ranges == [(0, 5)]

    assert len(report.disputed_fields) == 1
    dispute = report.disputed_fields[0]
    assert dispute['name'] == 'B'
    pair = dispute['pair']
    assert pair['candidates'] == ('test_vfp:cumulative',
                                  'test_vfp:address')
    assert pair['ranges'] == ((5, 9), (6, 10))
    assert pair['raw_values'] == (b'BBBB', b'BBBX')

    with pytest.raises(AmbiguousHeader) as excinfo:
        report.raise_if_ambiguous()
    assert excinfo.value.report is report


def test_dbf_refuses_to_decode_ambiguous_header(
        tmp_path, ambiguous_bytes, permissive_variant):
    path = write(tmp_path, ambiguous_bytes)
    before = open(path, 'rb').read()

    with pytest.raises(AmbiguousHeader):
        DBF(path, header_audit=True,
            format_variants=[permissive_variant])

    # The original file must be left untouched.
    assert open(path, 'rb').read() == before


def test_known_rows_select_unique_layout_and_decode(
        tmp_path, ambiguous_bytes, permissive_variant):
    path = write(tmp_path, ambiguous_bytes)
    known_rows = [{'B': b'BBBX'}]
    table = DBF(path, header_audit=True,
                format_variants=[permissive_variant],
                known_rows=known_rows)

    assert table.audit_report.unique
    assert table.audit_report.layout.strategy == 'address'
    rows = list(table.records)
    assert rows == [{'A': 'AAAA', 'B': 'BBBX'},
                    {'A': 'AAAA', 'B': 'BBBX'}]


def test_known_rows_that_match_nothing_fail(
        tmp_path, ambiguous_bytes, permissive_variant):
    path = write(tmp_path, ambiguous_bytes)
    with pytest.raises(AuditFailure):
        DBF(path, header_audit=True,
            format_variants=[permissive_variant],
            known_rows=[{'B': b'NOPE'}])


def test_reserved_byte_violation_rejects_every_layout(tmp_path):
    descriptors = [
        build_descriptor('A', 'C', 4, reserved1=1),
        build_descriptor('B', 'C', 4),
    ]
    data = build_dbf(descriptors,
                     [(b' ', b'AAAABBBB')], recordlen=9)
    with pytest.raises(AuditFailure) as excinfo:
        audit_header(write(tmp_path, data)).raise_if_ambiguous()
    assert 'reserved1' in str(excinfo.value)


def test_unknown_version_byte_has_no_declared_variant(tmp_path):
    descriptors = [build_descriptor('A', 'C', 4)]
    data = build_dbf(descriptors, [(b' ', b'AAAA')],
                     recordlen=5, version=0x77)
    with pytest.raises(AuditFailure) as excinfo:
        audit_header(write(tmp_path, data))
    assert '0x77' in str(excinfo.value)


def test_truncated_file_is_rejected(tmp_path):
    descriptors = [build_descriptor('A', 'C', 4)]
    data = build_dbf(descriptors,
                     [(b' ', b'AAAA'), (b' ', b'AAAA')],
                     recordlen=5)
    truncated = data[:-6]
    with pytest.raises(AuditFailure) as excinfo:
        audit_header(write(tmp_path, truncated)).raise_if_ambiguous()
    assert 'at least' in str(excinfo.value)


def test_gapped_unique_layout_decodes_at_audited_offsets(tmp_path):
    # Address layout leaves a one byte gap between A [1, 5) and
    # B [6, 9); the gap byte is skipped while decoding.
    variant = FormatVariant(
        'gapped', (0x30,), strategies=('address',),
        require_contiguous=False)
    descriptors = [
        build_descriptor('A', 'C', 4, address=1),
        build_descriptor('B', 'C', 3, address=6),
    ]
    record_body = b'AAAA' + b'_' + b'BBB'
    data = build_dbf(descriptors, [(b' ', record_body)], recordlen=9)

    table = DBF(write(tmp_path, data), header_audit=True,
                format_variants=[variant])
    assert table._contiguous_fields is False
    assert list(table.records) == [{'A': 'AAAA', 'B': 'BBB'}]


def test_audit_does_not_change_default_reader_path(tmp_path):
    descriptors = [build_descriptor('NAME', 'C', 5)]
    data = build_dbf(descriptors, [(b' ', b'Carol')], recordlen=6)
    path = write(tmp_path, data)

    default_rows = list(DBF(path))
    audited_rows = list(DBF(path, header_audit=True))
    assert default_rows == audited_rows == [{'NAME': 'Carol'}]


def test_rejected_candidates_carry_reasons(
        tmp_path, ambiguous_bytes, permissive_variant):
    # A strict contiguous variant rejects the address strategy because
    # of the gap, while the permissive one keeps both interpretations.
    strict = FormatVariant(
        'strict', (0x30,), strategies=('cumulative', 'address'),
        require_contiguous=True, allow_trailing_padding=True)
    path = write(tmp_path, ambiguous_bytes)
    report = HeaderAuditor([strict, permissive_variant]).audit(path)

    rejected = [c for c in report.candidates if not c.valid]
    assert any('strict:address' in c.name and c.reasons for c in rejected)
    assert not report.unique


def test_audit_bytes_works_without_a_file(ambiguous_bytes,
                                          permissive_variant):
    report = HeaderAuditor([permissive_variant]).audit_bytes(
        ambiguous_bytes, filename='<memory>')
    assert not report.unique
    dispute = report.disputed_fields[0]
    assert dispute['pair']['raw_values'] == (b'BBBB', b'BBBX')


def test_header_backlink_is_allowed_only_for_variants_that_declare_it():
    descriptors = [build_descriptor('A', 'C', 4, address=1)]
    data = build_dbf(descriptors, [(b' ', b'AAAA')], recordlen=5)

    # One field means a 65 byte header. Insert a 263 byte backlink area
    # right after the terminator and patch headerlen accordingly.
    assert data[64] == 0x0D
    backlink = b'\x00' * 263
    new_headerlen = 65 + len(backlink)
    data = data[:65] + backlink + data[65:]
    data = data[:8] + struct.pack('<H', new_headerlen) + data[10:]

    default_report = HeaderAuditor().audit_bytes(data)
    assert default_report.unique

    strict = FormatVariant('no_backlink', (0x30,),
                           strategies=('cumulative', 'address'))
    strict_report = HeaderAuditor([strict]).audit_bytes(data)
    with pytest.raises(AuditFailure):
        strict_report.raise_if_ambiguous()


def test_file_is_read_only(tmp_path, ambiguous_bytes, permissive_variant):
    path = write(tmp_path, ambiguous_bytes)
    mode_before = os.stat(path).st_mode
    report = HeaderAuditor([permissive_variant]).audit(path)
    with pytest.raises(AmbiguousHeader):
        report.raise_if_ambiguous()
    assert os.stat(path).st_mode == mode_before
