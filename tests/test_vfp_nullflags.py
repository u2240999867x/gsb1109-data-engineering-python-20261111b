"""
Tests for Visual FoxPro null value flags (_NullFlags).
"""
import struct

import pytest

from dbfread import DBF
from dbfread.exceptions import NullFlagsUnavailable

HEADER_FORMAT = '<BBBBLHHHBBLLLBBH'
FIELD_FORMAT = '<11scLBBHBBBB7sB'

NULLABLE = 0x02
SYSTEM = 0x01


def pack_field(name, type_, length, decimals=0, flags=0):
    return struct.pack(
        FIELD_FORMAT,
        name.encode('ascii'),
        type_.encode('ascii'),
        0,          # address
        length,
        decimals,
        0,          # reserved1
        0,          # workarea_id
        0,          # reserved2
        0,          # reserved3
        flags,      # set_fields_flag
        b'\0' * 7,  # reserved4
        0,          # index_field_flag
    )


def write_dbf(path, fields, records, recordlen_delta=0):
    """Write a minimal Visual FoxPro DBF file.

    fields is a list of (name, type, length, decimals, flags) tuples
    and records is a list of (separator, body) tuples where body is
    the raw record data without the deletion flag.
    """
    headerlen = 32 + 32 * len(fields) + 1
    recordlen = 1 + sum(field[2] for field in fields) + recordlen_delta

    header = struct.pack(
        HEADER_FORMAT,
        0x30,           # dbversion: Visual FoxPro
        126, 11, 13,    # date
        len(records),
        headerlen,
        recordlen,
        0, 0, 0, 0, 0, 0, 0,
        0,              # language_driver
        0,
    )

    with open(str(path), 'wb') as outfile:
        outfile.write(header)
        for field in fields:
            outfile.write(pack_field(*field))
        outfile.write(b'\r')
        for sep, body in records:
            outfile.write(sep)
            assert len(body) == recordlen - 1
            outfile.write(body)
        outfile.write(b'\x1a')


@pytest.fixture
def vfp_path(tmp_path):
    """A VFP table mixing nullable fixed length fields and a nullable
    varchar field.

    Null flag bits: TAG null=0, TAG varlength=1, AMOUNT null=2,
    NOTE null=3.
    """
    fields = [
        ('NAME', 'C', 8, 0, 0),
        ('TAG', 'V', 6, 0, NULLABLE),
        ('AMOUNT', 'N', 8, 2, NULLABLE),
        ('NOTE', 'C', 5, 0, NULLABLE),
        ('_NullFlags', '0', 1, 0, SYSTEM),
    ]

    def rec(name, tag, amount, note, nullflags):
        return name + tag + amount + note + bytes([nullflags])

    records = [
        # Zero amount (not null).
        (b' ', rec(b'Alice   ', b'ab    ', b'    0.00', b'x    ', 0x00)),
        # Deleted record occupying physical record number 2.
        (b'*', rec(b'Del     ', b'      ', b'        ', b'     ', 0x00)),
        # Blank amount (bit clear), NOTE is NULL (bit 3 set).
        (b' ', rec(b'Bob     ', b'      ', b'        ', b'     ', 0x08)),
        # TAG and AMOUNT are NULL (bits 0 and 2 set).
        (b' ', rec(b'Cara    ', b'      ', b'        ', b'hi   ', 0x05)),
        # Only the varchar varlength bit (bit 1) is set.
        (b' ', rec(b'Dan     ', b'zz    ', b'    7.25', b'n    ', 0x02)),
    ]

    path = tmp_path / 'detail.dbf'
    write_dbf(path, fields, records)
    return path


@pytest.fixture
def truncated_path(tmp_path):
    """A VFP table whose _NullFlags system area is one byte short.

    Ten nullable numeric fields need ten null flag bits (two bytes),
    but the record only has room for the first byte.
    """
    fields = [('F{}'.format(i), 'N', 4, 0, NULLABLE) for i in range(10)]
    fields.append(('_NullFlags', '0', 2, 0, SYSTEM))

    # F8 contains data (provably not null), F9 is empty and its null
    # bit lives in the missing second byte.
    body = b'    ' * 8 + b'   1' + b'    ' + b'\x00'
    records = [(b' ', body)]

    path = tmp_path / 'truncated.dbf'
    write_dbf(path, fields, records, recordlen_delta=-1)
    return path


def test_nullflags_field_is_interpreted_not_returned(vfp_path):
    table = DBF(str(vfp_path))

    assert '_NullFlags' not in table.field_names
    assert '_NullFlags' not in list(table)[0]
    # The physical field header is still available.
    assert any(field.name == '_NullFlags' for field in table.fields)


def test_zero_blank_and_null_are_distinct(vfp_path):
    table = DBF(str(vfp_path))
    records = list(table)

    # Zero, blank and NULL all parse differently only through the
    # null flags, so the meta output must tell them apart.
    assert records[0]['AMOUNT'] == 0.0      # zero
    assert records[1]['AMOUNT'] is None     # blank (not null)
    assert records[2]['AMOUNT'] is None     # NULL

    meta = list(table.iter_with_meta())
    assert meta[0][2]['AMOUNT'] == (2, False)   # zero
    assert meta[1][2]['AMOUNT'] == (2, False)   # blank
    assert meta[2][2]['AMOUNT'] == (2, True)    # NULL


def test_null_bit_positions(vfp_path):
    table = DBF(str(vfp_path))

    # The varchar field TAG occupies an extra bit (varlength), so
    # AMOUNT's null flag is bit 2, not bit 1.
    assert table.null_bit_positions == {
        'NAME': None,
        'TAG': 0,
        'AMOUNT': 2,
        'NOTE': 3,
    }


def test_nullable_varchar_mixed_with_fixed_length(vfp_path):
    records = list(DBF(str(vfp_path)))

    assert records[0] == {'NAME': 'Alice', 'TAG': 'ab',
                          'AMOUNT': 0.0, 'NOTE': 'x'}
    assert records[1] == {'NAME': 'Bob', 'TAG': '',
                          'AMOUNT': None, 'NOTE': None}
    assert records[2] == {'NAME': 'Cara', 'TAG': None,
                          'AMOUNT': None, 'NOTE': 'hi'}
    assert records[3] == {'NAME': 'Dan', 'TAG': 'zz',
                          'AMOUNT': 7.25, 'NOTE': 'n'}


def test_iter_with_recno_uses_physical_record_numbers(vfp_path):
    table = DBF(str(vfp_path))

    # Record 2 is deleted but still occupies a physical slot.
    assert [recno for recno, _ in table.iter_with_recno()] == [1, 3, 4, 5]
    assert [recno for recno, _ in table.iter_with_recno(b'*')] == [2]


def test_iter_with_meta(vfp_path):
    meta = list(DBF(str(vfp_path)).iter_with_meta())

    recno, record, nulls = meta[2]
    assert recno == 4
    assert record['NAME'] == 'Cara'
    assert nulls == {'TAG': (0, True), 'AMOUNT': (2, True),
                     'NOTE': (3, False)}


def test_projection_reconciles_with_full_read(vfp_path):
    full = [record['AMOUNT'] for record in DBF(str(vfp_path))]
    projected = [record['AMOUNT']
                 for record in DBF(str(vfp_path), include_fields=['AMOUNT'])]

    assert projected == full == [0.0, None, None, 7.25]


def test_projection_does_not_shift_bit_positions(vfp_path):
    table = DBF(str(vfp_path), include_fields=['AMOUNT'])

    # Skipping the fields before AMOUNT must not reassign the bits
    # they occupy (including the varchar field's extra bit) to AMOUNT.
    assert table.field_names == ['AMOUNT']
    assert table.null_bit_positions['AMOUNT'] == 2

    # NOTE comes after the varchar field; its null flag is bit 3 even
    # when it is the only column read.
    notes = [record['NOTE']
             for record in DBF(str(vfp_path), include_fields=['NOTE'])]
    assert notes == ['x', None, 'hi', 'n']


def test_projection_of_unknown_field_raises(vfp_path):
    with pytest.raises(ValueError):
        DBF(str(vfp_path), include_fields=['AMOUNT', 'NOSUCH'])


def test_raw_mode_returns_nullflags_as_is(vfp_path):
    record = list(DBF(str(vfp_path), raw=True))[0]

    assert record['_NullFlags'] == b'\x00'
    assert record['AMOUNT'] == b'    0.00'


def test_truncated_nullflags_raises_for_unknown_bits(truncated_path):
    with pytest.raises(NullFlagsUnavailable) as excinfo:
        list(DBF(str(truncated_path)))

    error = excinfo.value
    assert isinstance(error, ValueError)
    assert error.field_name == 'F9'
    assert error.recno == 1
    assert error.bit == 9


def test_truncated_nullflags_still_allows_provable_columns(truncated_path):
    # F0's null bit is in the available byte, F8 contains data and is
    # therefore provably not null even though its bit is lost.
    table = DBF(str(truncated_path), include_fields=['F0', 'F8'])

    assert list(table) == [{'F0': None, 'F8': 1}]


def test_truncated_nullflags_projection_of_unknown_column(truncated_path):
    table = DBF(str(truncated_path), include_fields=['F9'])

    with pytest.raises(NullFlagsUnavailable):
        list(table)


def test_non_nullable_varchar_still_occupies_a_bit(tmp_path):
    # A varchar field gets its extra flag bit even when it is not
    # nullable, so the following field's null flag is bit 1.
    fields = [
        ('TAG', 'V', 6, 0, 0),
        ('AMOUNT', 'N', 8, 2, NULLABLE),
        ('_NullFlags', '0', 1, 0, SYSTEM),
    ]
    records = [
        (b' ', b'ab    ' + b'    1.50' + b'\x00'),
        (b' ', b'      ' + b'        ' + b'\x02'),  # AMOUNT is NULL
    ]
    path = tmp_path / 'varlen.dbf'
    write_dbf(path, fields, records)

    table = DBF(str(path))
    assert table.null_bit_positions == {'TAG': None, 'AMOUNT': 1}
    assert [record['AMOUNT'] for record in table] == [1.5, None]


def test_table_without_nullflags(tmp_path):
    fields = [('NAME', 'C', 8, 0, 0)]
    records = [(b' ', b'Alice   ')]
    path = tmp_path / 'plain.dbf'
    write_dbf(path, fields, records)

    table = DBF(str(path))
    assert table.null_bit_positions == {}
    assert list(table.iter_with_meta()) == [
        (1, {'NAME': 'Alice'}, {}),
    ]
