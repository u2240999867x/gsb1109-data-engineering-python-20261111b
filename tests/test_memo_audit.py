"""
Tests for memo auditing: per-value limits, the distinct-range read
budget, shared memo blocks, declared range conflicts and quarantine.

The DBF/FPT/DBT files used here are synthesized from the documented
binary layout so that memo block placement and lengths are fully under
test control.
"""
import struct

import pytest

from dbfread import DBF, QuarantinedMemo, RangeConflict
from dbfread.memo_audit import (
    ReadBudget,
    BUDGET_EXHAUSTED,
    MEMO_INDEX_INVALID,
    MEMO_TRUNCATED,
    VALUE_EXCEEDS_LIMIT,
)


# ---------------------------------------------------------------------
# File builders
# ---------------------------------------------------------------------

def write_dbf(path, version, fields, records, memo_field_width=None):
    """Write a minimal DBF.

    fields is a list of (name, type, length). Memo ('M') fields store a
    4 byte little endian block index unless memo_field_width requests
    the 10 byte ASCII layout used by dBase.
    """
    record_length = 1 + sum(length for _, _, length in fields)
    header_length = 32 + 32 * len(fields) + 1
    header = bytearray(32)
    header[0] = version
    header[1] = 126 - 100  # year 2026
    header[2] = 11
    header[3] = 13
    struct.pack_into('<I', header, 4, len(records))
    struct.pack_into('<H', header, 8, header_length)
    struct.pack_into('<H', header, 10, record_length)

    descriptors = bytearray()
    for name, field_type, length in fields:
        descriptor = bytearray(32)
        descriptor[0:len(name)] = name.encode('ascii')
        descriptor[11] = ord(field_type)
        descriptor[16] = length
        descriptors += descriptor

    body = bytearray()
    for record in records:
        body += b' '
        for (name, field_type, length), value in zip(fields, record):
            if field_type == 'M':
                if memo_field_width == 10:
                    body += str(value).encode('ascii').ljust(10, b' ')
                else:
                    body += struct.pack('<I', value)
            else:
                body += value.encode('ascii').ljust(length, b' ')
    body += b'\x1a'

    with open(str(path), 'wb') as handle:
        handle.write(bytes(header))
        handle.write(bytes(descriptors))
        handle.write(b'\x0d')
        handle.write(bytes(body))


def write_fpt(path, blocks, blocksize=512):
    """blocks: {index: (memo_type, payload)}."""
    highest = max(blocks)
    size = (highest + 1) * blocksize
    content = bytearray(size)
    struct.pack_into('>I', content, 0, highest + 1)
    struct.pack_into('>H', content, 6, blocksize)
    for index, (memo_type, payload) in blocks.items():
        offset = index * blocksize
        struct.pack_into('>I', content, offset, memo_type)
        struct.pack_into('>I', content, offset + 4, len(payload))
        content[offset + 8:offset + 8 + len(payload)] = payload
    with open(str(path), 'wb') as handle:
        handle.write(bytes(content))


def write_db3_dbt(path, blocks):
    """blocks: {index: payload}. Payload is terminated with 0x1a."""
    end = max(index * 512 + len(payload) + 1
              for index, payload in blocks.items())
    size = ((end + 511) // 512) * 512
    content = bytearray(size)
    for index, payload in blocks.items():
        offset = index * 512
        content[offset:offset + len(payload)] = payload
        content[offset + len(payload)] = 0x1a
    with open(str(path), 'wb') as handle:
        handle.write(bytes(content))


def write_db4_dbt(path, blocks):
    """blocks: {index: payload} using the dBase IV block header."""
    highest = max(blocks)
    content = bytearray((highest + 1) * 512)
    for index, payload in blocks.items():
        offset = index * 512
        struct.pack_into('<I', content, offset, 0x0808ffff)
        struct.pack_into('<I', content, offset + 4, len(payload))
        start = offset + 8
        content[start:start + len(payload)] = payload
        content[start + len(payload)] = 0x1f
    with open(str(path), 'wb') as handle:
        handle.write(bytes(content))


@pytest.fixture
def fpt_table(tmp_path):
    dbf = tmp_path / 'shared.fpt_table.dbf'
    fpt = tmp_path / 'shared.fpt_table.fpt'
    # Both records point at block 1; block 2 is only referenced later.
    write_dbf(str(dbf), 0x30,
              [('NAME', 'C', 10), ('MEMO', 'M', 4)],
              [('alpha', 1), ('beta', 1), ('gamma', 2)])
    write_fpt(str(fpt), {
        1: (0x1, b'shared memo'),
        2: (0x1, b'second memo'),
    })
    return dbf


# ---------------------------------------------------------------------
# ReadBudget bookkeeping
# ---------------------------------------------------------------------

def test_budget_counts_overlapping_ranges_once():
    budget = ReadBudget(max_bytes=100)
    assert budget.reserve(10, 30)
    assert budget.reserve(20, 40)
    assert budget.used == 30
    assert budget.ranges == [(10, 40)]
    assert budget.remaining == 70


def test_budget_rejects_range_that_does_not_fit_atomically():
    budget = ReadBudget(max_bytes=10)
    assert budget.reserve(0, 4)
    assert not budget.reserve(100, 110)
    # The failed reservation changes nothing.
    assert budget.used == 4
    assert budget.ranges == [(0, 4)]


def test_unlimited_budget_never_rejects():
    budget = ReadBudget()
    assert budget.remaining is None
    assert not budget.exhausted
    assert budget.reserve(0, 5000)


def test_declared_range_conflict_names_all_references():
    budget = ReadBudget()
    budget.declare(512, 1032, ref=(0, 'MEMO'))
    conflict = budget.declare(1024, 1100, ref=(1, 'MEMO'))

    assert isinstance(conflict, RangeConflict)
    assert conflict.first_range == (512, 1032)
    assert conflict.second_range == (1024, 1100)
    assert conflict.refs == [(0, 'MEMO'), (1, 'MEMO')]
    assert budget.conflicts == [conflict]


def test_identical_declared_ranges_are_not_conflicts():
    budget = ReadBudget()
    budget.declare(512, 600, ref=(0, 'MEMO'))
    assert budget.declare(512, 600, ref=(2, 'MEMO')) is None
    assert budget.conflicts == []


# ---------------------------------------------------------------------
# FPT (known length) behaviour
# ---------------------------------------------------------------------

def test_shared_memo_block_is_read_and_billed_once(fpt_table):
    table = DBF(str(fpt_table), memo_budget=4096)
    records = list(table)

    assert records[0]['MEMO'] == 'shared memo'
    assert records[1]['MEMO'] == 'shared memo'
    assert records[2]['MEMO'] == 'second memo'

    budget = table.memo_audit
    assert sorted(budget.cache) == [1, 2]
    # Block 1's extent appears once even though two records reference it.
    assert budget.used == sum(end - start for start, end in budget.ranges)
    assert any(start == 512 and end == 512 + 8 + len(b'shared memo')
               for start, end in budget.ranges)


def test_two_fields_in_one_record_sharing_a_block_bill_once(tmp_path):
    dbf = tmp_path / 'twofields.dbf'
    fpt = tmp_path / 'twofields.fpt'
    # Both memo columns of the single record point at block 1.
    write_dbf(str(dbf), 0x30,
              [('NOTE_A', 'M', 4), ('NOTE_B', 'M', 4)],
              [(1, 1)])
    write_fpt(str(fpt), {1: (0x1, b'twice used')})

    table = DBF(str(dbf), memo_budget=4096)
    record = next(iter(table))
    assert record['NOTE_A'] == 'twice used'
    assert record['NOTE_B'] == 'twice used'

    budget = table.memo_audit
    extent_size = 8 + len(b'twice used')
    assert budget.used == extent_size
    assert budget.ranges == [(512, 512 + extent_size)]
    assert sorted(budget.cache) == [1]


def test_cached_shared_memo_survives_budget_exhaustion(fpt_table):
    # Only block 1 fits; block 2 must be quarantined.
    table = DBF(str(fpt_table), memo_budget=32)
    records = list(table)
    budget = table.memo_audit

    assert records[0]['MEMO'] == 'shared memo'
    assert records[1]['MEMO'] == 'shared memo'
    assert isinstance(records[2]['MEMO'], QuarantinedMemo)
    assert records[2]['MEMO'].reason == BUDGET_EXHAUSTED
    assert records[2]['MEMO'].record == 2
    assert records[2]['MEMO'].field == 'MEMO'

    # A second pass still serves the fully cached shared memo while the
    # unexplored block remains unknown.
    second_pass = list(table)
    assert second_pass[0]['MEMO'] == 'shared memo'
    assert second_pass[1]['MEMO'] == 'shared memo'
    assert isinstance(second_pass[2]['MEMO'], QuarantinedMemo)
    assert budget.used <= 32


def test_per_value_limit_quarantines_without_reading_payload(tmp_path):
    dbf = tmp_path / 'big.dbf'
    fpt = tmp_path / 'big.fpt'
    write_dbf(str(dbf), 0x30, [('MEMO', 'M', 4)], [(1,)])
    write_fpt(str(fpt), {1: (0x1, b'x' * 100)})

    table = DBF(str(dbf), max_memo_size=50)
    record = next(iter(table))
    memo = record['MEMO']

    assert isinstance(memo, QuarantinedMemo)
    assert memo.reason == VALUE_EXCEEDS_LIMIT
    assert memo.record == 0 and memo.field == 'MEMO'
    # Only the 8 byte block header was read; no payload was pulled in.
    assert table.memo_audit.ranges == [(512, 520)]


def test_truncated_fpt_block_is_quarantined(tmp_path):
    dbf = tmp_path / 'trunc.dbf'
    fpt = tmp_path / 'trunc.fpt'
    write_dbf(str(dbf), 0x30, [('MEMO', 'M', 4)], [(1,)])
    # Header claims 100 bytes but the file ends 50 bytes into the payload.
    write_fpt(str(fpt), {1: (0x1, b'')})
    with open(str(fpt), 'r+b') as handle:
        handle.seek(512 + 4)
        handle.write(struct.pack('>I', 100))
        handle.truncate(520 + 50)

    table = DBF(str(dbf), memo_budget=4096)
    memo = next(iter(table))['MEMO']
    assert isinstance(memo, QuarantinedMemo)
    assert memo.reason == MEMO_TRUNCATED


def test_overlapping_fpt_blocks_record_conflict_with_records(tmp_path):
    dbf = tmp_path / 'overlap.dbf'
    fpt = tmp_path / 'overlap.fpt'
    write_dbf(str(dbf), 0x30, [('MEMO', 'M', 4)], [(1,), (2,)])
    # Block 1 declares an extent that runs into block 2's territory.
    write_fpt(str(fpt), {
        1: (0x1, b'y' * 520),
        2: (0x1, b'z' * 10),
    })

    table = DBF(str(dbf), memo_budget=8192, max_memo_size=2048)
    list(table)
    conflicts = table.memo_audit.conflicts
    assert len(conflicts) == 1
    refs = conflicts[0].refs
    assert (0, 'MEMO') in refs and (1, 'MEMO') in refs


# ---------------------------------------------------------------------
# DBT behaviour
# ---------------------------------------------------------------------

def test_db3_dbt_finds_end_marker_under_budget(tmp_path):
    dbf = tmp_path / 'd3.dbf'
    dbt = tmp_path / 'd3.dbt'
    write_dbf(str(dbf), 0x83, [('NAME', 'C', 10), ('MEMO', 'M', 4)],
              [('a', 1), ('b', 2)])
    write_db3_dbt(str(dbt), {1: b'first note', 2: b'second note'})

    table = DBF(str(dbf), memo_budget=4096)
    records = list(table)
    assert records[0]['MEMO'] == 'first note'
    assert records[1]['MEMO'] == 'second note'


def test_db3_dbt_probing_stops_when_budget_runs_out(tmp_path):
    dbf = tmp_path / 'd3budget.dbf'
    dbt = tmp_path / 'd3budget.dbt'
    write_dbf(str(dbf), 0x83, [('MEMO', 'M', 4)], [(1,)])
    # The marker only appears in the second 512 byte chunk.
    payload = b'q' * 600
    write_db3_dbt(str(dbt), {1: payload})

    table = DBF(str(dbf), memo_budget=512)
    memo = next(iter(table))['MEMO']
    assert isinstance(memo, QuarantinedMemo)
    assert memo.reason == BUDGET_EXHAUSTED
    assert memo.range == (512, None)
    # Reading never crossed the funded first chunk.
    assert table.memo_audit.ranges == [(512, 1024)]


def test_db3_dbt_per_value_limit_during_probing(tmp_path):
    dbf = tmp_path / 'd3big.dbf'
    dbt = tmp_path / 'd3big.dbt'
    write_dbf(str(dbf), 0x83, [('MEMO', 'M', 4)], [(1,)])
    write_db3_dbt(str(dbt), {1: b'q' * 600})

    table = DBF(str(dbf), memo_budget=4096, max_memo_size=512)
    memo = next(iter(table))['MEMO']
    assert isinstance(memo, QuarantinedMemo)
    assert memo.reason == VALUE_EXCEEDS_LIMIT
    # First chunk was read, the second one was never requested.
    assert table.memo_audit.ranges == [(512, 1024)]


def test_db3_dbt_eof_terminates_last_memo(tmp_path):
    dbf = tmp_path / 'd3eof.dbf'
    dbt = tmp_path / 'd3eof.dbt'
    write_dbf(str(dbf), 0x83, [('MEMO', 'M', 4)], [(1,)])
    # The file ends right after the payload, with no end marker.
    content = bytearray(512) + bytearray(b'end of file memo')
    with open(str(dbt), 'wb') as handle:
        handle.write(bytes(content))

    table = DBF(str(dbf), memo_budget=4096)
    assert next(iter(table))['MEMO'] == 'end of file memo'


def test_db4_dbt_uses_known_length_and_splits_marker(tmp_path):
    dbf = tmp_path / 'd4.dbf'
    dbt = tmp_path / 'd4.dbt'
    write_dbf(str(dbf), 0x8b, [('MEMO', 'M', 4)], [(1,)])
    write_db4_dbt(str(dbt), {1: b'fourth'})

    table = DBF(str(dbf), memo_budget=4096)
    assert next(iter(table))['MEMO'] == 'fourth'

    limited = DBF(str(dbf), max_memo_size=3)
    memo = next(iter(limited))['MEMO']
    assert isinstance(memo, QuarantinedMemo)
    assert memo.reason == VALUE_EXCEEDS_LIMIT
    # Known length means the payload itself is not read.
    assert limited.memo_audit.ranges == [(512, 520)]


# ---------------------------------------------------------------------
# Invalid memo index
# ---------------------------------------------------------------------

def test_invalid_memo_index_is_quarantined_when_auditing(tmp_path):
    dbf = tmp_path / 'badindex.dbf'
    dbt = tmp_path / 'badindex.dbt'
    write_dbf(str(dbf), 0x83, [('MEMO', 'M', 10)], [('XXXXXXXX',)],
              memo_field_width=10)
    write_db3_dbt(str(dbt), {1: b'unused'})

    table = DBF(str(dbf), memo_budget=4096)
    memo = next(iter(table))['MEMO']
    assert isinstance(memo, QuarantinedMemo)
    assert memo.reason == MEMO_INDEX_INVALID
    assert memo.record == 0 and memo.field == 'MEMO'


def test_invalid_memo_index_still_raises_without_audit(tmp_path):
    dbf = tmp_path / 'badindex_plain.dbf'
    dbt = tmp_path / 'badindex_plain.dbt'
    write_dbf(str(dbf), 0x83, [('MEMO', 'M', 10)], [('XXXXXXXX',)],
              memo_field_width=10)
    write_db3_dbt(str(dbt), {1: b'unused'})

    with pytest.raises(ValueError):
        list(DBF(str(dbf)))


def test_no_limits_preserves_plain_memo_values(fpt_table):
    table = DBF(str(fpt_table))
    assert table.memo_audit is None
    records = list(table)
    assert [record['MEMO'] for record in records] == [
        'shared memo', 'shared memo', 'second memo']
    assert all(not isinstance(record['MEMO'], QuarantinedMemo)
               for record in records)
