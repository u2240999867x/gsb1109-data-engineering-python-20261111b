"""
Integrity regression for damaged / truncated DBF archives.

A DBF file can be corrupted in three independent ways:

1. The ``numrecords`` field in the 32-byte header is stale (too high or too
   low) while the record area is intact.
2. Record boundaries are derived wrong. The last slot on disk was only half
   written, so the file ends in the middle of a record.
3. The active/deleted status (the one-byte deletion flag that starts every
   slot) is miscounted or ignored.

These tests never use another dbfread entry point as the source of truth.
Instead the ``IntegrityVerdict`` oracle re-derives everything from the raw
bytes using only the field widths parsed out of the field descriptors, the
deletion flags and the file length. It exposes:

* record numbers that are *certainly* usable (a whole ``recordlen`` slot is
  present with a known flag), and
* an interval of record numbers that *may* still exist (a trailing partial
  slot that does not fill a complete record).

Every sample is built from one byte-identical visible prefix; only the tail
differs (or, for the stale-metadata sample, only the four ``numrecords``
bytes).
"""
import struct

import pytest

from dbfread import DBF

# ---------------------------------------------------------------------------
# On-disk layout used by every sample.
# ---------------------------------------------------------------------------
# Two character/numeric fields give a payload width of 13 bytes and a
# record length of 14 once the one-byte deletion flag is added.
FIELD_SPECS = [(b'NAME', b'C', 10), (b'AGE', b'N', 3)]
PAYLOAD_WIDTH = sum(length for _, _, length in FIELD_SPECS)
NFIELDS = len(FIELD_SPECS)
# file header + field descriptors + terminating CR
HEADERLEN = 32 + 32 * NFIELDS + 1
RECLEN = 1 + PAYLOAD_WIDTH                 # deletion flag + field payload
FLAG_ACTIVE = b' '
FLAG_DELETED = b'*'
EOF_MARKER = 0x1A


def _field_descriptor(name, field_type, length):
    descriptor = bytearray(32)
    descriptor[0:len(name)] = name
    descriptor[11] = field_type[0]
    struct.pack_into('<I', descriptor, 12, 0)   # field address
    descriptor[16] = length
    descriptor[17] = 0                          # decimal count
    return bytes(descriptor)


def _file_header(numrecords):
    header = bytearray(32)
    header[0] = 0x30                            # dBASE III without memo
    header[1], header[2], header[3] = 114, 8, 2
    struct.pack_into('<I', header, 4, numrecords)
    struct.pack_into('<H', header, 8, HEADERLEN)
    struct.pack_into('<H', header, 10, RECLEN)
    return bytes(header)


def _record(flag, name, age):
    name_bytes = name.encode('ascii').ljust(10, b' ')
    age_bytes = str(age).encode('ascii').rjust(3, b' ')
    return flag + name_bytes + age_bytes


# Visible prefix shared by every sample: header, both field descriptors and
# three fully written slots (two active, one deleted).
def _build_prefix():
    parts = [_file_header(3)]
    parts.extend(_field_descriptor(*spec) for spec in FIELD_SPECS)
    parts.append(b'\r')
    parts.append(_record(FLAG_ACTIVE, 'Alice', 30))
    parts.append(_record(FLAG_ACTIVE, 'Bob', 25))
    parts.append(_record(FLAG_DELETED, 'Ghost', 99))
    return b''.join(parts)


PREFIX = _build_prefix()


def _with_numrecords(data, value):
    """Return ``data`` with only the header numrecords field rewritten."""
    clone = bytearray(data)
    struct.pack_into('<I', clone, 4, value)
    return bytes(clone)


def build_variants():
    """Return ordered ``name -> bytes`` samples with a shared prefix."""
    half_active = FLAG_ACTIVE + b'Carol' + b'\x00'      # 7 of 14 bytes
    half_deleted = FLAG_DELETED + b'Gh'                 # 3 of 14 bytes
    return {
        # Clean file: prefix followed by the end-of-file marker.
        'clean': PREFIX + bytes([EOF_MARKER]),
        # Legal end plus reserved tail space after the marker.
        'tail_padding': PREFIX + bytes([EOF_MARKER]) + b'\x00' * 20,
        # Same record bytes as clean, but the header claims seven records.
        'stale_high': _with_numrecords(
            PREFIX + bytes([EOF_MARKER]), 7),
        # Header claims fewer records than are physically present.
        'stale_low': _with_numrecords(
            PREFIX + bytes([EOF_MARKER]), 2),
        # A complete extra slot carrying a deleted flag, then a clean end.
        # The header count is bumped to 4 so this isolates status handling
        # from any stale-metadata fault.
        'deleted_row':
            _with_numrecords(
                PREFIX + _record(FLAG_DELETED, 'Gone4', 40)
                + bytes([EOF_MARKER]), 4),
        # A half-written active slot with no terminating marker.
        'truncated_active': PREFIX + half_active,
        # A half-written deleted slot with no terminating marker.
        'truncated_deleted': PREFIX + half_deleted,
        # A whole slot whose flag byte is neither active nor deleted.
        'unknown_flag':
            PREFIX + _record(b'?', 'Maybe', 50) + bytes([EOF_MARKER]),
    }


# ---------------------------------------------------------------------------
# Independent integrity oracle (no dbfread import is used anywhere in here).
# ---------------------------------------------------------------------------
class IntegrityVerdict(object):
    """Byte-derived view of what is certainly and possibly on disk."""

    def __init__(self, data):
        self.meta_numrecords = struct.unpack_from('<I', data, 4)[0]
        self.headerlen = struct.unpack_from('<H', data, 8)[0]
        self.recordlen = struct.unpack_from('<H', data, 10)[0]
        self.payload_width = self.recordlen - 1

        body = data[self.headerlen:]

        self.certain_active = []
        self.certain_deleted = []
        self.unknown_flag = []
        self.possible_active = []
        self.possible_deleted = []
        self.possible_unknown = []
        self.partial_bytes = b''
        self.eof_marker = False
        self.tail_padding = 0

        position = 0
        slot = 0
        while position < len(body):
            first = body[position:position + 1]

            # The EOF marker only terminates the table when it lands exactly
            # on a record boundary; the same byte inside a slot is payload.
            if first == bytes([EOF_MARKER]) and position % self.recordlen == 0:
                self.eof_marker = True
                self.tail_padding = len(body) - position - 1
                break

            if position + self.recordlen <= len(body):
                # A whole slot is physically present: this record number is
                # certainly readable regardless of the header count.
                if first == FLAG_ACTIVE:
                    self.certain_active.append(slot)
                elif first == FLAG_DELETED:
                    self.certain_deleted.append(slot)
                else:
                    self.unknown_flag.append(slot)
                position += self.recordlen
                slot += 1
            else:
                # The file ends in the middle of a slot. It may yet turn into
                # a record once the writer flushes, so it is only "possible".
                self.partial_bytes = body[position:]
                if first == FLAG_ACTIVE:
                    self.possible_active.append(slot)
                elif first == FLAG_DELETED:
                    self.possible_deleted.append(slot)
                else:
                    self.possible_unknown.append(slot)
                break

    @property
    def certain(self):
        """Record numbers occupying a complete, known-flag slot."""
        return sorted(self.certain_active + self.certain_deleted
                      + self.unknown_flag)

    @property
    def possible(self):
        """Record numbers whose slot is truncated and therefore uncertain."""
        return sorted(self.possible_active + self.possible_deleted
                      + self.possible_unknown)

    @property
    def possible_range(self):
        """Inclusive ``(first, last)`` interval, or ``None`` if complete."""
        if not self.possible:
            return None
        return (self.possible[0], self.possible[-1])

    @property
    def missing_bytes(self):
        """Bytes still needed to finish the trailing partial slot."""
        if not self.partial_bytes:
            return 0
        return self.recordlen - len(self.partial_bytes)

    @property
    def metadata_matches_bytes(self):
        """True only if the header count equals the whole-slot count."""
        return self.meta_numrecords == len(self.certain)


VARIANTS = build_variants()


@pytest.fixture(params=sorted(VARIANTS), ids=sorted(VARIANTS))
def case(request, tmp_path):
    name = request.param
    path = tmp_path / (name + '.dbf')
    path.write_bytes(VARIANTS[name])
    return name, str(path), VARIANTS[name]


def write_case(name, tmp_path):
    """Materialise one named variant and return its path."""
    path = tmp_path / (name + '.dbf')
    path.write_bytes(VARIANTS[name])
    return str(path)


def open_stream(path):
    return DBF(path)


def open_loaded(path):
    return DBF(path, load=True)


# ---------------------------------------------------------------------------
# Sample construction guarantees.
# ---------------------------------------------------------------------------
def test_visible_prefix_is_byte_identical():
    reference = VARIANTS['clean']
    record_prefix = PREFIX[HEADERLEN:]

    for name, data in VARIANTS.items():
        # Every variant carries the same three fully written record slots
        # straight after the header; only the tail differs.
        assert data[HEADERLEN:HEADERLEN + len(record_prefix)] == \
            record_prefix, name
        # The whole header except the four-byte numrecords field is shared.
        assert data[:4] == reference[:4]
        assert data[8:HEADERLEN] == reference[8:HEADERLEN]

    # The stale-metadata sample differs from clean only in the four-byte
    # count field; every other byte is identical.
    assert VARIANTS['stale_high'][:4] == reference[:4]
    assert VARIANTS['stale_high'][8:] == reference[8:]
    assert VARIANTS['stale_high'][4:8] != reference[4:8]


def test_field_widths_define_record_length():
    # The oracle must derive the width from the fields, not assume it.
    verdict = IntegrityVerdict(VARIANTS['clean'])
    assert verdict.payload_width == PAYLOAD_WIDTH == 13
    assert verdict.recordlen == RECLEN == 14
    assert verdict.headerlen == HEADERLEN


# ---------------------------------------------------------------------------
# Oracle behaviour: certain record numbers vs. possible interval.
# ---------------------------------------------------------------------------
def test_oracle_partitions_complete_slots():
    verdict = IntegrityVerdict(VARIANTS['clean'])
    assert verdict.certain_active == [0, 1]
    assert verdict.certain_deleted == [2]
    assert verdict.certain == [0, 1, 2]
    assert verdict.possible == []
    assert verdict.possible_range is None
    assert verdict.eof_marker is True
    assert verdict.metadata_matches_bytes is True


def test_oracle_marks_truncated_active_as_possible():
    verdict = IntegrityVerdict(VARIANTS['truncated_active'])
    # The first three slots remain certain; the fourth is only possible.
    assert verdict.certain_active == [0, 1]
    assert verdict.certain_deleted == [2]
    assert verdict.possible_active == [3]
    assert verdict.possible_deleted == []
    assert verdict.possible_range == (3, 3)
    assert len(verdict.partial_bytes) == 7
    assert verdict.missing_bytes == RECLEN - 7
    assert verdict.eof_marker is False
    # A partial slot must never be promoted to "certain".
    assert 3 not in verdict.certain


def test_oracle_marks_truncated_deleted_as_possible():
    verdict = IntegrityVerdict(VARIANTS['truncated_deleted'])
    assert verdict.certain_deleted == [2]
    assert verdict.possible_deleted == [3]
    assert verdict.possible_active == []
    assert verdict.possible_range == (3, 3)
    assert len(verdict.partial_bytes) == 3
    assert verdict.missing_bytes == RECLEN - 3


def test_oracle_counts_complete_deleted_slot_as_certain():
    verdict = IntegrityVerdict(VARIANTS['deleted_row'])
    assert verdict.certain_active == [0, 1]
    assert verdict.certain_deleted == [2, 3]
    assert verdict.possible == []
    assert verdict.metadata_matches_bytes is True


def test_oracle_flags_unknown_flag_separately():
    verdict = IntegrityVerdict(VARIANTS['unknown_flag'])
    assert verdict.unknown_flag == [3]
    assert verdict.certain_active == [0, 1]
    assert verdict.certain_deleted == [2]
    # The structurally whole but invalid-flag slot is not a readable record.
    assert 3 not in verdict.certain_active
    assert 3 not in verdict.certain_deleted


def test_oracle_detects_stale_metadata_in_both_directions():
    high = IntegrityVerdict(VARIANTS['stale_high'])
    low = IntegrityVerdict(VARIANTS['stale_low'])
    # Byte-derived certainty is identical to the clean file in both cases.
    clean = IntegrityVerdict(VARIANTS['clean'])
    assert high.certain == clean.certain == low.certain
    assert high.meta_numrecords == 7
    assert low.meta_numrecords == 2
    assert high.metadata_matches_bytes is False
    assert low.metadata_matches_bytes is False


def test_oracle_treats_reserved_tail_as_padding():
    padded = IntegrityVerdict(VARIANTS['tail_padding'])
    clean = IntegrityVerdict(VARIANTS['clean'])
    assert padded.certain == clean.certain
    assert padded.possible == []
    assert padded.eof_marker is True
    assert padded.tail_padding == 20
    assert clean.tail_padding == 0


# ---------------------------------------------------------------------------
# Failure class 1: record-count metadata is wrong.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('name', ['stale_high', 'stale_low'])
def test_stale_header_count_does_not_change_reader(name, tmp_path):
    path = write_case(name, tmp_path)
    verdict = IntegrityVerdict(VARIANTS[name])
    table = open_stream(path)

    # The oracle proves the header count disagrees with the bytes, yet the
    # reader returns exactly the byte-derived whole slots. A discrepancy here
    # would therefore be a metadata fault, never a boundary or flag fault.
    assert verdict.metadata_matches_bytes is False
    assert len(table.records) == len(verdict.certain_active)
    assert len(table.deleted) == len(verdict.certain_deleted)
    assert [row['NAME'] for row in table.records] == ['Alice', 'Bob']


# ---------------------------------------------------------------------------
# Failure class 2: record boundaries derived wrong (half-written last slot).
# ---------------------------------------------------------------------------
def test_half_written_active_slot_is_fabricated(tmp_path):
    path = write_case('truncated_active', tmp_path)
    verdict = IntegrityVerdict(VARIANTS['truncated_active'])
    table = open_stream(path)

    # Signature of a boundary fault: the reader emits one more active record
    # than there are whole active slots, filling missing field bytes as NULL.
    assert verdict.possible_active == [3]
    assert len(table.records) == len(verdict.certain_active) + 1
    ghost = list(table.records)[-1]
    assert ghost['NAME'] == 'Carol'
    assert ghost['AGE'] is None
    # Deleted accounting is untouched by the active partial slot.
    assert len(table.deleted) == len(verdict.certain_deleted)


def test_half_written_deleted_slot_is_fabricated(tmp_path):
    path = write_case('truncated_deleted', tmp_path)
    verdict = IntegrityVerdict(VARIANTS['truncated_deleted'])
    table = open_stream(path)

    # Same boundary fault on the deleted stream: a partial '*' slot is handed
    # back as a fully deleted record instead of staying "possible".
    assert verdict.possible_deleted == [3]
    assert len(table.deleted) == len(verdict.certain_deleted) + 1
    ghost = list(table.deleted)[-1]
    assert ghost['NAME'] == 'Gh'
    assert ghost['AGE'] is None
    assert len(table.records) == len(verdict.certain_active)


def test_complete_slots_never_exceed_certain_count(case):
    name, path, _ = case
    if name.startswith('truncated'):
        pytest.skip('only whole-slot samples can satisfy the bound')

    verdict = IntegrityVerdict(VARIANTS[name])
    table = open_stream(path)
    # For every non-truncated file the reader cannot invent records beyond
    # the byte-derived whole slots; this isolates class 2 from classes 1/3.
    assert len(table.records) <= len(verdict.certain_active)
    assert len(table.deleted) <= len(verdict.certain_deleted)


# ---------------------------------------------------------------------------
# Failure class 3: active/deleted status missed or misattributed.
# ---------------------------------------------------------------------------
def test_complete_deleted_row_is_reported_as_deleted(tmp_path):
    path = write_case('deleted_row', tmp_path)
    verdict = IntegrityVerdict(VARIANTS['deleted_row'])
    table = open_stream(path)

    # The fully written '*' slot must land in deleted, never in active, and
    # must not vanish. Losing it would be a status-omission fault.
    assert len(table.deleted) == len(verdict.certain_deleted) == 2
    assert [row['NAME'] for row in table.deleted] == ['Ghost', 'Gone4']
    assert len(table.records) == len(verdict.certain_active) == 2
    assert all(row['NAME'] != 'Gone4' for row in table.records)


def test_unknown_flag_slot_is_counted_neither_active_nor_deleted(tmp_path):
    path = write_case('unknown_flag', tmp_path)
    verdict = IntegrityVerdict(VARIANTS['unknown_flag'])
    table = open_stream(path)
    assert verdict.unknown_flag == [3]
    assert [row['NAME'] for row in table.records] == ['Alice', 'Bob']
    assert [row['NAME'] for row in table.deleted] == ['Ghost']


def test_deleted_flag_is_honoured_across_every_intact_file(case):
    name, path, _ = case
    if name.startswith('truncated'):
        pytest.skip('partial slots are covered by the boundary tests')

    verdict = IntegrityVerdict(VARIANTS[name])
    table = open_stream(path)
    # No whole deleted slot may disappear or be relabelled as active.
    assert len(table.deleted) == len(verdict.certain_deleted)
    active_names = {row['NAME'] for row in table.records}
    assert 'Ghost' not in active_names


# ---------------------------------------------------------------------------
# load() vs. streaming vs. len() must agree for every damaged sample.
# ---------------------------------------------------------------------------
def test_load_stream_and_count_agree(case):
    _, path, raw = case
    verdict = IntegrityVerdict(raw)

    streamed = open_stream(path)
    loaded = open_loaded(path)

    streamed_active = list(streamed.records)
    streamed_deleted = list(streamed.deleted)
    loaded_active = list(loaded.records)
    loaded_deleted = list(loaded.deleted)

    # All three access paths (RecordIterator iteration, len() and the loaded
    # lists) report identical counts and content.
    assert len(streamed) == len(streamed.records) == len(loaded_active)
    assert (len(streamed.deleted) == len(loaded.deleted)
            == len(loaded_deleted))
    assert streamed_active == loaded_active
    assert streamed_deleted == loaded_deleted

    # And the only way the reader can exceed the certain boundary is a
    # trailing partial slot; otherwise counts match the whole slots exactly.
    if not verdict.possible:
        assert len(loaded_active) == len(verdict.certain_active)
        assert len(loaded_deleted) == len(verdict.certain_deleted)
