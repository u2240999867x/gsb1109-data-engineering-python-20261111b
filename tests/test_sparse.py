"""
Tests for sparse read planning and execution.
"""
import pytest

from dbfread import DBF
from dbfread.sparse import (
    ReadCost,
    ReadRange,
    plan_sparse_reads,
    execute_sparse_plan,
    record_offsets,
)


HEADERLEN = 392
RECORDLEN = 29
NUMRECORDS = 3


class StubHeader(object):
    def __init__(self, headerlen=HEADERLEN, recordlen=RECORDLEN,
                 numrecords=NUMRECORDS):
        self.headerlen = headerlen
        self.recordlen = recordlen
        self.numrecords = numrecords


class StubTable(object):
    """Only exposes the header attributes used by the planner."""

    def __init__(self, **kwargs):
        self.header = StubHeader(**kwargs)


@pytest.fixture
def table():
    return DBF('tests/cases/memotest.dbf')


def expected_span(records, headerlen=HEADERLEN, recordlen=RECORDLEN):
    start = headerlen + (records[0] - 1) * recordlen
    end = headerlen + records[-1] * recordlen
    return start, end


def all_partitions(seq):
    if not seq:
        yield []
        return
    for cut in range(1, len(seq) + 1):
        for rest in all_partitions(seq[cut:]):
            yield [seq[:cut]] + rest


def brute_force_best(unique, recordlen, max_length, pricing):
    best = None
    for partition in all_partitions(unique):
        lengths = [(part[-1] - part[0] + 1) * recordlen
                   for part in partition]
        if max_length is not None and any(length > max_length
                                          for length in lengths):
            continue
        cost = sum(pricing.of(length) for length in lengths)
        candidate = (cost, len(partition))
        if best is None or candidate < best:
            best = candidate
    return best


def test_record_offsets(table):
    assert record_offsets(table, 1) == (392, 421)
    assert record_offsets(table, 2) == (421, 450)
    assert record_offsets(table, 3) == (450, 479)


def test_single_record_range(table):
    plan = plan_sparse_reads(table, [2])
    assert plan.ranges == (ReadRange(421, 450, (2,)),)
    assert plan.record_order == (2,)
    assert plan.record_ranges == plan.ranges
    assert plan.cost == 0


def test_ranges_merge_when_startup_dominates(table):
    plan = plan_sparse_reads(table, [1, 3], startup_cost=100,
                             byte_cost=1)
    assert plan.ranges == (ReadRange(392, 479, (1, 2, 3)),)
    # One request of 87 bytes.
    assert plan.cost == pytest.approx(100 + 87)


def test_ranges_split_when_bytes_dominate(table):
    plan = plan_sparse_reads(table, [1, 3], startup_cost=0,
                             byte_cost=1)
    assert plan.ranges == (
        ReadRange(392, 421, (1,)),
        ReadRange(450, 479, (3,)),
    )
    # Two requests of 29 bytes each instead of 87 bytes.
    assert plan.cost == pytest.approx(2 * 29)


def test_request_order_and_duplicates_are_preserved(table):
    plan = plan_sparse_reads(table, [3, 1, 3, 1, 2], startup_cost=100)
    assert plan.record_order == (3, 1, 3, 1, 2)
    # Everything is covered by a single merged range ...
    assert plan.ranges == (ReadRange(392, 479, (1, 2, 3)),)
    # ... and every request points back at that range.
    assert plan.record_ranges == plan.ranges * 5
    assert plan.range_for(2).records == (1, 2, 3)


def test_empty_request():
    plan = plan_sparse_reads(StubTable(), [])
    assert plan.ranges == ()
    assert plan.cost == 0
    assert plan.record_order == ()
    result = execute_sparse_plan(DBF('tests/cases/memotest.dbf'), plan)
    assert result.records == []
    assert result.short_reads == []
    assert result.requests == 0
    assert result.bytes_read == 0


def test_max_length_forces_multiple_reads(table):
    # Two records span 58 bytes so a 40 byte cap needs two reads.
    plan = plan_sparse_reads(table, [1, 2], startup_cost=0,
                             byte_cost=1, max_length=40)
    assert plan.ranges == (
        ReadRange(392, 421, (1,)),
        ReadRange(421, 450, (2,)),
    )
    for read_range in plan.ranges:
        assert read_range.length <= 40


def test_plan_matches_brute_force():
    recordlen = 10
    table = StubTable(headerlen=0, recordlen=recordlen, numrecords=12)
    unique_full = list(range(1, 13))

    for seed, subset in enumerate((
            [1, 12],
            [1, 2, 12],
            [1, 5, 6, 12],
            [2, 4, 7, 8, 11],
            unique_full,
    )):
        for startup, per_byte, max_length in (
                (0, 1, None),
                (5, 1, None),
                (50, 1, None),
                (5, 0, None),
                (3.5, 0.25, None),
                (5, 1, 25),
                (5, 1, 20),
                (0, 1, 10),
        ):
            pricing = ReadCost(startup, per_byte)
            plan = plan_sparse_reads(table, subset, startup, per_byte,
                                     max_length)
            expected_cost, expected_count = brute_force_best(
                subset, recordlen, max_length, pricing)
            assert plan.cost == pytest.approx(expected_cost)
            assert len(plan.ranges) == expected_count
            for read_range in plan.ranges:
                if max_length is not None:
                    assert read_range.length <= max_length


def test_gap_rows_are_listed_but_cost_once():
    table = StubTable(headerlen=100, recordlen=4, numrecords=10)
    plan = plan_sparse_reads(table, [2, 5], startup_cost=20,
                             byte_cost=2)
    # One range covers physical records 2..5, including 3 and 4.
    assert plan.ranges == (ReadRange(104, 120, (2, 3, 4, 5)),)
    assert plan.cost == pytest.approx(20 + 16 * 2)


@pytest.mark.parametrize('kwargs', [
    dict(startup_cost=-1, byte_cost=0),
    dict(startup_cost=0, byte_cost=-1),
])
def test_negative_costs_rejected(table, kwargs):
    with pytest.raises(ValueError):
        plan_sparse_reads(table, [1], **kwargs)


@pytest.mark.parametrize('record_number', [0, -1, 4])
def test_record_number_outside_table_rejected(table, record_number):
    with pytest.raises(ValueError):
        plan_sparse_reads(table, [record_number])


def test_non_integer_record_number_rejected(table):
    with pytest.raises(TypeError):
        plan_sparse_reads(table, [1.5])


def test_max_length_smaller_than_record_rejected(table):
    with pytest.raises(ValueError):
        plan_sparse_reads(table, [1], max_length=RECORDLEN - 1)


def test_non_positive_max_length_rejected(table):
    with pytest.raises(ValueError):
        plan_sparse_reads(table, [1], max_length=0)


def test_end_to_end_sparse_read_matches_full_scan(table):
    full = list(table)
    result = table.read_sparse([3, 1, 3], startup_cost=100)
    # Deleted record 3 is decoded because sparse reads ignore the
    # deletion marker, and request order plus duplicates are kept.
    assert [number for number, _ in result.records] == [3, 1, 3]
    assert result.records[0][1] == list(table.deleted)[0]
    assert result.records[1][1] == full[0]
    assert result.records[2][1] == list(table.deleted)[0]
    assert result.short_reads == []
    assert result.requests == 1
    assert result.bytes_read == 87


def test_raw_sparse_read(table):
    raw_table = DBF('tests/cases/memotest.dbf', raw=True)
    result = raw_table.read_sparse([1])
    _, record = result.records[0]
    # Raw records are the full slot, marker byte included.
    assert isinstance(record, bytes)
    assert len(record) == RECORDLEN


def test_short_read_at_eof(tmp_path, table):
    copy_path = tmp_path / 'truncated.dbf'
    with open(table.filename, 'rb') as source:
        data = source.read()
    # Drop the whole third record and the EOF marker.
    with open(str(copy_path), 'wb') as target:
        target.write(data[:HEADERLEN + 2 * RECORDLEN])

    short_table = DBF(str(copy_path), ignore_missing_memofile=True)
    plan = short_table.plan_sparse_reads([1, 2, 3])
    result = execute_sparse_plan(short_table, plan)

    assert [number for number, _ in result.records] == [1, 2]
    assert len(result.short_reads) == 1
    short = result.short_reads[0]
    assert short.record == 3
    assert short.have_start == HEADERLEN
    assert short.have_end == HEADERLEN + 2 * RECORDLEN
    # The continuation starts exactly where the returned bytes ended.
    assert short.missing == ReadRange(
        HEADERLEN + 2 * RECORDLEN, HEADERLEN + 3 * RECORDLEN, (3,))
    assert result.missing_ranges == (short.missing,)
    assert result.bytes_read == 2 * RECORDLEN

    # Reading the missing range from the intact file supplies the row.
    with open(table.filename, 'rb') as intact:
        intact.seek(short.missing.start)
        tail = intact.read(short.missing.length)
    assert len(tail) == RECORDLEN
    resumed = table.read_sparse([3])
    assert resumed.records[0][1] == list(table.deleted)[0]


class PartialFile(object):
    """File stand-in that only returns the first ``limit`` bytes."""

    def __init__(self, real, limit):
        self._real = real
        self._limit = limit

    def seek(self, offset, whence=0):
        return self._real.seek(offset, whence)

    def read(self, size):
        position = self._real.tell()
        size = min(size, max(0, self._limit - position))
        return self._real.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._real.close()


def test_partial_read_marks_only_undecodable_rows(table, monkeypatch):
    import dbfread.sparse as sparse_module

    real_open = open
    limit = HEADERLEN + RECORDLEN + 11  # full record 1, half of record 2

    def fake_open(path, mode='rb'):
        return PartialFile(real_open(path, mode), limit)

    monkeypatch.setattr(sparse_module, '_open', fake_open)

    plan = table.plan_sparse_reads([1, 2, 3])
    assert plan.ranges == (ReadRange(392, 479, (1, 2, 3)),)
    result = execute_sparse_plan(table, plan)

    # Record 1 is fully present and decoded.
    assert [number for number, _ in result.records] == [1]
    # Records 2 and 3 lack enough bytes and share one continuation.
    assert [short.record for short in result.short_reads] == [2, 3]
    missing = result.short_reads[0].missing
    assert result.short_reads[1].missing is missing
    assert missing.start == limit
    assert missing.end == 479
    assert missing.records == (2, 3)
    assert result.missing_ranges == (missing,)
    assert result.requests == 1
    assert result.bytes_read == limit - HEADERLEN


def test_cost_helpers():
    pricing = ReadCost(7, 0.5)
    assert pricing.of(10) == 12
    read_range = ReadRange(0, 10, (1,))
    assert read_range.length == 10
    assert read_range.cost(pricing) == 12
