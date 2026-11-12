"""Import reconciliation regression tests.

Scenario: one batch of money amounts and timestamps arrives twice.

* The FoxPro side stores binary currency (``Y``) and timestamp (``T``)
  fields, decoded straight from the raw bytes.
* The vendor side writes the same business values as decimal character
  columns (numeric ``N`` text and ISO datetime ``C`` text).

An independent oracle reads every value from the original bytes.  The
dbfread parser output is then checked in three processing stages:

1. parse      -- raw bytes become Python objects,
2. normalize  -- both representations become exact ``Decimal`` /
                 ``datetime`` values and must agree,
3. rollup     -- amounts are summed per calendar day.

The fixture contains paired records for negative amounts, a timestamp on
the smallest time unit that crosses midnight, and null values.  One
record carries a corrupted field, which must be quarantined instead of
touching any total.  Every failure report names the original row, the
offending raw bytes and the first stage where the value was distorted.
"""
import datetime
import struct
from collections import defaultdict, namedtuple
from decimal import Decimal

import pytest

from dbfread import DBF, FieldParser
from dbfread.field_parser import InvalidValue

FOXPRO_VERSION = 0x30
HEADER_FMT = '<BBBBLHHHBBLLLBBH'
FIELD_FMT = '<11scLBBHBBBB7sB'
JULIAN_OFFSET = 1721425
N_LEN = 14
C_TS_LEN = 23

FieldSpec = namedtuple('FieldSpec', 'name type length decimals')

FIELDS = [
    FieldSpec('RID', 'C', 6, 0),
    FieldSpec('AMT_BIN', 'Y', 8, 4),
    FieldSpec('AMT_TXT', 'N', N_LEN, 4),
    FieldSpec('TS_BIN', 'T', 8, 0),
    FieldSpec('TS_TXT', 'C', C_TS_LEN, 0),
]


# ---------------------------------------------------------------------
# Byte-level builders (the "files" delivered by the two systems)
# ---------------------------------------------------------------------

def currency_bytes(amount):
    """Encode a Decimal amount as a Visual FoxPro Y (8-byte int, 4 dp)."""
    return struct.pack('<q', int(amount * 10000))


def numeric_bytes(text):
    """Right-justify a decimal string inside an N field."""
    return text.encode('ascii').rjust(N_LEN, b' ')


def timestamp_bytes(value):
    """Encode a datetime as a Visual FoxPro T (JDN + ms since midnight)."""
    if value is None:
        return b' ' * 8
    jdn = value.toordinal() + JULIAN_OFFSET
    msec = (value.hour * 3600 + value.minute * 60 + value.second) * 1000
    msec += value.microsecond // 1000
    return struct.pack('<LL', jdn, msec)


def char_bytes(text, length):
    return text.encode('ascii').ljust(length, b' ')


def build_dbf(path, raw_rows):
    """Write a minimal Visual FoxPro DBF containing the given raw rows."""
    header_len = 32 + 32 * len(FIELDS) + 1
    record_len = 1 + sum(field.length for field in FIELDS)
    header = struct.pack(
        HEADER_FMT,
        FOXPRO_VERSION,
        126, 11, 13,                # last update: 2026-11-13
        len(raw_rows),
        header_len,
        record_len,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
    )

    chunks = [header]
    for field in FIELDS:
        name = field.name.encode('ascii').ljust(11, b'\0')
        chunks.append(struct.pack(
            FIELD_FMT, name, field.type.encode('ascii'), 0,
            field.length, field.decimals,
            0, 0, 0, 0, 0, b'\0' * 7, 0))
    chunks.append(b'\r')

    for row in raw_rows:
        chunks.append(b' ' + b''.join(row))
    chunks.append(b'\x1a')

    path.write_bytes(b''.join(chunks))


# ---------------------------------------------------------------------
# Independent oracle: exact values computed only from the raw bytes
# ---------------------------------------------------------------------

def oracle_currency(raw):
    if not raw.strip():
        return None
    return Decimal(struct.unpack('<q', raw)[0]) / Decimal(10000)


def oracle_numeric(raw):
    cleaned = raw.strip().strip(b'*\0')
    if not cleaned:
        return None
    try:
        return Decimal(cleaned.decode('ascii').replace(',', '.'))
    except (ValueError, ArithmeticError) as exc:
        raise ValueError('not a decimal: {!r}'.format(raw)) from exc


def oracle_timestamp(raw):
    if not raw.strip():
        return None
    day, msec = struct.unpack('<LL', raw)
    if day == 0:
        return None
    moment = datetime.datetime.fromordinal(day - JULIAN_OFFSET)
    return moment + datetime.timedelta(seconds=msec / 1000)


def oracle_text_timestamp(raw):
    text = raw.rstrip(b'\0 ').decode('ascii')
    if not text:
        return None
    return datetime.datetime.strptime(text, '%Y-%m-%d %H:%M:%S.%f')


# ---------------------------------------------------------------------
# Parsers under test and reconciliation machinery
# ---------------------------------------------------------------------

class TolerantFieldParser(FieldParser):
    """Parser that maps blank/corrupt bytes to None / InvalidValue."""

    def parseY(self, field, data):
        if not data.strip():
            return None
        return super().parseY(field, data)

    def parseN(self, field, data):
        try:
            return super().parseN(field, data)
        except ValueError:
            return InvalidValue(data)


class RoundingFieldParser(TolerantFieldParser):
    """Deliberately buggy parser: currency is rounded to 2 decimals."""

    def parseY(self, field, data):
        if not data.strip():
            return None
        return super().parseY(field, data).quantize(Decimal('0.01'))


class StockCurrencyFieldParser(FieldParser):
    """Stock Y behavior (no blank handling), tolerant N for the bad row."""

    def parseN(self, field, data):
        try:
            return super().parseN(field, data)
        except ValueError:
            return InvalidValue(data)


CorruptRecord = namedtuple(
    'CorruptRecord', 'row rid field raw_hex stage detail')


class ReconciliationFailure(AssertionError):
    """A mismatch annotated with row, raw bytes and distortion stage."""


def fail(row, rid, field, stage, raw, expected, actual):
    raise ReconciliationFailure(
        'row={} rid={!r} field={} stage={} raw={} '
        'expected={!r} actual={!r}'.format(
            row, rid, field, stage,
            raw.hex() if isinstance(raw, (bytes, bytearray)) else raw,
            expected, actual))


def normalize_amount(value):
    if value is None:
        return None
    return Decimal(str(value))


def normalize_time(value):
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        return datetime.datetime.strptime(text, '%Y-%m-%d %H:%M:%S.%f') \
            if text else None
    raise TypeError('unsupported timestamp: {!r}'.format(value))


def _read_raw(path):
    return list(DBF(str(path), raw=True, load=True).records)


def _read_parsed(path, parserclass, raw_records):
    table = DBF(str(path), parserclass=parserclass)
    iterator = iter(table)
    parsed = []
    row = 0
    while True:
        try:
            record = next(iterator)
        except StopIteration:
            return parsed
        except Exception as exc:
            raw = raw_records[row]
            rid = raw['RID'].decode('ascii').strip()
            raise ReconciliationFailure(
                'row={} rid={!r} stage=parse raw={} detail={}'.format(
                    row, rid,
                    ' '.join('{}:{}'.format(name, raw[name].hex())
                             for name in raw),
                    exc)) from exc
        else:
            parsed.append(record)
            row += 1


def reconcile(path, parserclass=TolerantFieldParser):
    """Run the full import reconciliation and return a result report."""
    raw_records = _read_raw(path)
    corruptions = []
    oracle_rows = []

    for row, raw in enumerate(raw_records):
        rid = raw['RID'].decode('ascii').strip()
        oracle = {
            'AMT_BIN': oracle_currency(raw['AMT_BIN']),
            'TS_BIN': oracle_timestamp(raw['TS_BIN']),
            'TS_TXT': oracle_text_timestamp(raw['TS_TXT']),
        }
        try:
            oracle['AMT_TXT'] = oracle_numeric(raw['AMT_TXT'])
        except ValueError as exc:
            corruptions.append(CorruptRecord(
                row, rid, 'AMT_TXT', raw['AMT_TXT'].hex(),
                'parse', str(exc)))
            continue
        oracle_rows.append((row, rid, raw, oracle))

    parsed_records = _read_parsed(path, parserclass, raw_records)

    normalized = []
    for row, rid, raw, oracle in oracle_rows:
        parsed = parsed_records[row]

        for field, expected in oracle.items():
            value = parsed[field]
            if isinstance(value, InvalidValue):
                fail(row, rid, field, 'parse', raw[field], expected, value)

        actual = {
            'AMT_BIN': normalize_amount(parsed['AMT_BIN']),
            'AMT_TXT': normalize_amount(parsed['AMT_TXT']),
            'TS_BIN': normalize_time(parsed['TS_BIN']),
            'TS_TXT': normalize_time(parsed['TS_TXT']),
        }

        for field, expected in oracle.items():
            if actual[field] != expected:
                fail(row, rid, field, 'normalize', raw[field],
                     expected, actual[field])

        if actual['AMT_BIN'] != actual['AMT_TXT']:
            fail(row, rid, 'AMT_TXT', 'normalize', raw['AMT_TXT'],
                 actual['AMT_BIN'], actual['AMT_TXT'])
        if actual['TS_BIN'] != actual['TS_TXT']:
            fail(row, rid, 'TS_TXT', 'normalize', raw['TS_TXT'],
                 actual['TS_BIN'], actual['TS_TXT'])

        normalized.append((rid, raw, actual))

    expected_totals, expected_grand, expected_unbucketed = _rollup(
        (rid, oracle['AMT_BIN'], oracle['TS_BIN'])
        for _, rid, _, oracle in oracle_rows)
    actual_totals, actual_grand, actual_unbucketed = _rollup(
        (rid, values['AMT_BIN'], values['TS_BIN'])
        for rid, _, values in normalized)

    if actual_totals != expected_totals \
            or actual_grand != expected_grand \
            or actual_unbucketed != expected_unbucketed:
        raise ReconciliationFailure(
            'stage=rollup expected_totals={!r} actual_totals={!r} '
            'expected_grand={!r} actual_grand={!r}'.format(
                expected_totals, actual_totals,
                expected_grand, actual_grand))

    return {
        'corruptions': corruptions,
        'totals': actual_totals,
        'grand_total': actual_grand,
        'unbucketed': actual_unbucketed,
        'normalized': normalized,
    }


def _rollup(rows):
    totals = defaultdict(Decimal)
    grand_total = Decimal('0')
    unbucketed = []
    for rid, amount, moment in rows:
        if amount is None:
            continue
        grand_total += amount
        if moment is None:
            unbucketed.append(rid)
        else:
            totals[moment.date()] += amount
    return dict(totals), grand_total, unbucketed


# ---------------------------------------------------------------------
# Fixture: one batch, paired binary and textual representations
# ---------------------------------------------------------------------

CORRUPT_TEXT = b'**CORRUPT**   '
assert len(CORRUPT_TEXT) == N_LEN

EXPECTED_TOTALS = {
    datetime.date(2026, 1, 1): Decimal('-1124.3178'),
    datetime.date(2026, 1, 2): Decimal('20.0000'),
}
EXPECTED_GRAND = Decimal('-1099.3178')


@pytest.fixture(scope='module')
def batch_path(tmp_path_factory):
    timestamp = datetime.datetime
    rows = [
        # ordinary positive pair
        ('POS', Decimal('100.2500'),
         timestamp(2026, 1, 1, 8, 0, 0)),
        # negative amount pair
        ('NEG', Decimal('-1234.5678'),
         timestamp(2026, 1, 1, 10, 0, 0)),
        # cross-midnight pair, millisecond resolution, same wall clock;
        # one more millisecond would push MID1 onto the next day
        ('MID1', Decimal('10.0000'),
         timestamp(2026, 1, 1, 23, 59, 59, 999000)),
        ('MID2', Decimal('20.0000'),
         timestamp(2026, 1, 2, 23, 59, 59, 999000)),
        # null timestamp, amount must still count
        ('TNULL', Decimal('5.0000'), None),
        # null amount, timestamp must still land in its day bucket
        ('ANULL', None,
         timestamp(2026, 1, 1, 0, 0, 1)),
        # corrupted vendor amount: valid binary, unreadable text
        ('BAD', Decimal('999.0000'),
         timestamp(2026, 1, 3, 12, 0, 0)),
    ]

    raw_rows = []
    for rid, amount, moment in rows:
        amount_bin = b' ' * 8 if amount is None \
            else currency_bytes(amount)
        amount_txt = b' ' * N_LEN if amount is None \
            else numeric_bytes(str(amount))
        if rid == 'BAD':
            amount_txt = CORRUPT_TEXT
        time_bin = timestamp_bytes(moment)
        time_txt = char_bytes(
            '' if moment is None
            else moment.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            C_TS_LEN)
        raw_rows.append((
            char_bytes(rid, 6),
            amount_bin, amount_txt,
            time_bin, time_txt,
        ))

    path = tmp_path_factory.mktemp('reconcile') / 'batch.dbf'
    build_dbf(path, raw_rows)
    return path


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_fixture_opens_with_expected_layout(batch_path):
    table = DBF(str(batch_path), raw=True, load=True)
    assert len(table) == 7
    assert [(f.name, f.type, f.length) for f in table.fields] == [
        ('RID', 'C', 6),
        ('AMT_BIN', 'Y', 8),
        ('AMT_TXT', 'N', N_LEN),
        ('TS_BIN', 'T', 8),
        ('TS_TXT', 'C', C_TS_LEN),
    ]

    raw = table.records
    neg = next(row for row in raw
               if row['RID'].strip() == b'NEG')
    assert neg['AMT_BIN'] == struct.pack('<q', -12345678)
    assert neg['AMT_TXT'] == numeric_bytes('-1234.5678')
    assert neg['AMT_BIN'] != neg['AMT_TXT']


def test_different_raw_bytes_same_business_value(batch_path):
    raw = _read_raw(batch_path)
    for row in raw:
        rid = row['RID'].decode().strip()
        assert row['AMT_BIN'] != row['AMT_TXT']
        assert row['TS_BIN'] != row['TS_TXT']

        if rid == 'BAD':
            with pytest.raises(ValueError):
                oracle_numeric(row['AMT_TXT'])
            continue

        assert oracle_currency(row['AMT_BIN']) == \
            oracle_numeric(row['AMT_TXT'])
        assert oracle_timestamp(row['TS_BIN']) == \
            oracle_text_timestamp(row['TS_TXT'])


def test_negative_amount_survives_every_stage(batch_path):
    report = reconcile(batch_path)
    rid, _, values = next(item for item in report['normalized']
                          if item[0] == 'NEG')
    assert values['AMT_BIN'] == Decimal('-1234.5678')
    assert values['AMT_TXT'] == Decimal('-1234.5678')
    assert report['grand_total'] == EXPECTED_GRAND
    assert report['totals'][datetime.date(2026, 1, 1)] == \
        Decimal('-1124.3178')


def test_cross_midnight_same_display_goes_to_different_partitions(
        batch_path):
    report = reconcile(batch_path)
    by_rid = {rid: values for rid, _, values in report['normalized']}

    first = by_rid['MID1']['TS_BIN']
    second = by_rid['MID2']['TS_BIN']
    assert first.strftime('%H:%M:%S') == second.strftime('%H:%M:%S')
    assert first.strftime('%H:%M:%S') == '23:59:59'
    assert first.microsecond == 999000
    assert first.date() != second.date()

    # The smallest T-field unit (one millisecond) crosses the day line.
    one_ms = datetime.timedelta(milliseconds=1)
    assert (first + one_ms).date() == datetime.date(2026, 1, 2)

    assert first.date() == datetime.date(2026, 1, 1)
    assert second.date() == datetime.date(2026, 1, 2)
    assert report['totals'][first.date()] == Decimal('-1124.3178')
    assert report['totals'][second.date()] == Decimal('20.0000')


def test_null_timestamp_keeps_amount_out_of_day_buckets(batch_path):
    report = reconcile(batch_path)
    rid, _, values = next(item for item in report['normalized']
                          if item[0] == 'TNULL')
    assert values['TS_BIN'] is None
    assert values['TS_TXT'] is None
    assert values['AMT_BIN'] == Decimal('5.0000')
    assert 'TNULL' in report['unbucketed']
    assert report['grand_total'] == EXPECTED_GRAND


def test_null_amount_is_never_summed(batch_path):
    report = reconcile(batch_path)
    rid, raw, values = next(item for item in report['normalized']
                            if item[0] == 'ANULL')
    assert values['AMT_BIN'] is None
    assert values['AMT_TXT'] is None
    assert values['TS_BIN'] == datetime.datetime(2026, 1, 1, 0, 0, 1)
    assert report['totals'][datetime.date(2026, 1, 1)] == \
        Decimal('-1124.3178')
    assert raw['AMT_BIN'] == b' ' * 8


def test_stock_parser_distorts_blank_currency_at_normalize_stage(
        batch_path):
    raw = _read_raw(batch_path)
    anull = next(row for row in raw
                 if row['RID'].strip() == b'ANULL')

    # The stock parser happily unpacks eight spaces into a bogus integer:
    # the distortion is silent and only surfaces during normalization.
    parse = make_stock_parser()
    bogus = parse(dummy_field('Y'), anull['AMT_BIN'])
    assert isinstance(bogus, Decimal)
    assert bogus != 0

    with pytest.raises(ReconciliationFailure) as excinfo:
        reconcile(batch_path, parserclass=StockCurrencyFieldParser)
    message = str(excinfo.value)
    assert 'stage=normalize' in message
    assert 'ANULL' in message
    assert 'field=AMT_BIN' in message
    assert anull['AMT_BIN'].hex() in message
    assert 'expected=None' in message


def test_corrupt_field_is_quarantined_and_never_totaled(batch_path):
    report = reconcile(batch_path)
    assert len(report['corruptions']) == 1
    corruption = report['corruptions'][0]

    assert corruption.row == 6
    assert corruption.rid == 'BAD'
    assert corruption.field == 'AMT_TXT'
    assert corruption.stage == 'parse'
    assert corruption.raw_hex == CORRUPT_TEXT.hex()

    parsed = DBF(str(batch_path), parserclass=TolerantFieldParser,
                 load=True).records[6]
    assert isinstance(parsed['AMT_TXT'], InvalidValue)

    assert datetime.date(2026, 1, 3) not in report['totals']
    assert report['totals'] == EXPECTED_TOTALS
    assert report['grand_total'] == EXPECTED_GRAND


def test_hard_parse_failure_is_localized_to_row_and_bytes(batch_path):
    raw = _read_raw(batch_path)
    bad = raw[6]

    with pytest.raises(ReconciliationFailure) as excinfo:
        reconcile(batch_path, parserclass=FieldParser)

    message = str(excinfo.value)
    assert 'stage=parse' in message
    assert 'row=6' in message
    assert "rid='BAD'" in message
    assert bad['AMT_TXT'].hex() in message
    assert 'could not convert string to float' in message


def test_diagnostic_names_row_bytes_and_first_distortion_stage(
        batch_path):
    raw = _read_raw(batch_path)
    neg = next(row for row in raw
               if row['RID'].strip() == b'NEG')

    with pytest.raises(ReconciliationFailure) as excinfo:
        reconcile(batch_path, parserclass=RoundingFieldParser)

    message = str(excinfo.value)
    assert 'stage=normalize' in message
    assert 'stage=rollup' not in message
    assert "rid='NEG'" in message
    assert 'row=1' in message
    assert neg['AMT_BIN'].hex() in message
    assert "expected=Decimal('-1234.5678')" in message
    assert "actual=Decimal('-1234.57')" in message


def test_rollup_matches_independently_computed_totals(batch_path):
    report = reconcile(batch_path)
    assert report['totals'] == EXPECTED_TOTALS
    assert report['grand_total'] == EXPECTED_GRAND
    assert report['unbucketed'] == ['TNULL']


# ---------------------------------------------------------------------
# Minimal stand-ins for direct parser calls
# ---------------------------------------------------------------------

class _DummyTable(object):
    class header(object):
        dbversion = FOXPRO_VERSION
    encoding = 'ascii'
    char_decode_errors = 'strict'


def dummy_field(field_type):
    return type('Field', (), {'type': field_type})()


def make_stock_parser():
    parser = FieldParser(_DummyTable())
    return lambda field, data: parser.parse(field, data)
