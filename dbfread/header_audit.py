"""
Header audit for DBF files whose field descriptors admit more than one
layout.

Old batches of DBF files sometimes describe their field layout in two
incompatible ways: the offsets stored in the descriptors and the offsets
implied by accumulating field widths do not agree.  This module never
repairs or rewrites anything.  Instead it:

1. enumerates the layouts allowed by the finite set of format variants
   declared by the caller (reserved byte rules, field boundary rules,
   record width rules),
2. rejects layouts that violate those rules or that fail to reproduce a
   small set of known raw record samples supplied by the caller,
3. reports either the single surviving layout (safe to decode with) or
   the columns and byte ranges that every surviving layout agrees on,
   together with pairs of candidate layouts that read different raw
   values for the columns still in dispute.

The module only opens files in binary read mode.
"""
import struct

HEADER_SIZE = 32
FIELD_SIZE = 32
TERMINATOR = 0x0D
END_OF_FILE = 0x1A
RECORD_MARKERS = (0x20, 0x2A, END_OF_FILE)
DEFAULT_SAMPLE_COUNT = 8

# Same on-disk descriptor as DBFField in dbf.py, duplicated here so the
# auditor does not depend on the reader module.
_FIELD_FORMAT = '<11scLBBHBBBB7sB'
_FIELD_NAMES = ['name', 'type', 'address', 'length', 'decimal_count',
                'reserved1', 'workarea_id', 'reserved2', 'reserved3',
                'set_fields_flag', 'reserved4', 'index_field_flag']
_field_struct = struct.Struct(_FIELD_FORMAT)

# Offsets inside the 32 byte file header.
_H = struct.Struct('<BBBBLHHHBBLLLBBH')

DEFAULT_FIELD_TYPES = frozenset('CNFDLMGPB@+VTIYO0')


class AuditFailure(ValueError):
    """Raised when no declared variant yields a legal layout."""


class AmbiguousHeader(ValueError):
    """Raised when more than one layout survives the audit.

    The report is attached so callers can inspect the consensus columns
    and the disputed candidate layouts without touching the file again.
    """
    def __init__(self, report):
        self.report = report
        super(AmbiguousHeader, self).__init__(
            '{} layouts survive the header audit; refusing to decode. '
            'Consensus columns: {}'.format(
                len(report.valid_candidates),
                ', '.join(field.name for field in report.consensus_fields)
                or '(none)'))


class FormatVariant(object):
    """A declared DBF format variant and the rules its header must obey.

    Parameters
    ----------
    name:
        Human readable name used in audit reports.
    versions:
        Set of accepted version bytes (the first byte of the file).
    strategies:
        Iterable of allowed layout strategies. ``'cumulative'`` derives
        field offsets by accumulating field widths starting at byte 1.
        ``'address'`` takes the offsets from the descriptor address
        fields (as written by Visual FoxPro).
    reserved_must_be_zero:
        If true, every reserved byte in header and descriptors must be
        zero, the transaction/encryption flags must be clear and the
        work area id / set-fields flag must be zero.
    address_must_be_zero:
        If true, descriptor address fields must be zero (typical for
        dBASE III style files).
    require_contiguous:
        If true, fields must be packed with no gaps or overlaps.
    allow_trailing_padding:
        If true, a recordlen larger than the computed field width is
        accepted as trailing padding rather than a contradiction.
    allow_header_backlink:
        If true, bytes may follow the field terminator (the Visual
        FoxPro database backlink area) up to the declared headerlen.
    field_types:
        Optional set of accepted one-character field types.
    """

    def __init__(self, name, versions, strategies=('cumulative',),
                 reserved_must_be_zero=True, address_must_be_zero=False,
                 require_contiguous=True, allow_trailing_padding=False,
                 allow_header_backlink=False, field_types=None):
        self.name = name
        self.versions = frozenset(versions)
        self.strategies = tuple(strategies)
        self.reserved_must_be_zero = reserved_must_be_zero
        self.address_must_be_zero = address_must_be_zero
        self.require_contiguous = require_contiguous
        self.allow_trailing_padding = allow_trailing_padding
        self.allow_header_backlink = allow_header_backlink
        self.field_types = (frozenset(field_types) if field_types is not None
                            else DEFAULT_FIELD_TYPES)


DEFAULT_VARIANTS = (
    FormatVariant('visual_foxpro', (0x30, 0x31, 0x32),
                  strategies=('cumulative', 'address'),
                  allow_header_backlink=True),
    FormatVariant('foxbase_dbase3', (0x02, 0x03, 0x83, 0xFB, 0xF5),
                  strategies=('cumulative',),
                  address_must_be_zero=True),
    FormatVariant('dbase4', (0x43, 0x63, 0x8B, 0xCB, 0xE5),
                  strategies=('cumulative',),
                  address_must_be_zero=True),
)


class FieldLayout(object):
    """One field descriptor plus the offset assigned by a layout."""

    def __init__(self, descriptor, offset):
        self.name = descriptor['name']
        self.type = descriptor['type']
        self.length = descriptor['length']
        self.decimal_count = descriptor['decimal_count']
        self.address = descriptor['address']
        self.offset = offset
        self.end = offset + self.length

    def as_range(self):
        return (self.offset, self.end)

    def __repr__(self):
        return ('FieldLayout(name={!r}, type={!r}, offset={}, '
                'length={})').format(
                    self.name, self.type, self.offset, self.length)


class CandidateLayout(object):
    """One concrete (variant, strategy) interpretation of a file."""

    def __init__(self, variant_name, strategy, header, fields):
        self.variant_name = variant_name
        self.strategy = strategy
        self.name = '{}:{}'.format(variant_name, strategy)
        self.header = header
        self.fields = fields
        self.reasons = []
        self.valid = True
        self.computed_width = None

    @property
    def headerlen(self):
        return self.header['headerlen']

    @property
    def recordlen(self):
        return self.header['recordlen']

    def reject(self, reason):
        self.valid = False
        self.reasons.append(reason)

    def field_by_name(self, name):
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def read_field(self, data, record_index, field):
        start = self.headerlen + record_index * self.recordlen + field.offset
        return data[start:start + field.length]

    def signature(self):
        """Tuple used to tell whether two layouts are truly distinct."""
        return tuple((field.name, field.type, field.offset, field.length)
                     for field in self.fields) + (
            self.headerlen, self.recordlen)

    def __repr__(self):
        return '<CandidateLayout {} valid={}>'.format(self.name, self.valid)


class AuditReport(object):
    """Result of an audit.

    Attributes
    ----------
    candidates:
        Every layout that was tried, with validity flags and reasons.
    valid_candidates:
        Layouts that survived every rule and sample check.
    unique:
        True when exactly one distinct layout survived.
    layout:
        The unique ``CandidateLayout``, or ``None`` when the header is
        ambiguous or no layout survived.
    consensus_fields:
        Fields whose name, type, offset and length agree across every
        surviving layout.
    agreed_ranges:
        Merged ``[start, end)`` byte ranges (relative to the start of a
        record, byte 0 being the deletion marker) that every surviving
        layout reads identically.
    disputed_fields:
        One entry per field whose range differs between surviving
        layouts, each with a pair of candidates that read different raw
        bytes for it in the first physical record.
    """

    def __init__(self, filename, data, candidates):
        self.filename = filename
        self.data = data
        self.candidates = candidates
        self.valid_candidates = [c for c in candidates if c.valid]
        signatures = set(c.signature() for c in self.valid_candidates)
        self.unique = len(signatures) == 1
        self.layout = self.valid_candidates[0] if self.unique else None
        self.consensus_fields = []
        self.agreed_ranges = []
        self.disputed_fields = []
        if self.valid_candidates:
            self._build_consensus()
            self._build_disputes()

    def _build_consensus(self):
        survivors = self.valid_candidates
        first = survivors[0]
        for field in first.fields:
            if all(self._field_matches(candidate, field)
                   for candidate in survivors[1:]):
                self.consensus_fields.append(field)

        ranges = [(0, 1)] if len(set(c.recordlen for c in survivors)) == 1 \
            else []
        for field in self.consensus_fields:
            ranges.append(field.as_range())
        self.agreed_ranges = _merge_ranges(ranges)

    @staticmethod
    def _field_matches(candidate, field):
        other = candidate.field_by_name(field.name)
        return (other is not None
                and other.type == field.type
                and other.offset == field.offset
                and other.length == field.length)

    def _build_disputes(self):
        if len(self.valid_candidates) < 2:
            return
        data = self.data
        first = self.valid_candidates[0]
        for field in first.fields:
            witnesses = []
            for candidate in self.valid_candidates:
                other = candidate.field_by_name(field.name)
                if other is None or other.as_range() != field.as_range():
                    witnesses.append((candidate, other))
            if not witnesses:
                continue
            pair = self._find_distinct_pair(data, field, witnesses)
            self.disputed_fields.append({
                'name': field.name,
                'pair': pair,
                'candidates': [candidate.name for candidate, _ in witnesses],
            })

    def _find_distinct_pair(self, data, field, witnesses):
        baseline = self.valid_candidates[0]
        base_bytes = baseline.read_field(data, 0, field)
        for candidate, other in witnesses:
            if other is None:
                continue
            raw = candidate.read_field(data, 0, other)
            if raw != base_bytes:
                return {
                    'candidates': (baseline.name, candidate.name),
                    'ranges': (field.as_range(), other.as_range()),
                    'raw_values': (base_bytes, raw),
                }
        candidate, other = witnesses[0]
        return {
            'candidates': (baseline.name, candidate.name),
            'ranges': (field.as_range(),
                       other.as_range() if other is not None else None),
            'raw_values': (
                base_bytes,
                candidate.read_field(data, 0, other)
                if other is not None else None),
            'note': 'ranges differ but the first record holds equal bytes',
        }

    def raise_if_ambiguous(self):
        if not self.valid_candidates:
            reasons = []
            for candidate in self.candidates:
                for reason in candidate.reasons:
                    reasons.append('{}: {}'.format(candidate.name, reason))
            raise AuditFailure(
                'no legal layout found for {!r}:\n{}'.format(
                    self.filename, '\n'.join(reasons)))
        if not self.unique:
            raise AmbiguousHeader(self)


def _merge_ranges(ranges):
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _parse_header(data):
    values = _H.unpack(data[:HEADER_SIZE])
    return dict(zip(
        ['dbversion', 'year', 'month', 'day', 'numrecords', 'headerlen',
         'recordlen', 'reserved1', 'incomplete_transaction',
         'encryption_flag', 'free_record_thread', 'reserved2', 'reserved3',
         'mdx_flag', 'language_driver', 'reserved4'],
        values))


def _parse_descriptors(data, headerlen, encoding):
    descriptors = []
    pos = HEADER_SIZE
    while pos < len(data):
        if data[pos] == TERMINATOR:
            return descriptors, pos
        if pos + FIELD_SIZE > len(data) or pos + FIELD_SIZE > headerlen:
            break
        chunk = data[pos:pos + FIELD_SIZE]
        values = _field_struct.unpack(chunk)
        descriptor = dict(zip(_FIELD_NAMES, values))
        descriptor['type'] = chr(ord(descriptor['type']))
        if descriptor['type'] == 'C':
            descriptor['length'] |= descriptor['decimal_count'] << 8
            descriptor['decimal_count'] = 0
        descriptor['name'] = descriptor['name'].split(b'\0')[0].decode(
            encoding, errors='replace')
        descriptors.append(descriptor)
        pos += FIELD_SIZE
    return descriptors, None


def _cumulative_fields(descriptors):
    fields = []
    offset = 1
    for descriptor in descriptors:
        fields.append(FieldLayout(descriptor, offset))
        offset += descriptor['length']
    return fields, offset


def _address_fields(descriptors):
    fields = []
    expected = 1
    for descriptor in descriptors:
        address = descriptor['address']
        if address < expected:
            raise ValueError(
                'descriptor address {} is not strictly increasing'.format(
                    address))
        fields.append(FieldLayout(descriptor, address))
        expected = address + descriptor['length']
    if not fields:
        return fields, 1
    return fields, fields[-1].end


def _check_header_reserved(candidate, variant):
    if not variant.reserved_must_be_zero:
        return
    header = candidate.header
    for name in ('reserved1', 'free_record_thread', 'reserved2',
                 'reserved3', 'reserved4'):
        if header[name] != 0:
            candidate.reject('header {} is {} (reserved, must be zero)'.format(
                name, header[name]))
    for name in ('incomplete_transaction', 'encryption_flag'):
        if header[name] != 0:
            candidate.reject('header {} is {} (must be zero)'.format(
                name, header[name]))


def _check_descriptors(candidate, variant, terminator_pos, raw_descriptors):
    headerlen = candidate.headerlen
    descriptor_end = (HEADER_SIZE + FIELD_SIZE * len(candidate.fields) + 1
                      if terminator_pos is not None else None)
    if terminator_pos is None:
        candidate.reject('field terminator (0x0D) not found')
    elif terminator_pos != headerlen - 1:
        # Visual FoxPro keeps a database backlink area between the
        # terminator and the first record; other variants do not.
        if not variant.allow_header_backlink or descriptor_end > headerlen:
            candidate.reject(
                'headerlen {} does not match descriptor table ending at {}'
                .format(headerlen, terminator_pos + 1))
    if descriptor_end is not None and \
            not variant.allow_header_backlink and \
            headerlen != descriptor_end:
        candidate.reject(
            'headerlen {} != 32 + 32*{} fields + 1'.format(
                headerlen, len(candidate.fields)))

    for descriptor, field in zip(raw_descriptors, candidate.fields):
        if field.length < 1:
            candidate.reject('field {!r} has zero length'.format(field.name))
        if field.type not in variant.field_types:
            candidate.reject(
                'field {!r} has unsupported type {!r}'.format(
                    field.name, field.type))
        if field.decimal_count > field.length:
            candidate.reject(
                'field {!r} has more decimals ({}) than width ({})'.format(
                    field.name, field.decimal_count, field.length))
        if variant.reserved_must_be_zero:
            for name in ('reserved1', 'reserved2', 'reserved3',
                         'reserved4', 'workarea_id', 'set_fields_flag'):
                if _is_nonzero(descriptor[name]):
                    candidate.reject(
                        'field {!r} reserved {} is {!r} (must be zero)'.format(
                            field.name, name, descriptor[name]))
                    break
            if descriptor['index_field_flag'] not in (0, 1):
                candidate.reject(
                    'field {!r} has illegal index flag {!r}'.format(
                        field.name, descriptor['index_field_flag']))
        if variant.address_must_be_zero and descriptor['address'] != 0:
            candidate.reject(
                'field {!r} address is {} (variant requires zero)'.format(
                    field.name, descriptor['address']))


def _is_nonzero(value):
    if isinstance(value, bytes):
        return value.strip(b'\x00') != b''
    return value != 0


def _iter_raw_descriptors(data, count):
    for index in range(count):
        pos = HEADER_SIZE + index * FIELD_SIZE
        values = _field_struct.unpack(data[pos:pos + FIELD_SIZE])
        yield dict(zip(_FIELD_NAMES, values))


def _check_record_width(candidate, variant, computed_width):
    if variant.require_contiguous:
        for previous, current in zip(candidate.fields, candidate.fields[1:]):
            if current.offset != previous.end:
                candidate.reject(
                    'field {!r} starts at {} but previous field ends at {}'
                    .format(current.name, current.offset, previous.end))
    if candidate.recordlen == computed_width:
        return
    if variant.allow_trailing_padding and \
            candidate.recordlen > computed_width:
        return
    candidate.reject(
        'header recordlen {} != computed record width {}'.format(
            candidate.recordlen, computed_width))


def _check_file_boundary(candidate, data, sample_count):
    header = candidate.header
    expected_size = header['headerlen'] + \
        header['numrecords'] * candidate.recordlen
    if len(data) < expected_size:
        candidate.reject(
            'file is {} bytes but header requires at least {} bytes'.format(
                len(data), expected_size))
        return
    checked = min(header['numrecords'], sample_count)
    for index in range(checked):
        marker = data[header['headerlen'] + index * candidate.recordlen]
        if marker not in RECORD_MARKERS:
            candidate.reject(
                'record {} starts with marker 0x{:02x}, not a record marker'
                .format(index, marker))
            return
        if marker == END_OF_FILE:
            candidate.reject(
                'end-of-file marker appears {} records early'.format(
                    header['numrecords'] - index))
            return


def _check_known_rows(candidate, data, known_rows):
    if not known_rows:
        return
    for record_index, expectations in enumerate(known_rows):
        for name, expectation in expectations.items():
            field = candidate.field_by_name(name)
            if field is None:
                candidate.reject(
                    'known row {} references unknown field {!r}'.format(
                        record_index, name))
                continue
            raw = candidate.read_field(data, record_index, field)
            if callable(expectation):
                if not expectation(raw):
                    candidate.reject(
                        'known row {} field {!r} failed predicate on {!r}'
                        .format(record_index, name, raw))
            elif raw != expectation:
                candidate.reject(
                    'known row {} field {!r} reads {!r}, expected {!r}'
                    .format(record_index, name, raw, expectation))


class HeaderAuditor(object):
    """Audits DBF headers against a declared set of format variants."""

    def __init__(self, variants=None, sample_count=DEFAULT_SAMPLE_COUNT):
        self.variants = tuple(variants if variants is not None
                              else DEFAULT_VARIANTS)
        self.sample_count = sample_count

    def audit(self, filename, known_rows=None, encoding='ascii'):
        with open(filename, 'rb') as infile:
            data = infile.read()
        return self.audit_bytes(data, filename, known_rows=known_rows,
                                encoding=encoding)

    def audit_bytes(self, data, filename='<bytes>', known_rows=None,
                    encoding='ascii'):
        if len(data) < HEADER_SIZE:
            raise AuditFailure(
                '{!r} is shorter than the 32 byte DBF header'.format(
                    filename))
        header = _parse_header(data)
        descriptors, terminator_pos = _parse_descriptors(
            data, header['headerlen'], encoding)
        raw_descriptors = list(
            _iter_raw_descriptors(data, len(descriptors)))

        candidates = []
        for variant in self.variants:
            if header['dbversion'] not in variant.versions:
                continue
            for strategy in variant.strategies:
                candidate = self._build_candidate(
                    variant, strategy, header, descriptors)
                candidates.append(candidate)
                self._validate(candidate, variant, terminator_pos,
                               raw_descriptors, data, known_rows)
        if not candidates:
            raise AuditFailure(
                'no declared variant accepts version byte 0x{:02x} of {!r}'
                .format(header['dbversion'], filename))
        return AuditReport(filename, data, candidates)

    def _build_candidate(self, variant, strategy, header, descriptors):
        try:
            if strategy == 'cumulative':
                fields, computed_width = _cumulative_fields(descriptors)
            elif strategy == 'address':
                fields, computed_width = _address_fields(descriptors)
            else:
                raise ValueError('unknown strategy {!r}'.format(strategy))
        except ValueError as err:
            candidate = CandidateLayout(variant.name, strategy, header, [])
            candidate.reject(str(err))
            return candidate
        candidate = CandidateLayout(variant.name, strategy, header, fields)
        candidate.computed_width = computed_width
        return candidate

    def _validate(self, candidate, variant, terminator_pos,
                  raw_descriptors, data, known_rows):
        if not candidate.valid:
            # Layout construction already failed (e.g. bad addresses).
            return
        if not candidate.fields:
            candidate.reject('no field descriptors found')
            return
        _check_header_reserved(candidate, variant)
        _check_descriptors(candidate, variant, terminator_pos,
                           raw_descriptors)
        _check_record_width(candidate, variant, candidate.computed_width)
        if candidate.valid:
            _check_file_boundary(candidate, data, self.sample_count)
        _check_known_rows(candidate, data, known_rows)


def audit_header(filename, variants=None, known_rows=None, encoding='ascii',
                 sample_count=DEFAULT_SAMPLE_COUNT):
    """Convenience wrapper around :class:`HeaderAuditor`."""
    return HeaderAuditor(variants, sample_count=sample_count).audit(
        filename, known_rows=known_rows, encoding=encoding)
