"""
Read-only preview for claiming orphan memo (``.FPT`` / ``.DBT``) files.

A single DBF table can be found together with several same-named memo files
(``memotest.FPT``, ``memotest.fpt``, ``memotest.dbt`` ...) and there is often
no reliable record of which one actually belongs to the table.  This module
never writes to anything.  It opens each candidate read-only and decides, from
the bytes alone, whether the candidate can plausibly back the memo blocks the
table actually references.

Four independent pieces of evidence are used to reject incompatible
candidates:

1. **table format** -- the candidate must be a memo format the table version
   can use (Visual FoxPro ``.FPT``, dBase III/IV ``.DBT``);
2. **referenced block range** -- every memo index referenced by the table must
   resolve to a block that lies entirely inside the candidate file;
3. **declared length** -- where a block carries an explicit length (FPT and
   dBase IV) the declared number of bytes must fit inside the file;
4. **caller supplied fingerprints** -- a small set of original memo notes the
   caller has independently verified, given as ``(record_number, text)``.

Candidates that survive are compared row by row.  Rows on which every
surviving candidate produces the same note are *consensus* rows and can be
adopted first; rows where they disagree are kept as full candidate lists
together with the original record number.  Fully equivalent copies are grouped
into one class and can never be split by file-name case.
"""
import os
import struct

from .ifiles import iglob
from .memo import (
    VFPFileHeader,
    VFPMemoHeader,
    DB4MemoHeader,
    BinaryMemo,
    open_memofile,
)

# Block size used by the dBase III/IV .DBT formats.  The library itself hard
# codes this value in DB3MemoFile/DB4MemoFile.
DBT_BLOCK_SIZE = 512

FPT_EXTENSIONS = ('.fpt',)
DBT_EXTENSIONS = ('.dbt',)

# Table versions (the first header byte) that use Visual FoxPro .FPT files.
VFP_DBVERSIONS = frozenset((0x30, 0x31, 0x32))

# dBase III style memo (version 0x83) uses the marker-terminated .DBT layout;
# every other memo-capable version uses the length-prefixed dBase IV layout.
DB3_DBVERSIONS = frozenset((0x83,))


def memo_extensions_for(dbversion):
    """Return the memo file extensions a table version may use."""
    if dbversion in VFP_DBVERSIONS:
        return FPT_EXTENSIONS
    return DBT_EXTENSIONS


class MemoProbe(object):
    """Read-only structural description of one candidate memo file."""

    def __init__(self, path, kind, block_size, file_size,
                 block_count, next_block=None):
        self.path = path
        self.kind = kind
        self.block_size = block_size
        self.block_count = block_count
        self.file_size = file_size
        self.next_block = next_block

    def __repr__(self):
        return ('MemoProbe(path={!r}, kind={!r}, block_size={}, '
                'block_count={}, file_size={}, next_block={})').format(
                    self.path, self.kind, self.block_size,
                    self.block_count, self.file_size, self.next_block)


def probe_memofile(path, dbversion):
    """Open ``path`` read-only and return a :class:`MemoProbe`.

    Raises :class:`MemoClaimError` (without raising any ``IOError``) if the
    file cannot be parsed as a memo file the table version can use.
    """
    ext = os.path.splitext(path)[1].lower()
    expected = memo_extensions_for(dbversion)
    if ext not in expected:
        raise MemoClaimError(
            'format mismatch: {!r} has extension {} but table version '
            '0x{:02x} requires {}'.format(
                path, ext or '(none)', dbversion,
                ' or '.join(expected)))

    try:
        with open(path, 'rb') as handle:
            head = handle.read(512)
            file_size = handle.seek(0, os.SEEK_END)
    except (IOError, OSError) as error:
        raise MemoClaimError(
            'cannot read {!r}: {}'.format(path, error))

    if ext == '.fpt':
        if len(head) < VFPFileHeader.size:
            raise MemoClaimError(
                'truncated FPT header in {!r}'.format(path))
        header = VFPFileHeader.unpack(head[:VFPFileHeader.size])
        block_size = header.blocksize
        if block_size <= 0:
            raise MemoClaimError(
                'invalid FPT block size {} in {!r}'.format(
                    block_size, path))
        block_count = file_size // block_size
        return MemoProbe(path, 'fpt', block_size, file_size,
                         block_count, next_block=header.nextblock)
    else:
        kind = 'db3' if dbversion in DB3_DBVERSIONS else 'db4'
        block_count = file_size // DBT_BLOCK_SIZE
        return MemoProbe(path, kind, DBT_BLOCK_SIZE, file_size,
                         block_count)


def discover_candidates(dbf_filename, dbversion):
    """Find every same-named memo candidate on disk (case sensitive listing).

    Unlike :func:`dbfread.memo.find_memofile` this returns *all* matches,
    including files that only differ in case, so the caller can decide instead
    of silently picking an arbitrary one.
    """
    stem = os.path.splitext(dbf_filename)[0]
    candidates = []
    seen = set()
    for ext in memo_extensions_for(dbversion):
        for path in iglob(stem + ext):
            if path not in seen:
                seen.add(path)
                candidates.append(path)
    return sorted(candidates)


def _is_memo_field(field, dbversion):
    """B is a memo index except in Visual FoxPro; M/G/P always are."""
    if field.type == 'B':
        return dbversion not in VFP_DBVERSIONS
    return field.type in 'MGP'


def _read_referenced_indices(table):
    """Yield ``(record_number, field_name, index)`` for every memo field."""
    dbversion = table.header.dbversion
    memo_fields = [field for field in table.fields
                   if _is_memo_field(field, dbversion)]
    if not memo_fields:
        return

    with open(table.filename, 'rb') as handle:
        handle.seek(table.header.headerlen)
        record_length = table.header.recordlen
        record_number = 0
        while True:
            separator = handle.read(1)
            if separator in (b'\x1a', b''):
                break
            record_number += 1
            data = handle.read(record_length - 1)
            if separator != b' ':
                # Skip deleted records but keep the numbering aligned.
                continue
            offset = 0
            for field in table.fields:
                chunk = data[offset:offset + field.length]
                offset += field.length
                if _is_memo_field(field, dbversion):
                    index = _parse_memo_index(chunk)
                    if index > 0:
                        yield record_number, field.name, index


def _parse_memo_index(data):
    """Decode a memo index the same way FieldParser._parse_memo_index does."""
    if len(data) == 4:
        return struct.unpack('<I', data)[0]
    try:
        return int(data)
    except ValueError:
        if data.strip(b' \x00') == b'':
            return 0
        raise ValueError(
            'Memo index is not an integer: {!r}'.format(data))


class BlockInspection(object):
    """What one referenced block looks like inside one candidate file."""

    def __init__(self, index, offset, declared_length=None,
                 actual_length=None, truncated=False, error=None):
        self.index = index
        self.offset = offset
        self.declared_length = declared_length
        self.actual_length = actual_length
        self.truncated = truncated
        self.error = error

    @property
    def fits(self):
        if self.error is not None:
            return False
        if self.truncated:
            return False
        if self.declared_length is not None:
            return self.actual_length >= self.declared_length
        return True


def _inspect_fpt_block(handle, probe, index):
    offset = index * probe.block_size
    if offset + VFPMemoHeader.size > probe.file_size:
        return BlockInspection(index, offset, truncated=True)
    handle.seek(offset)
    memo_header = VFPMemoHeader.read(handle)
    available = probe.file_size - handle.tell()
    truncated = memo_header.length > available
    return BlockInspection(index, offset,
                           declared_length=memo_header.length,
                           actual_length=available,
                           truncated=truncated)


def _inspect_db4_block(handle, probe, index):
    offset = index * probe.block_size
    if offset + DB4MemoHeader.size > probe.file_size:
        return BlockInspection(index, offset, truncated=True)
    handle.seek(offset)
    memo_header = DB4MemoHeader.read(handle)
    available = probe.file_size - handle.tell()
    truncated = memo_header.length > available
    return BlockInspection(index, offset,
                           declared_length=memo_header.length,
                           actual_length=available,
                           truncated=truncated)


def _inspect_db3_block(handle, probe, index):
    # dBase III memos carry no length; the only requirement is that the
    # referenced block exists inside the file.
    offset = index * probe.block_size
    if offset >= probe.file_size:
        return BlockInspection(index, offset, truncated=True)
    return BlockInspection(index, offset,
                           actual_length=probe.file_size - offset)


def inspect_referenced_blocks(probe, references):
    """Inspect every referenced block inside one probed candidate."""
    inspections = {}
    with open(probe.path, 'rb') as handle:
        for reference in references:
            index = reference[2]
            if index in inspections:
                continue
            if probe.kind == 'fpt':
                inspection = _inspect_fpt_block(handle, probe, index)
            elif probe.kind == 'db4':
                inspection = _inspect_db4_block(handle, probe, index)
            else:
                inspection = _inspect_db3_block(handle, probe, index)
            inspections[index] = inspection
    return inspections


class CandidateVerdict(object):
    """Compatibility verdict for one candidate memo file."""

    def __init__(self, path, probe, compatible, reasons,
                 fingerprint_matches=None, fingerprint_mismatches=None):
        self.path = path
        self.probe = probe
        self.compatible = compatible
        self.reasons = list(reasons)
        self.fingerprint_matches = dict(fingerprint_matches or {})
        self.fingerprint_mismatches = dict(fingerprint_mismatches or {})

    def __repr__(self):
        return ('CandidateVerdict(path={!r}, compatible={}, '
                'reasons={!r})').format(
                    self.path, self.compatible, self.reasons)


def _notes_for_indices(memofile, indices):
    notes = {}
    for index in indices:
        try:
            value = memofile[index]
        except (IOError, OSError, ValueError, struct.error) as error:
            notes[index] = MemoReadError(index, str(error))
        else:
            notes[index] = value
    return notes


class MemoReadError(object):
    """Placeholder for a block that could not be read from a candidate."""

    def __init__(self, index, message):
        self.index = index
        self.message = message

    def __repr__(self):
        return 'MemoReadError(index={!r}, message={!r})'.format(
            self.index, self.message)

    def __eq__(self, other):
        if not isinstance(other, MemoReadError):
            return False
        same_index = self.index == other.index
        same_message = self.message == other.message
        return same_index and same_message

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((MemoReadError, self.index, self.message))


def _decode_note(value, encoding, errors):
    if value is None:
        return None
    if isinstance(value, MemoReadError):
        return value
    if isinstance(value, BinaryMemo):
        # Keep binary memos as raw bytes so they are compared exactly.
        return bytes(value)
    return value.decode(encoding, errors=errors)


def _fingerprint_indices(references, fingerprints):
    """Map verified record numbers to the memo indices they reference."""
    by_record = {}
    for record_number, field_name, index in references:
        by_record.setdefault(record_number, []).append(index)
    result = {}
    for record_number, expected_text in fingerprints.items():
        indices = by_record.get(record_number)
        if not indices:
            raise MemoClaimError(
                'fingerprint record {} has no memo field or does not '
                'exist'.format(record_number))
        if len(indices) > 1:
            raise MemoClaimError(
                'fingerprint record {} references {} memo fields; '
                'fingerprints require one memo field per record'.format(
                    record_number, len(indices)))
        result[record_number] = indices[0]
    return result


def claim_memo_files(table, candidates=None, fingerprints=None,
                     char_decode_errors=None):
    """Preview which memo files can plausibly back ``table``.

    ``table`` is a :class:`dbfread.DBF` instance.  ``candidates`` is an
    optional iterable of memo file paths; when omitted every same-named
    ``.FPT``/``.DBT`` next to the table is considered.  ``fingerprints`` is an
    optional mapping of ``{record_number: expected_note_text}`` for notes the
    caller has independently verified.

    Returns a :class:`MemoClaimReport`.  Nothing is written or modified.
    """
    fingerprints = dict(fingerprints or {})
    if char_decode_errors is None:
        char_decode_errors = table.char_decode_errors

    dbversion = table.header.dbversion
    has_memo = any(_is_memo_field(field, dbversion)
                   for field in table.fields)
    if not has_memo:
        if fingerprints:
            raise MemoClaimError(
                'the table has no memo fields, so fingerprints cannot be '
                'checked')
        return MemoClaimReport(table, [], [], [], [], fingerprints)

    references = list(_read_referenced_indices(table))
    referenced_indices = sorted({ref[2] for ref in references})

    if candidates is None:
        candidates = discover_candidates(table.filename, dbversion)
    candidates = list(candidates)

    fingerprint_lookup = _fingerprint_indices(references, fingerprints)

    verdicts = []
    for path in candidates:
        verdicts.append(_evaluate_candidate(
            path, table, referenced_indices, references,
            fingerprint_lookup, fingerprints, char_decode_errors))

    compatible = [verdict for verdict in verdicts if verdict.compatible]
    return MemoClaimReport(table, references, referenced_indices,
                           verdicts, compatible, fingerprints)


def _evaluate_candidate(path, table, referenced_indices, references,
                        fingerprint_lookup, fingerprints,
                        char_decode_errors):
    reasons = []
    try:
        probe = probe_memofile(path, table.header.dbversion)
    except MemoClaimError as error:
        return CandidateVerdict(path, None, False, [str(error)])

    inspections = inspect_referenced_blocks(probe, references)
    for index in referenced_indices:
        inspection = inspections[index]
        if not inspection.fits:
            if inspection.truncated:
                reasons.append(
                    'block {} at offset {} is truncated'.format(
                        index, inspection.offset))
            else:
                reasons.append(
                    'block {} at offset {} does not fit in the '
                    'file'.format(index, inspection.offset))

    fingerprint_matches = {}
    fingerprint_mismatches = {}
    # Fingerprints are content evidence and only meaningful once the block
    # range and declared lengths have already been shown to fit.
    if fingerprints and not reasons:
        try:
            memofile = open_memofile(path, table.header.dbversion)
        except (IOError, OSError, MemoClaimError) as error:
            reasons.append('cannot open as memo file: {}'.format(error))
        else:
            with memofile:
                notes = _notes_for_indices(
                    memofile, set(fingerprint_lookup.values()))
            for record_number, expected_text in fingerprints.items():
                index = fingerprint_lookup[record_number]
                actual = _decode_note(
                    notes[index], table.encoding, char_decode_errors)
                if isinstance(actual, MemoReadError):
                    reasons.append(
                        'fingerprint record {} (block {}): '
                        'unreadable ({})'.format(
                            record_number, index, actual.message))
                    fingerprint_mismatches[record_number] = (
                        expected_text, None)
                elif actual == expected_text:
                    fingerprint_matches[record_number] = expected_text
                else:
                    fingerprint_mismatches[record_number] = (
                        expected_text, actual)
                    reasons.append(
                        'fingerprint mismatch on record {} (block {}): '
                        'expected {!r}, got {!r}'.format(
                            record_number, index,
                            expected_text, actual))

    compatible = not reasons
    return CandidateVerdict(path, probe, compatible, reasons,
                            fingerprint_matches, fingerprint_mismatches)


class RowComparison(object):
    """Decoded notes for one table row across every compatible candidate."""

    def __init__(self, record_number, field_name, index, values_by_candidate,
                 consensus, equivalence_classes):
        self.record_number = record_number
        self.field_name = field_name
        self.index = index
        self.values_by_candidate = dict(values_by_candidate)
        self.consensus = consensus
        self.equivalence_classes = equivalence_classes

    @property
    def disputed(self):
        return not self.consensus

    @property
    def candidates(self):
        return sorted(self.values_by_candidate)

    def value(self, path):
        return self.values_by_candidate[path]


class EquivalenceClass(object):
    """A group of candidates that produce byte-identical notes."""

    def __init__(self, class_id, members, signature):
        self.class_id = class_id
        self.members = tuple(sorted(members))
        self.signature = signature

    def __repr__(self):
        return 'EquivalenceClass(id={!r}, members={!r})'.format(
            self.class_id, self.members)


class MemoClaimReport(object):
    """Result of :func:`claim_memo_files`."""

    def __init__(self, table, references, referenced_indices,
                 verdicts, compatible, fingerprints):
        self.table = table
        self.references = references
        self.referenced_indices = referenced_indices
        self.verdicts = verdicts
        self.compatible_verdicts = compatible
        self.fingerprints = fingerprints
        self._row_comparisons = None
        self._decode_errors_used = None
        self._equivalence_classes = None

    @property
    def rejected(self):
        return [verdict for verdict in self.verdicts
                if not verdict.compatible]

    @property
    def compatible_paths(self):
        return [verdict.path for verdict in self.compatible_verdicts]

    @property
    def has_unique_claim(self):
        return len(self.compatible_verdicts) == 1

    def _decode_all_notes(self, char_decode_errors):
        decoded = {}
        for verdict in self.compatible_verdicts:
            with open_memofile(verdict.path,
                               self.table.header.dbversion) as memofile:
                raw = _notes_for_indices(memofile,
                                         set(self.referenced_indices))
            decoded[verdict.path] = {
                index: _decode_note(value, self.table.encoding,
                                    char_decode_errors)
                for index, value in raw.items()
            }
        return decoded

    def compare_rows(self, char_decode_errors=None):
        """Decode every referenced note in each compatible candidate.

        Returns a list of :class:`RowComparison`, one per referenced memo
        field, ordered by record number.
        """
        if char_decode_errors is None:
            char_decode_errors = self.table.char_decode_errors
        have_cache = self._row_comparisons is not None
        same_errors = self._decode_errors_used == char_decode_errors
        if have_cache and same_errors:
            return self._row_comparisons

        # With no surviving candidates there is nothing to compare.
        if not self.compatible_verdicts:
            self._row_comparisons = []
            self._decode_errors_used = char_decode_errors
            self._equivalence_classes = []
            return self._row_comparisons

        decoded = self._decode_all_notes(char_decode_errors)
        paths = self.compatible_paths

        comparisons = []
        for record_number, field_name, index in self.references:
            values = {path: decoded[path][index] for path in paths}
            distinct = {_note_signature(value) for value in values.values()}
            consensus = len(distinct) <= 1
            classes = self._classes_for_values(values)
            comparisons.append(RowComparison(
                record_number, field_name, index, values,
                consensus, classes))

        self._row_comparisons = comparisons
        self._decode_errors_used = char_decode_errors
        # Row-level equivalence depends on the decoded values, so any cached
        # whole-file classes must be rebuilt for the new error scheme.
        self._equivalence_classes = None
        return comparisons

    def _classes_for_values(self, values):
        groups = {}
        for path, value in values.items():
            groups.setdefault(_note_signature(value), []).append(path)
        classes = []
        for class_id, (signature, members) in enumerate(
                sorted(groups.items()), start=1):
            classes.append(EquivalenceClass(class_id, members, signature))
        return classes

    @property
    def consensus_rows(self):
        return [row for row in self.compare_rows() if row.consensus]

    @property
    def disputed_rows(self):
        return [row for row in self.compare_rows() if row.disputed]

    def equivalence_classes(self):
        """Group fully equivalent candidates into single classes.

        Two candidates are equivalent when they decode identically on *every*
        referenced block.  Candidates that differ anywhere end up in separate
        classes.
        """
        if self._equivalence_classes is not None:
            return self._equivalence_classes

        rows = self.compare_rows()
        paths = self.compatible_paths
        vectors = {
            path: tuple(_note_signature(row.values_by_candidate[path])
                        for row in rows)
            for path in paths
        }
        groups = {}
        for path in paths:
            groups.setdefault(vectors[path], []).append(path)

        classes = []
        for class_id, (signature, members) in enumerate(
                sorted(groups.items()), start=1):
            classes.append(EquivalenceClass(class_id, members, signature))
        self._equivalence_classes = classes
        return classes

    def forensic_rows(self, verifiable_records=None):
        """Pick the fewest rows that distinguish every distinct candidate.

        ``verifiable_records`` restricts the choice to record numbers the
        caller can actually check (defaults to every referenced record).  The
        returned :class:`RowComparison` rows form a minimum set whose value
        vectors separate every equivalence class; if all compatible candidates
        are equivalent an empty list is returned.
        """
        rows = self.compare_rows()
        classes = self.equivalence_classes()
        if len(classes) <= 1:
            return []

        if verifiable_records is None:
            pool = list(rows)
        else:
            allowed = set(verifiable_records)
            pool = [row for row in rows
                    if row.record_number in allowed]

        # Reduce each row to the pairs of equivalence classes it separates.
        # Members of one class always agree on a given row, so only one
        # representative per class has to be read.  Two classes are separated
        # precisely when their representatives decode to different notes.
        class_ids = sorted(cls.class_id for cls in classes)
        row_coverages = []
        for row in pool:
            notes = {}
            for cls in classes:
                representative = cls.members[0]
                notes[cls.class_id] = _note_signature(
                    row.values_by_candidate[representative])
            separated = frozenset(
                frozenset((left, right))
                for position, left in enumerate(class_ids)
                for right in class_ids[position + 1:]
                if notes[left] != notes[right])
            if separated:
                row_coverages.append((row, separated))

        # Pairs of classes that still need to be separated.
        required_pairs = frozenset(
            frozenset((left, right))
            for position, left in enumerate(class_ids)
            for right in class_ids[position + 1:])

        chosen_rows = _minimum_distinguishing_rows(
            row_coverages, required_pairs)
        if chosen_rows is None:
            raise MemoClaimError(
                'the verifiable rows cannot distinguish all candidate '
                'contents; provide more record numbers')

        chosen_rows = list(chosen_rows)
        chosen_rows.sort(key=lambda row: row.record_number)
        return chosen_rows


def _minimum_distinguishing_rows(row_coverages, required_pairs):
    """Exact minimum set cover of the class pairs by row coverages.

    Each row covers the pairs of equivalence classes whose notes differ on
    that row.  Branch-and-bound over the (usually small) row set returns a
    minimum-cardinality cover; ties are broken by record number so the result
    is deterministic.  Returns ``None`` when no cover exists.
    """
    # Rows that cover the same pairs are interchangeable; keep the one with the
    # lowest record number for each coverage signature.
    unique = {}
    for row, covered in row_coverages:
        current = unique.get(covered)
        if current is None or row.record_number < current.record_number:
            unique[covered] = row
    options = sorted(unique.items(),
                     key=lambda item: item[1].record_number)

    best = [None]

    def search(remaining_pairs, selected, start):
        if not remaining_pairs:
            if best[0] is None or len(selected) < len(best[0]):
                best[0] = list(selected)
            return
        if best[0] is not None and len(selected) >= len(best[0]):
            return
        for index in range(start, len(options)):
            covered, row = options[index]
            gain = remaining_pairs & covered
            if not gain:
                continue
            selected.append(row)
            search(remaining_pairs - gain, selected, index + 1)
            selected.pop()

    search(required_pairs, [], 0)
    return best[0]


def _note_signature(value):
    """Hashable, equality-faithful key for a decoded note."""
    if value is None:
        return (0,)
    if isinstance(value, MemoReadError):
        return (1, value.index, value.message)
    if isinstance(value, bytes):
        return (2, value)
    return (3, value)


class MemoClaimError(Exception):
    """Raised when a claim preview cannot be produced."""
