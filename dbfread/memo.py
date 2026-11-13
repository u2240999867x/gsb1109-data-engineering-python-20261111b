"""
Reads data from FPT (memo) files.

FPT files are used to varying lenght text or binary data which is too
large to fit in a DBF field.

VFP == Visual FoxPro
DB3 == dBase III
DB4 == dBase IV

When a :class:`~dbfread.memo_audit.ReadBudget` is supplied the memo
files are read under audit control. Two cases are kept separate:

* Visual FoxPro (FPT) and dBase IV memo blocks carry an explicit length
  in their block header, so the whole extent is known before any of the
  payload is read. An oversized block is quarantined without ever
  reading its payload.
* dBase III (DBT) memo blocks have no length field. The end of the
  block has to be discovered by reading fixed size chunks and looking
  for the end marker, so budget and per-value limits are enforced one
  chunk at a time.
"""
from .ifiles import ifind
from .struct_parser import StructParser
from .memo_audit import (
    BUDGET_EXHAUSTED,
    MEMO_TRUNCATED,
    VALUE_EXCEEDS_LIMIT,
)


VFPFileHeader = StructParser(
    'FPTHeader',
    '>LHH504s',
    ['nextblock',
     'reserved1',
     'blocksize',
     'reserved2'])

VFPMemoHeader = StructParser(
    'FoxProMemoHeader',
    '>LL',
    ['type',
     'length'])

DB4MemoHeader = StructParser(
    'DBase4MemoHeader',
    '<LL',
    ['reserved',  # Always 0xff 0xff 0x08 0x08.
     'length'])


class VFPMemo(bytes):
    pass


# Used for Visual FoxPro memos to distinguish binary from text memos.
class BinaryMemo(VFPMemo):
    pass


class PictureMemo(BinaryMemo):
    pass


class ObjectMemo(BinaryMemo):
    pass


class TextMemo(VFPMemo):
    pass


VFP_TYPE_MAP = {
    0x0: PictureMemo,
    0x1: TextMemo,
    0x2: ObjectMemo,
}


class MemoFile(object):
    def __init__(self, filename, budget=None):
        self.filename = filename
        # ReadBudget instance when audit limits are in force, else None.
        self.budget = budget
        # (record number, field name) of the field currently being parsed.
        self.audit_ref = None
        self._open()
        self._init()

    def _init(self):
        pass

    def _open(self):
        self.file = open(self.filename, 'rb')
        # Shortcuts for speed.
        self._read = self.file.read
        self._seek = self.file.seek

    def _close(self):
        self.file.close()

    def __getitem__(self, index):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._close()
        return False


class FakeMemoFile(MemoFile):
    def __init__(self, filename, budget=None):
        # There is no underlying file, so the budget is irrelevant.
        super(FakeMemoFile, self).__init__(filename, budget=None)

    def __getitem__(self, i):
        return None

    def _open(self):
        pass

    _init = _close = _open


class _LengthKnownMemoFile(MemoFile):
    """Shared logic for memo blocks whose header declares their length."""

    block_size = 0
    memo_header = None

    def _block_offset(self, index):
        return index * self.block_size

    def _read_unlimited(self, index):
        block_offset = self._block_offset(index)
        self._seek(block_offset)
        memo_header = self.memo_header.read(self.file)

        data = self._read(memo_header.length)
        if len(data) != memo_header.length:
            raise IOError('EOF reached while reading memo')

        return self._build_memo(memo_header, data)

    def _read_budgeted(self, index):
        budget = self.budget
        if index in budget.cache:
            return budget.cache[index]
        if index in budget.quarantined:
            return budget.quarantined[index]
        if budget.exhausted:
            return budget.quarantine(index, BUDGET_EXHAUSTED,
                                     detail='budget exhausted before the '
                                            'memo block header was read')

        block_offset = self._block_offset(index)
        header_end = block_offset + self.memo_header.size

        # Gate the header read itself so no physical byte crosses the
        # remaining budget. Overlapping headers of already reserved
        # blocks cost nothing extra.
        remaining = budget.remaining
        if remaining is not None and budget.unreserved_count(
                block_offset, header_end) > remaining:
            return budget.quarantine(
                index, BUDGET_EXHAUSTED, range=(block_offset, None),
                detail='no budget left to read the memo block header')

        self._seek(block_offset)
        memo_header = self.memo_header.read(self.file)
        data_end = header_end + memo_header.length
        declared_range = (block_offset, data_end)
        budget.declare(block_offset, data_end, ref=self.audit_ref)

        # The header itself has now been read and is paid for.
        budget.reserve(block_offset, header_end)

        if not budget.value_is_allowed(memo_header.length):
            return budget.quarantine(
                index, VALUE_EXCEEDS_LIMIT, range=declared_range,
                detail='declared memo length {} exceeds the per-value '
                       'limit'.format(memo_header.length))

        # Reserve the whole extent up front. Overlap with the header read
        # is merged away, so shared blocks are only counted once. If this
        # fails the payload is never read.
        if not budget.reserve(block_offset, data_end):
            return budget.quarantine(index, BUDGET_EXHAUSTED,
                                     range=declared_range)

        data = self._read(memo_header.length)
        if len(data) != memo_header.length:
            return budget.quarantine(
                index, MEMO_TRUNCATED, range=declared_range,
                detail='only {} of {} declared bytes could be read'
                       .format(len(data), memo_header.length))

        memo = self._build_memo(memo_header, data)
        budget.cache[index] = memo
        return memo

    def _build_memo(self, memo_header, data):
        raise NotImplementedError

    def __getitem__(self, index):
        if index <= 0:
            return None
        if self.budget is None:
            return self._read_unlimited(index)
        return self._read_budgeted(index)


class VFPMemoFile(_LengthKnownMemoFile):
    memo_header = VFPMemoHeader

    def _init(self):
        self.header = VFPFileHeader.read(self.file)

    @property
    def block_size(self):
        return self.header.blocksize

    def _build_memo(self, memo_header, data):
        return VFP_TYPE_MAP.get(memo_header.type, BinaryMemo)(data)


class DB3MemoFile(MemoFile):
    """dBase III memo file.

    Blocks have no length field. The end of a memo is found by scanning
    fixed size chunks for the ``\\x1a`` marker. Under a budget this
    probing stops as soon as a limit is hit, leaving the rest of the
    block unread and unknown.
    """

    block_size = 512
    end_marker = b'\x1a'

    def _read_unlimited(self, index):
        block_offset = index * self.block_size
        self._seek(block_offset)
        data = b''
        while True:
            newdata = self._read(self.block_size)
            if not newdata:
                return data
            data += newdata

            # Todo: some files (help.dbt) has only one field separator.
            # Is this enough for all file though?
            end_of_memo = data.find(self.end_marker)
            if end_of_memo != -1:
                return data[:end_of_memo]

            # Alternative end of memo markers:
            # \x1a\x1a
            # \x0d\x0a

    def _read_budgeted(self, index):
        budget = self.budget
        if index in budget.cache:
            return budget.cache[index]
        if index in budget.quarantined:
            return budget.quarantined[index]

        block_offset = index * self.block_size
        self._seek(block_offset)
        data = b''
        position = block_offset

        while True:
            if budget.exhausted:
                return self._quarantine_unknown(
                    index, block_offset, BUDGET_EXHAUSTED,
                    'budget exhausted before the end marker was found')

            # Stop before a chunk that would cross the per-value limit,
            # so an oversized memo is never read in full.
            next_size = len(data) + self.block_size
            if not budget.value_is_allowed(next_size):
                return self._quarantine_unknown(
                    index, block_offset, VALUE_EXCEEDS_LIMIT,
                    'memo grows past the per-value limit before an end '
                    'marker was found')

            # Never physically read more bytes than the budget allows.
            read_size = self.block_size
            if budget.remaining is not None:
                read_size = min(read_size, budget.remaining)
            chunk = self._read(read_size)
            if not chunk:
                # EOF acts as the boundary of the last memo in the file.
                end = position
                budget.declare(block_offset, end, ref=self.audit_ref)
                budget.reserve(block_offset, end)
                budget.cache[index] = data
                return data

            chunk_end = position + len(chunk)
            if not budget.reserve(position, chunk_end):
                return self._quarantine_unknown(
                    index, block_offset, BUDGET_EXHAUSTED,
                    'no budget left to read the next memo chunk')

            marker = chunk.find(self.end_marker)
            if marker != -1:
                end = position + marker
                budget.declare(block_offset, end, ref=self.audit_ref)
                memo = data + chunk[:marker]
                budget.cache[index] = memo
                return memo

            data += chunk
            position = chunk_end

    def _quarantine_unknown(self, index, block_offset, reason, detail):
        # The boundary was never reached, so no precise end is declared.
        self.budget.declare(block_offset, None, ref=self.audit_ref)
        return self.budget.quarantine(index, reason,
                                      range=(block_offset, None),
                                      detail=detail)

    def __getitem__(self, index):
        if index <= 0:
            return None
        if self.budget is None:
            return self._read_unlimited(index)
        return self._read_budgeted(index)


class DB4MemoFile(_LengthKnownMemoFile):
    """dBase IV memo file"""

    block_size = 512
    memo_header = DB4MemoHeader
    end_marker = b'\x1f'

    def _build_memo(self, memo_header, data):
        # Todo: fields are terminated in different ways.
        # \x1a is one of them
        # \x1f seems to be another (dbase_8b.dbt)
        return data.split(self.end_marker, 1)[0]


def find_memofile(dbf_filename):
    for ext in ['.fpt', '.dbt']:
        name = ifind(dbf_filename, ext=ext)
        if name:
            return name
    else:
        return None


def open_memofile(filename, dbversion, budget=None):
    if filename.lower().endswith('.fpt'):
        return VFPMemoFile(filename, budget=budget)
    else:
        # print('######', dbversion)
        if dbversion == 0x83:
            return DB3MemoFile(filename, budget=budget)
        else:
            return DB4MemoFile(filename, budget=budget)
