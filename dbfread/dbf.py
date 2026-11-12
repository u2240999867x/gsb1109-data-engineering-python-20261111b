"""
Class to read DBF files.
"""
import os
import sys
import datetime
import platform
import collections

from .ifiles import ifind
from .struct_parser import StructParser
from .field_parser import FieldParser
from .memo import find_memofile, open_memofile, FakeMemoFile
from .codepages import guess_encoding
from .dbversions import get_dbversion_string
from .exceptions import DBFNotFound, MissingMemoFile, NullFlagsUnavailable
from .nullflags import is_nullflags_field, null_flag_layout, NullFlags

_py_version = (sys.version_info.major, sys.version_info.minor)
_py_impl = platform.python_implementation()
if _py_version >= (3, 7) or (_py_version == (3, 6) and _py_impl == 'CPython'):
    ORDERED_DICT = dict
else:
    ORDERED_DICT = collections.OrderedDict


DBFHeader = StructParser(
    'DBFHeader',
    '<BBBBLHHHBBLLLBBH',
    ['dbversion',
     'year',
     'month',
     'day',
     'numrecords',
     'headerlen',
     'recordlen',
     'reserved1',
     'incomplete_transaction',
     'encryption_flag',
     'free_record_thread',
     'reserved2',
     'reserved3',
     'mdx_flag',
     'language_driver',
     'reserved4',
     ])

DBFField = StructParser(
    'DBFField',
    '<11scLBBHBBBB7sB',
    ['name',
     'type',
     'address',
     'length',
     'decimal_count',
     'reserved1',
     'workarea_id',
     'reserved2',
     'reserved3',
     'set_fields_flag',
     'reserved4',
     'index_field_flag',
     ])


def expand_year(year):
    """Convert 2-digit year to 4-digit year."""

    if year < 80:
        return 2000 + year
    else:
        return 1900 + year


class RecordIterator(object):
    def __init__(self, table, record_type):
        self._record_type = record_type
        self._table = table

    def __iter__(self):
        return self._table._iter_records(self._record_type)

    def __len__(self):
        return self._table._count_records(self._record_type)


class DBF(object):
    """DBF table."""
    def __init__(self, filename, encoding=None, ignorecase=True,
                 lowernames=False,
                 parserclass=FieldParser,
                 recfactory=ORDERED_DICT,
                 load=False,
                 raw=False,
                 ignore_missing_memofile=False,
                 char_decode_errors='strict',
                 include_fields=None):

        self.encoding = encoding
        self.ignorecase = ignorecase
        self.lowernames = lowernames
        self.parserclass = parserclass
        self.raw = raw
        self.ignore_missing_memofile = ignore_missing_memofile
        self.char_decode_errors = char_decode_errors
        self.include_fields = include_fields

        if recfactory is None:
            self.recfactory = lambda items: items
        else:
            self.recfactory = recfactory

        # Name part before .dbf is the table name
        self.name = os.path.basename(filename)
        self.name = os.path.splitext(self.name)[0].lower()
        self._records = None
        self._deleted = None

        if ignorecase:
            self.filename = ifind(filename)
            if not self.filename:
                raise DBFNotFound('could not find file {!r}'.format(filename))
        else:
            self.filename = filename

        # Filled in by self._read_headers()
        self.memofilename = None
        self.header = None
        self.fields = []       # namedtuples
        self.field_names = []  # strings

        with open(self.filename, mode='rb') as infile:
            self._read_header(infile)
            self._read_field_headers(infile)
            self._check_headers()

            try:
                self.date = datetime.date(expand_year(self.header.year),
                                          self.header.month,
                                          self.header.day)
            except ValueError:
                # Invalid date or '\x00\x00\x00'.
                self.date = None

        self._setup_record_fields()

        self.memofilename = self._get_memofilename()

        if load:
            self.load()

    @property
    def dbversion(self):
        return get_dbversion_string(self.header.dbversion)

    def _get_memofilename(self):
        # Does the table have a memo field?
        field_types = [field.type for field in self.fields]
        if not set(field_types) & set('MGPB'):
            # No memo fields.
            return None

        path = find_memofile(self.filename)
        if path is None:
            if self.ignore_missing_memofile:
                return None
            else:
                raise MissingMemoFile('missing memo file for {}'.format(
                    self.filename))
        else:
            return path

    @property
    def loaded(self):
        """``True`` if records are loaded into memory."""
        return self._records is not None

    def load(self):
        """Load records into memory.

        This loads both records and deleted records. The ``records``
        and ``deleted`` attributes will now be lists of records.

        """
        if not self.loaded:
            self._records = list(self._iter_records(b' '))
            self._deleted = list(self._iter_records(b'*'))

    def unload(self):
        """Unload records from memory.

        The records and deleted attributes will now be instances of
        ``RecordIterator``, which streams records from disk.
        """
        self._records = None
        self._deleted = None

    @property
    def records(self):
        """Records (not included deleted ones). When loaded a list of records,
        when not loaded a new ``RecordIterator`` object.
        """
        if self.loaded:
            return self._records
        else:
            return RecordIterator(self, b' ')

    @property
    def deleted(self):
        """Deleted records. When loaded a list of records, when not loaded a
        new ``RecordIterator`` object.
        """
        if self.loaded:
            return self._deleted
        else:
            return RecordIterator(self, b'*')

    def _read_header(self, infile):
        # Todo: more checks?
        self.header = DBFHeader.read(infile)

        if self.encoding is None:
            try:
                self.encoding = guess_encoding(self.header.language_driver)
            except LookupError:
                self.encoding = 'ascii'

    def _decode_text(self, data):
        return data.decode(self.encoding, errors=self.char_decode_errors)

    def _read_field_headers(self, infile):
        while True:
            sep = infile.read(1)
            if sep in (b'\r', b'\n', b''):
                # End of field headers
                break

            field = DBFField.unpack(sep + infile.read(DBFField.size - 1))

            field.type = chr(ord(field.type))

            # For character fields > 255 bytes the high byte
            # is stored in decimal_count.
            if field.type in 'C':
                field.length |= field.decimal_count << 8
                field.decimal_count = 0

            # Field name is b'\0' terminated.
            field.name = self._decode_text(field.name.split(b'\0')[0])
            if self.lowernames:
                field.name = field.name.lower()

            self.fields.append(field)

    def _setup_record_fields(self):
        """Set up field lists used when reading records.

        This finds the Visual FoxPro _NullFlags system field (if any),
        computes the null flag bit layout from the field declarations
        and decides which fields appear in records.
        """
        self._nullflags_field = None
        for field in self.fields:
            if is_nullflags_field(field):
                self._nullflags_field = field

        if self._nullflags_field is None:
            self._null_flag_layout = None
        else:
            # Bit positions are computed from all field declarations,
            # so they stay correct even when only some of the columns
            # are read (include_fields).
            self._null_flag_layout = null_flag_layout(self.fields)

        # The _NullFlags field is interpreted and does not appear in
        # records. (In raw mode no interpretation happens, so the
        # field is returned like any other.)
        data_fields = [field for field in self.fields
                       if not is_nullflags_field(field)]

        if self.include_fields is None:
            self._record_fields = data_fields
            self._raw_record_fields = list(self.fields)
        else:
            known_names = [field.name for field in data_fields]
            unknown = [name for name in self.include_fields
                       if name not in known_names]
            if unknown:
                message = 'include_fields: unknown field(s): {}'
                raise ValueError(message.format(', '.join(unknown)))
            wanted = set(self.include_fields)
            self._record_fields = [field for field in data_fields
                                   if field.name in wanted]
            self._raw_record_fields = [field for field in self.fields
                                       if field.name in wanted]

        self.field_names = [field.name for field in self._record_fields]

    def _open_memofile(self):
        if self.memofilename and not self.raw:
            return open_memofile(self.memofilename, self.header.dbversion)
        else:
            return FakeMemoFile(self.memofilename)

    def _check_headers(self):
        field_parser = self.parserclass(self)

        """Check headers for possible format errors."""
        for field in self.fields:

            if field.type == 'I' and field.length != 4:
                message = 'Field type I must have length 4 (was {})'
                raise ValueError(message.format(field.length))

            elif field.type == 'L' and field.length != 1:
                message = 'Field type L must have length 1 (was {})'
                raise ValueError(message.format(field.length))

            elif not field_parser.field_type_supported(field.type):
                # Todo: return as byte string?
                raise ValueError('Unknown field type: {!r}'.format(field.type))

    def _skip_record(self, infile):
        # -1 for the record separator which was already read.
        infile.seek(self.header.recordlen - 1, 1)

    def _count_records(self, record_type=b' '):
        count = 0

        with open(self.filename, 'rb') as infile:
            # Skip to first record.
            infile.seek(self.header.headerlen, 0)

            while True:
                sep = infile.read(1)
                if sep == record_type:
                    count += 1
                    self._skip_record(infile)
                elif sep in (b'\x1a', b''):
                    # End of records.
                    break
                else:
                    self._skip_record(infile)

        return count

    @property
    def null_bit_positions(self):
        """Bit positions used for null determination.

        Returns an ordered dictionary mapping each field name to the
        absolute position of its null flag bit in the ``_NullFlags``
        bitmap, or ``None`` if the field is not nullable. Varchar
        fields also occupy an extra flag bit, which is included in the
        positions of the fields that follow them.

        The positions are computed from the field declarations in the
        file header, so they are the same whether all columns are read
        or only a subset (``include_fields``). If the table has no
        ``_NullFlags`` field an empty dictionary is returned.
        """
        if self._null_flag_layout is None:
            return ORDERED_DICT()
        return ORDERED_DICT((name, bits.null)
                            for name, bits in self._null_flag_layout.items())

    def _iter_records(self, record_type=b' ', meta=False):
        with open(self.filename, 'rb') as infile, \
             self._open_memofile() as memofile:

            # Skip to first record.
            infile.seek(self.header.headerlen, 0)

            if self.raw:
                def parse(_, data):
                    return data
                record_fields = self._raw_record_fields
                null_layout = None
            else:
                field_parser = self.parserclass(self, memofile)
                parse = field_parser.parse
                record_fields = self._record_fields
                null_layout = self._null_flag_layout

            # Shortcuts for speed.
            read = infile.read
            recordlen = self.header.recordlen
            fields = self.fields

            # Physical (1-based) record number. Deleted records count
            # too, since they occupy a slot in the file.
            recno = 0

            while True:
                sep = read(1)

                if sep in (b'\x1a', b''):
                    # End of records.
                    break

                recno += 1
                body = read(recordlen - 1)

                if sep != record_type:
                    continue

                # Slice the record body into field data. If the record
                # is truncated (for example a _NullFlags system area
                # that is one byte short) the last slices will be
                # shorter than declared.
                field_data = {}
                nullflags_data = b''
                pos = 0
                for field in fields:
                    data = body[pos:pos + field.length]
                    pos += field.length
                    if is_nullflags_field(field):
                        nullflags_data = data
                    field_data[field.name] = data

                if null_layout is None:
                    nullflags = None
                else:
                    nullflags = NullFlags(nullflags_data)

                nulls = {}
                items = []
                for field in record_fields:
                    data = field_data[field.name]

                    if nullflags is not None:
                        bit = null_layout[field.name].null
                        if bit is not None:
                            if nullflags.has_bit(bit):
                                is_null = nullflags.is_set(bit)
                            elif data.strip(b' \0'):
                                # The bit is lost, but the field
                                # contains data so it cannot be null.
                                is_null = False
                            else:
                                # The bit is lost and the field is
                                # empty, so the null status cannot be
                                # determined. Don't silently treat the
                                # field as not null.
                                raise NullFlagsUnavailable(
                                    field.name, recno, bit)
                            nulls[field.name] = (bit, is_null)
                            if is_null:
                                items.append((field.name, None))
                                continue

                    items.append((field.name, parse(field, data)))

                record = self.recfactory(items)

                if meta:
                    yield recno, record, nulls
                else:
                    yield record

    def iter_with_recno(self, record_type=b' '):
        """Iterate over ``(recno, record)`` pairs.

        ``recno`` is the physical (1-based) record number in the file.
        Records of other types (for example deleted records when
        reading normal records) are skipped but still count towards
        the record number.
        """
        for recno, record, _ in self._iter_records(record_type, meta=True):
            yield recno, record

    def iter_with_meta(self, record_type=b' '):
        """Iterate over ``(recno, record, nulls)`` tuples.

        ``recno`` is the physical (1-based) record number in the file.

        ``nulls`` maps the name of each nullable field to a ``(bit,
        is_null)`` tuple, where ``bit`` is the absolute position of
        the field's null flag in the ``_NullFlags`` bitmap and
        ``is_null`` tells whether the flag was set for this record.
        This makes it possible to tell a NULL value apart from an
        empty (but not null) value, since both are returned as
        ``None`` in the record. If the table has no ``_NullFlags``
        field ``nulls`` is empty.
        """
        return self._iter_records(record_type, meta=True)

    def __iter__(self):
        if self.loaded:
            return list.__iter__(self._records)
        else:
            return self._iter_records()

    def __len__(self):
        return len(self.records)

    def __repr__(self):
        if self.loaded:
            status = 'loaded'
        else:
            status = 'unloaded'
        return '<{} DBF table {!r}>'.format(status, self.filename)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.unload()
        return False
