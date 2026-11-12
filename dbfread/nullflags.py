"""
Support for Visual FoxPro null value flags.

Visual FoxPro tables that contain nullable or varchar fields have a
hidden system field called ``_NullFlags`` (field type ``'0'``). The
field is a bitmap located at the end of each record. It contains one
bit for every nullable field plus one extra bit for every varchar
field. The bits are assigned in field declaration order, starting at
the least significant bit of the first byte. A set null bit means the
field value is NULL.

Because the bit positions are computed from the field declarations in
the file header they stay the same no matter which columns are
actually read. Skipping fields (column projection) therefore never
shifts the bits of the remaining fields.
"""
import collections

NULLFLAGS_FIELD_TYPE = '0'
NULLFLAGS_FIELD_NAME = '_nullflags'

# Values of the field flags byte (byte 18 in the field header).
SYSTEM_FIELD = 0x01
NULLABLE_FIELD = 0x02
BINARY_FIELD = 0x04

# null_flag_layout() maps each field name to one of these. ``null`` is
# the absolute bit position of the field's null flag in the _NullFlags
# bitmap (None if the field is not nullable) and ``varlength`` is the
# position of the extra flag bit that every varchar field occupies
# (None for other field types).
FieldNullBits = collections.namedtuple('FieldNullBits',
                                       ['null', 'varlength'])


def is_nullflags_field(field):
    """Check if a field is the Visual FoxPro _NullFlags system field."""
    return (field.type == NULLFLAGS_FIELD_TYPE and
            field.name.lower() == NULLFLAGS_FIELD_NAME)


def null_flag_layout(fields):
    """Compute null flag bit positions from field declarations.

    Takes the full list of field headers (as found in the file, in
    declaration order) and returns an ordered dictionary mapping field
    names to ``FieldNullBits`` namedtuples. The _NullFlags field
    itself is skipped.

    Nullable fields get one bit. Varchar fields get one additional bit
    whether they are nullable or not, so a nullable varchar field
    occupies two consecutive bits (null flag first).
    """
    layout = collections.OrderedDict()
    bit = 0
    for field in fields:
        if is_nullflags_field(field):
            continue

        null_bit = None
        varlength_bit = None

        if field.set_fields_flag & NULLABLE_FIELD:
            null_bit = bit
            bit += 1

        if field.type == 'V':
            varlength_bit = bit
            bit += 1

        layout[field.name] = FieldNullBits(null_bit, varlength_bit)

    return layout


class NullFlags(object):
    """The null flag bitmap of one record.

    ``data`` is the raw content of the record's _NullFlags field. If
    the record is truncated (for example when the system area at the
    end of the record is one byte short) the data may be shorter than
    declared. ``has_bit()`` tells whether a bit is still available so
    callers can avoid guessing null status from zero padding.
    """
    def __init__(self, data):
        self.data = data

    def has_bit(self, bit):
        """Check if the bit is present in the (possibly truncated) data."""
        return bit // 8 < len(self.data)

    def is_set(self, bit):
        """Check if the bit is set.

        The caller must check ``has_bit()`` first. Raises IndexError
        if the bit is beyond the available data.
        """
        byte = self.data[bit // 8]
        if not isinstance(byte, int):
            # Python 2 returns a one character byte string.
            byte = ord(byte)
        return bool(byte & (1 << (bit % 8)))
