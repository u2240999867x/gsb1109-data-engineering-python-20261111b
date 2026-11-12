class DBFNotFound(IOError):
    """Raised if the DBF file was not found."""
    pass


class MissingMemoFile(IOError):
    """Raised if the corresponding memo file was not found."""


class NullFlagsUnavailable(ValueError):
    """Raised when the null status of a field cannot be determined.

    This happens when the ``_NullFlags`` system area at the end of a
    Visual FoxPro record is truncated (for example one byte short), so
    the bit that tells whether a field is NULL is no longer available
    and the field data itself is empty. Instead of silently treating
    such fields as not null, dbfread raises this error.

    Attributes:

    ``field_name``
      Name of the field whose null flag is unavailable.

    ``recno``
      Physical (1-based) record number.

    ``bit``
      Absolute position of the missing bit in the ``_NullFlags``
      bitmap.
    """
    def __init__(self, field_name, recno, bit):
        self.field_name = field_name
        self.recno = recno
        self.bit = bit
        message = ('null flag for field {!r} is unavailable: bit {} is '
                   'beyond the end of the (truncated) _NullFlags data '
                   'in record {}'.format(field_name, bit, recno))
        ValueError.__init__(self, message)


__all__ = ['DBFNotFound', 'MissingMemoFile', 'NullFlagsUnavailable']
