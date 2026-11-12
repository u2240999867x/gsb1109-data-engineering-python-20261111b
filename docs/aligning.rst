Aligning Two Versions of a Table
================================

When auditing two versions of a DBF file the deleted records must be
kept in the comparison, but the new version may have been packed
(``PACK``), physically removing the deleted records and shifting the
record numbers of all surviving records.  ``dbfread.align()`` scans
both files in physical record order (including deleted records) and
aligns them using only two assumptions:

* the relative order of the surviving records is unchanged
* the values of a set of caller declared witness columns are
  unchanged

.. code-block:: python

    >>> from dbfread import align
    >>> report = align('old.dbf', 'new.dbf', key_fields=['NAME', 'SSN'])

The alignment maximizes the number of continued (matched) records.
All alignments that are equally optimal are taken into account, and
the resulting ``AlignmentReport`` classifies the records:

* ``continued`` -- ``Match`` pairs matched in every optimal
  alignment.
* ``deleted`` / ``added`` -- ``RowRef`` records matched in no optimal
  alignment.
* ``ambiguous`` -- ``AmbiguousSegment`` segments where duplicate
  witness values allow several optimal alignments.  For every old
  record all candidate destinations are listed (as ``CandidateSet``);
  no candidate, for example the one nearest the original record
  number, is preferred.
* ``unknown_old`` / ``unknown_new`` -- ``UnknownSegment`` objects for
  corrupted or truncated tails of each file.

Every item carries the byte range of the record or segment in its
file (``byte_range``, or ``old_byte_range`` / ``new_byte_range`` for
pairs and segments), so the raw bytes can be inspected directly.

The physical order scan is also available on its own:

.. code-block:: python

    >>> from dbfread import scan_physical
    >>> records, unknown = scan_physical('old.dbf')
    >>> for record in records:
    ...     print(record.number, record.deleted, record.byte_range)
