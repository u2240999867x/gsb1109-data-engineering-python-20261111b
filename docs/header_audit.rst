Header Auditing
===============

Old batches of DBF files sometimes contain field descriptors that can be
read two ways: using the offsets stored in each descriptor, or using the
offsets you get by accumulating the field widths. When those two layouts
disagree, decoding immediately would mean silently betting on one of
them.

The header auditor never modifies a file. It enumerates every layout
allowed by the format variants you declare, discards the illegal ones,
and only allows decoding when exactly one interpretation survives.

Declaring format variants
-------------------------

A :class:`FormatVariant <dbfread.FormatVariant>` states which version
bytes belong to the variant and which rules a layout must satisfy:

* ``strategies`` -- ``'cumulative'`` and/or ``'address'``.
* ``reserved_must_be_zero`` -- reserved header and descriptor bytes must
  be clear.
* ``address_must_be_zero`` -- descriptor address fields must be zero, as
  in dBASE III style files.
* ``require_contiguous`` -- fields must be packed without gaps.
* ``allow_trailing_padding`` -- a record wider than the computed field
  width is accepted.
* ``allow_header_backlink`` -- bytes may follow the field terminator, as
  in the Visual FoxPro database backlink area.

A default set covering Visual FoxPro, FoxBASE/dBASE III and dBASE IV is
used when you pass nothing.

Running the audit
-----------------

.. code-block:: python

    from dbfread import HeaderAuditor

    report = HeaderAuditor().audit('old_batch.dbf')

    if report.unique:
        # Exactly one legal layout: safe to decode normally.
        ...
    else:
        # Columns every surviving layout agrees on.
        print(report.consensus_fields)
        print(report.agreed_ranges)
        # One entry per disputed column, with a pair of candidate
        # layouts that read different raw bytes for it.
        print(report.disputed_fields)

Call ``report.raise_if_ambiguous()`` to turn a non-unique result into an
:class:`AmbiguousHeader <dbfread.AmbiguousHeader>` exception. If no
layout survives, an :class:`AuditFailure <dbfread.AuditFailure>` lists
the reason each candidate was rejected.

Using known raw rows
--------------------

If you have a few rows whose original values are known from another
source, pass them as raw byte expectations. A layout that cannot
reproduce them is rejected, which can reduce several interpretations to
one. Expectations may be exact bytes or a predicate:

.. code-block:: python

    report = HeaderAuditor().audit(
        'old_batch.dbf',
        known_rows=[{'AMOUNT': b'  12.50'},
                    {'NOTE': lambda raw: raw.startswith(b'OK')}])

Gating the reader
-----------------

Pass ``header_audit=True`` to :class:`DBF <dbfread.DBF>` to run the
audit before any record is decoded. Decoding starts only when a single
layout wins. An ambiguous or fully rejected header raises instead, and
the file on disk is left byte-for-byte unchanged:

.. code-block:: python

    from dbfread import DBF, AmbiguousHeader

    try:
        table = DBF('old_batch.dbf', header_audit=True,
                    format_variants=my_variants,
                    known_rows=my_known_rows)
    except AmbiguousHeader as error:
        report = error.report  # consensus columns plus disputed pairs
