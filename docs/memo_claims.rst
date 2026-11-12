Memo File Claim Preview
=======================

A DBF table stores long text and binary data in a separate memo file
(``.FPT`` for Visual FoxPro, ``.DBT`` for dBase III/IV).  The table only
holds *block numbers* that point into that file, so when several same-named
memo files are lying around -- ``memotest.FPT``, ``memotest.fpt``,
``memotest.dbt`` -- there is no reliable pairing list saying which one
belongs to the table.

``dbfread.memo_claim`` is a strictly read-only preview that helps you decide.
It never writes to, truncates, or modifies any file.  It opens every
candidate, rejects the incompatible ones from the bytes alone, and compares
the survivors row by row.

Four independent pieces of evidence reject a candidate:

1. **table format** -- a Visual FoxPro table can only use an ``.FPT`` and a
   dBase table can only use a ``.DBT``;
2. **referenced block range** -- every block number the table actually points
   at must resolve to a block that lies inside the candidate;
3. **declared length** -- where a block carries an explicit length (FPT and
   dBase IV) that many bytes must fit inside the file;
4. **fingerprints** -- an optional, small set of original notes you have
   independently verified, given as ``{record_number: expected_text}``.

Basic usage
-----------

.. code-block:: python

    from dbfread import DBF
    from dbfread import claim_memo_files

    table = DBF('memotest.dbf', ignore_missing_memofile=True)
    report = claim_memo_files(
        table,
        fingerprints={1: 'Alice memo', 2: 'Bob memo'})

    for verdict in report.verdicts:
        print(verdict.path, verdict.compatible, verdict.reasons)

``claim_memo_files()`` discovers every same-named memo file next to the
table.  Pass ``candidates=[...]`` to supply an explicit list instead.
``report.compatible_paths`` lists the survivors and ``report.rejected`` the
candidates that were ruled out, each with human-readable reasons.

Row by row comparison
---------------------

When more than one candidate survives, decode every referenced note in each
of them and compare the results:

.. code-block:: python

    for row in report.compare_rows():
        print(row.record_number, row.field_name, row.consensus)
        for path, value in row.values_by_candidate.items():
            print('   ', path, repr(value))

* ``report.consensus_rows`` are rows where every compatible candidate yields
  the same note.  These can be adopted into the warehouse first.
* ``report.disputed_rows`` are rows where the candidates disagree.  The full
  candidate list is kept together with the original ``record_number`` so no
  evidence is lost.

Equivalent copies
-----------------

Two candidates are fully equivalent when they decode identically on *every*
referenced block.  These are merged into a single class:

.. code-block:: python

    for equivalence_class in report.equivalence_classes():
        print(equivalence_class.members)

A byte-identical copy that only differs by file-name case lands in the same
class as the original.  The library never silently picks one file over
another based on the name -- the choice of business record is left to you.

Forensic row selection
----------------------

To confirm which class is the genuine memo file you only need to check a
handful of original notes.  ``forensic_rows()`` returns the *minimum* set of
rows whose notes separate every distinct candidate content:

.. code-block:: python

    rows = report.forensic_rows()
    print([row.record_number for row in rows])

If only some record numbers are available for verification, restrict the
choice with ``forensic_rows(verifiable_records=[...])``.  If those rows
cannot separate all candidates a :class:`dbfread.MemoClaimError` is raised
asking for more record numbers.  When all compatible candidates are
equivalent there is nothing to distinguish and an empty list is returned.
