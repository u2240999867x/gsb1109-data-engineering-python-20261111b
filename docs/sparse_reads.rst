Sparse Reads
============

.. _sparse-reads:

Sometimes you already know which physical records you need: a bad-row
report typically lists them by record number.  When the storage layer
only serves byte ranges and bills per request and per byte transferred,
reading the whole table or issuing one request per row is wasteful.
The sparse read planner finds the cheapest set of contiguous byte
ranges that covers every requested row.

Record numbers are 1-based and follow physical file order, so deleted
rows still occupy their original position.  A range may pass over rows
that were not requested.  Those gap rows are transferred but never
decoded or returned.

Planning
--------

``plan_sparse_reads()`` takes the record numbers together with the
storage pricing and returns a ``SparseReadPlan``::

    >>> table = DBF('memotest.dbf')
    >>> plan = table.plan_sparse_reads([3, 1, 3],
    ...                                startup_cost=100,
    ...                                byte_cost=1)
    >>> plan.ranges
    (ReadRange(start=392, end=479, records=(1, 2, 3)),)
    >>> round(plan.cost)
    187
    >>> plan.record_order
    (3, 1, 3)

The request order, and any duplicated record numbers, are preserved in
``record_order``.  Each ``ReadRange`` is a half-open byte interval; its
``records`` tuple lists every physical slot it spans, including gap
rows.  ``max_length`` caps the length of a single read, forcing the
planner to split a range when one read could not return it all.

The planner minimises the total of the per-request startup cost and the
per-byte transfer cost.  A high startup cost merges nearby rows into one
read even when the gap has to be transferred, while a high per-byte
cost splits them into separate reads.

Executing
---------

``read_sparse()`` (or ``execute_sparse_plan()`` with a plan you built
earlier) issues one read per planned range and decodes only the
requested records::

    >>> result = table.read_sparse([3, 1, 3], startup_cost=100)
    >>> [number for number, record in result.records]
    [3, 1, 3]
    >>> result.requests
    1
    >>> result.bytes_read
    87

The deletion marker is ignored, so requesting a deleted row returns its
data instead of skipping it.  ``raw=True`` returns each record as the
raw fixed-length slot including the marker byte.

Short Reads
-----------

A read can come back shorter than planned, for example because the
object was truncated.  Rows fully contained in the returned bytes are
still decoded.  Rows with insufficient evidence are reported once per
request in ``short_reads`` rather than guessed at, and the exact
continuation range is given in both the short read and
``missing_ranges``::

    >>> result = short_table.read_sparse([1, 2, 3])
    >>> [short.record for short in result.short_reads]
    [2, 3]
    >>> result.short_reads[0].missing
    ReadRange(start=421, end=479, records=(2, 3))

Read the missing range from the storage layer and plan a follow-up
sparse read for its ``records`` to finish the job.
