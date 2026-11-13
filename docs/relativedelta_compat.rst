Compatibility checks for composed ``relativedelta`` rules
===========================================================

A chain of :class:`~dateutil.relativedelta.relativedelta` objects is not
interchangeable with the single object produced by adding them together.
Both forms run the same absolute-field replacement, month clamping and
weekday adjustment, but they run them once per step versus once for the
merged result.  The same is true when the chain is applied in reverse order.

The :mod:`dateutil.relativedelta_compat` module finds the start dates where
these forms disagree.  It returns maximal intervals; each interval records
the three results at its first and last date and which of the following
interactions contributed to the difference:

``absolute-field-override``
    An absolute ``year``, ``month``, ``day`` or time field replaced a
    component that another step had already changed.
``month-end-clamp``
    A step requested a day past the end of its target month, so the library
    clamped it before the next step was applied.
``leap-day``
    A February 29 was on the applied path, or ``leapdays`` affected the
    result.
``weekday-interaction``
    A ``weekday`` rule was applied to a date that had already been shifted
    by another step.

The labels are verified by re-running the same rule with each contributing
feature removed; they are not inferred from the month alone.

Library API
-----------

.. doctest:: relativedelta

    >>> from datetime import date
    >>> from dateutil.relativedelta import relativedelta
    >>> from dateutil.relativedelta_compat import find_divergences
    >>> month = relativedelta(months=1)
    >>> intervals = find_divergences(date(2021, 1, 1),
    ...                             date(2021, 12, 31),
    ...                             [month, month])
    >>> [(item.start.isoformat(), item.end.isoformat())
    ...  for item in intervals]
    [('2021-01-29', '2021-01-31'), ('2021-03-31', '2021-03-31'), ('2021-05-31', '2021-05-31'), ('2021-08-31', '2021-08-31'), ('2021-10-31', '2021-10-31')]

The first interval shows the mechanism: the sequential form clamps
January 29 to February 28 and then to March 28, while the merged two-month
delta keeps the original day and reaches March 29.  January 29 is the first
date affected; January 28 still agrees with the merged form.

.. doctest:: relativedelta

    >>> first = intervals[0]
    >>> first.causes
    ('month-end-clamp',)
    >>> first.first_results
    (datetime.date(2021, 3, 28), datetime.date(2021, 3, 29), datetime.date(2021, 3, 28))

The returned intervals cover the requested range inclusively.  For rule
chains that contain only relative fields, the Gregorian calendar and the
``relativedelta`` calculations repeat every 400 years; ranges longer than
that are evaluated by scanning one cycle and tiling it.  ``datetime``
ranges are supported when both endpoints have the same time of day; time
zone-aware ranges must also have the same UTC offset at both ends.

Command line
------------

The module is executable, which makes it usable as a regression check in a
scheduler deployment:

.. code-block:: console

    $ python -m dateutil.relativedelta_compat \
        --start 2019-01-01 --end 2021-12-31 \
        --delta 'months=1' --delta 'months=1'
    periodic=true mode=scan divergent_days=20 intervals=15
    2019-01-29..2019-01-31 (3 days) causes=month-end-clamp differs=sequential!=merged,merged!=swapped
    ...

Use ``--format json`` for structured output, ``--days`` for one JSON
object per divergent date, ``--no-cycle`` to force a literal scan, and
``--fail-on-divergence`` to make the command exit with status 1 when any
interval is found.  A delta is written as a comma-separated list of fields
(``months=1,day=31,weekday=FR(-1)``) or as a JSON object with the same
field names as :class:`~dateutil.relativedelta.relativedelta`.
