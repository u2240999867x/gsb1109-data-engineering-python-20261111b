======
parser
======

.. automodule:: dateutil.parser

Functions
---------

.. automethod:: dateutil.parser.parse

.. automethod:: dateutil.parser.isoparse

.. autofunction:: dateutil.parser.parse_consistent

Consistency checking
~~~~~~~~~~~~~~~~~~~~

``parse`` normally returns one datetime, choosing a single interpretation for
an ambiguous local time. When a local timestamp carries a wall time, an
abbreviation and/or a numeric offset, and the possible sites are known, use
``parse_consistent`` to keep every instant that agrees with all of them:

.. code-block:: python

    >>> from dateutil import tz
    >>> from dateutil.parser import parse_consistent
    >>> nyc = tz.gettz("America/New_York")
    >>> result = parse_consistent("2024-11-03 01:30 EST", sites={"NYC": nyc})
    >>> result.status
    'resolved'
    >>> result.instants
    (datetime.datetime(2024, 11, 3, 6, 30, tzinfo=tzutc()),)

Without the abbreviation, ``01:30`` occurs twice in New York, so both
candidates are retained and the result needs review:

.. code-block:: python

    >>> ambiguous = parse_consistent("2024-11-03 01:30", sites={"NYC": nyc})
    >>> ambiguous.status
    'conflict'
    >>> len(ambiguous.candidates)
    2

Passing ``sites`` (or ``consistent=True``) to ``parse`` selects the same
behavior; calls that use neither keep returning a single datetime.

Classes
-------

.. autoclass:: dateutil.parser.parserinfo
  :members:
  :undoc-members:

.. autoclass:: dateutil.parser.ConsistencyResult
  :members:

.. autoclass:: dateutil.parser.InstantCandidate
  :members:


Warnings and Exceptions
-----------------------

.. autoclass:: dateutil.parser.ParserError

.. autoclass:: dateutil.parser.UnknownTimezoneWarning

