"""Python Sorted Collections

SortedCollections is an Apache2 licensed Python sorted collections library.

>>> from sortedcollections import ValueSortedDict
>>> vsd = ValueSortedDict({'a': 2, 'b': 1, 'c': 3})
>>> list(vsd.keys())
['b', 'a', 'c']

:copyright: (c) 2015-2021 by Grant Jenks.
:license: Apache 2.0, see LICENSE for more details.

"""

from sortedcontainers import (
    SortedDict,
    SortedList,
    SortedListWithKey,
    SortedSet,
)

from .nearestdict import NearestDict
from .ordereddict import OrderedDict
from .recipes import (
    IndexableDict,
    IndexableSet,
    ItemSortedDict,
    OrderedSet,
    SegmentList,
    ValueSortedDict,
)
from .schedule import (
    NEAREST,
    NEXT,
    PREVIOUS,
    AlignmentResult,
    NoCandidate,
    ScheduleAlignment,
    ToleranceExceeded,
    ordinary_linear_nearest,
)

__all__ = [
    'IndexableDict',
    'IndexableSet',
    'ItemSortedDict',
    'NEAREST',
    'NEXT',
    'NearestDict',
    'NoCandidate',
    'OrderedDict',
    'OrderedSet',
    'PREVIOUS',
    'ScheduleAlignment',
    'SegmentList',
    'SortedDict',
    'SortedList',
    'SortedListWithKey',
    'SortedSet',
    'ToleranceExceeded',
    'ValueSortedDict',
    'ordinary_linear_nearest',
]

__version__ = '2.1.0'
