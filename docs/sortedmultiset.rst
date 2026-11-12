.. automodule:: sortedcontainers.sortedmultiset


SortedMultiset
..............

.. autoclass:: sortedcontainers.SortedMultiset
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: from_values
   .. automethod:: add
   .. automethod:: subtract
   .. automethod:: adjust
   .. automethod:: set_count
   .. automethod:: discard
   .. automethod:: remove
   .. automethod:: clear
   .. automethod:: merge
   .. automethod:: withdraw
   .. automethod:: apply_deltas
   .. automethod:: transaction
   .. automethod:: snapshot
   .. automethod:: __len__
   .. automethod:: __contains__
   .. automethod:: __iter__
   .. automethod:: __reversed__
   .. automethod:: __getitem__
   .. automethod:: __eq__
   .. automethod:: __repr__
   .. automethod:: _check


SortedMultisetView
..................

.. autoclass:: sortedcontainers.SortedMultisetView
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: distinct_count
   .. automethod:: multiplicity
   .. automethod:: items
   .. automethod:: elements
   .. automethod:: bisect_left
   .. automethod:: bisect_right
   .. automethod:: rank_range
   .. automethod:: quantile
   .. automethod:: percentile
   .. automethod:: snapshot
   .. automethod:: __len__
   .. automethod:: __contains__
   .. automethod:: __iter__
   .. automethod:: __reversed__
   .. automethod:: __getitem__
   .. automethod:: __eq__
   .. automethod:: __repr__
   .. automethod:: _check


NegativeCountError
..................

.. autoclass:: sortedcontainers.NegativeCountError
   :show-inheritance:

   .. autoattribute:: offending
