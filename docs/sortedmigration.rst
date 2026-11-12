.. automodule:: sortedcontainers.sortedmigration


Snapshot
........

.. autoclass:: sortedcontainers.Snapshot

   .. automethod:: __init__
   .. automethod:: capture
   .. automethod:: to_dict
   .. automethod:: from_dict
   .. automethod:: to_json
   .. automethod:: from_json
   .. automethod:: migrate
   .. automethod:: __len__
   .. automethod:: __repr__


Migration
.........

.. autoclass:: sortedcontainers.Migration

   .. automethod:: __init__
   .. automethod:: cursor
   .. automethod:: __len__
   .. automethod:: __repr__


CursorPlan
..........

.. autoclass:: sortedcontainers.CursorPlan

   .. autoattribute:: continuable
   .. automethod:: tail
   .. automethod:: tail_rows
   .. automethod:: __repr__


SortPolicyRegistry
..................

.. autoclass:: sortedcontainers.SortPolicyRegistry

   .. automethod:: __init__
   .. automethod:: register
   .. automethod:: name_of
   .. automethod:: __getitem__
   .. automethod:: __contains__
   .. automethod:: __len__
   .. automethod:: __iter__
   .. automethod:: __repr__

.. autofunction:: sortedcontainers.register_policy

.. autofunction:: sortedcontainers.migrate

.. autoexception:: sortedcontainers.InvalidCursorError
