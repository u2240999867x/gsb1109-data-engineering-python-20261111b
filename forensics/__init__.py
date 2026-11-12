"""Forensic replay of ambiguous TinyDB ``upsert`` import logs.

The old importer only persisted, per source change, the IDs returned by
:meth:`tinydb.table.Table.upsert` and the final row snapshots. When several
master rows share the same external key, a single condition-based upsert can
overwrite more than one row while the log keeps no before-image. This package
replays candidate histories on real TinyDB instances, enumerates every seed
state compatible with the known finite field domains, and derives inverse
patches for rollback.
"""

from .model import (
    CondChange,
    DocumentChange,
    LogEntry,
    ReplayResult,
    candidate_old_values,
    enumerate_seed_states,
    replay,
    rollback,
)

__all__ = (
    "CondChange",
    "DocumentChange",
    "LogEntry",
    "ReplayResult",
    "candidate_old_values",
    "enumerate_seed_states",
    "replay",
    "rollback",
)
