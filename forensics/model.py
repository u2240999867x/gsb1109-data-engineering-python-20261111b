"""Core replay and enumeration primitives.

Three ``upsert`` code paths (see ``tinydb/table.py`` ``Table.upsert``) are
modelled explicitly:

* ``cond-update``  -- plain ``dict`` payload + ``where(...)`` condition, one or
  more existing rows match and are merged in place;
* ``doc-id-update`` -- a :class:`tinydb.table.Document` carrying ``doc_id``
  targets exactly that row (an unknown ID falls through to insertion);
* ``insert``       -- zero matches, a brand-new row is created.

A replay records a full before/after trace for every step, which is the
information the legacy importer *did not* write to its log.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence, Union

from tinydb import TinyDB, where
from tinydb.storages import MemoryStorage
from tinydb.table import Document, Table

TABLE_NAME = "products"
EXT_KEY = "ext_id"


@dataclass(frozen=True)
class CondChange:
    """A source change applied as ``table.upsert(payload, where(field) == v)``.

    By default the condition is on the external key ``ext_id``; later source
    changes may filter on any field (e.g. ``where('status') == 'on'``), in
    which case the update-vs-insert outcome itself leaks information about
    pre-existing values.
    """

    cond_value: object
    payload: Mapping
    cond_field: str = EXT_KEY
    kind: str = "cond"

    def key(self) -> tuple[str, object]:
        return ("cond", (self.cond_field, self.cond_value))


@dataclass(frozen=True)
class DocumentChange:
    """A source change applied as ``table.upsert(Document(payload, doc_id=id))``."""

    doc_id: int
    payload: Mapping
    kind: str = "doc_id"

    def key(self) -> tuple[str, object]:
        return ("doc_id", self.doc_id)


Change = Union[CondChange, DocumentChange]


@dataclass(frozen=True)
class LogEntry:
    """One legacy log line: the step, its source change, and returned IDs."""

    step: int
    change: Change
    returned_ids: tuple[int, ...]


@dataclass
class StepTrace:
    step: int
    change: Change
    path: str
    returned_ids: tuple[int, ...]
    before: dict[int, dict]
    after: dict[int, dict]
    overwritten: dict[int, dict]
    added_fields: dict[int, list[str]]


@dataclass
class ReplayResult:
    final_table: dict[int, dict]
    returned_sequences: list[tuple[int, ...]]
    traces: list[StepTrace]

    def trace_at(self, step: int) -> StepTrace:
        return self.traces[step]


def _fresh_table() -> Table:
    db = TinyDB(storage=MemoryStorage)
    return db.table(TABLE_NAME)


def _snapshot_rows(table: Table, doc_ids: Sequence[int]) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for doc_id in doc_ids:
        doc = table.get(doc_id=doc_id)
        if doc is not None:
            rows[doc_id] = dict(doc)
    return rows


def _diff(before: Mapping, after: Mapping) -> tuple[dict, list[str]]:
    overwritten = {
        key: copy.deepcopy(before[key])
        for key in after
        if key in before and before[key] != after[key]
    }
    added = [key for key in after if key not in before]
    return overwritten, added


def replay(
    seed: Mapping[int, Mapping],
    changes: Sequence[Change],
    table_factory: Callable[[], Table] = _fresh_table,
) -> ReplayResult:
    """Replay ``changes`` on top of ``seed`` using real TinyDB semantics."""

    table = table_factory()
    for doc_id, row in seed.items():
        table.insert(Document(dict(row), doc_id=doc_id))

    traces: list[StepTrace] = []
    returned_sequences: list[tuple[int, ...]] = []

    for step, change in enumerate(changes):
        if isinstance(change, CondChange):
            cond = where(change.cond_field) == change.cond_value
            matched = [doc.doc_id for doc in table.search(cond)]
            before = _snapshot_rows(table, matched)
            if matched:
                returned = tuple(table.upsert(dict(change.payload), cond))
                path = "cond-update"
            else:
                returned = (table.upsert(dict(change.payload), cond)[0],)
                path = "insert"
        else:
            assert isinstance(change, DocumentChange)
            existed = table.contains(doc_id=change.doc_id)
            before = _snapshot_rows(table, [change.doc_id]) if existed else {}
            returned = tuple(
                table.upsert(Document(dict(change.payload), doc_id=change.doc_id))
            )
            path = "doc-id-update" if existed else "insert"

        after = _snapshot_rows(table, returned)
        overwritten: dict[int, dict] = {}
        added_fields: dict[int, list[str]] = {}
        for doc_id in returned:
            old = before.get(doc_id, {})
            over, added = _diff(old, after.get(doc_id, {}))
            if over:
                overwritten[doc_id] = over
            if added:
                added_fields[doc_id] = added

        traces.append(
            StepTrace(
                step=step,
                change=change,
                path=path,
                returned_ids=returned,
                before=before,
                after=after,
                overwritten=overwritten,
                added_fields=added_fields,
            )
        )
        returned_sequences.append(returned)

    final_table = {doc.doc_id: dict(doc) for doc in table.all()}
    return ReplayResult(
        final_table=final_table,
        returned_sequences=returned_sequences,
        traces=traces,
    )


def matches_log(
    result: ReplayResult,
    entries: Sequence[LogEntry],
    observed_final: Mapping[int, Mapping],
) -> bool:
    """Check the observable equivalence used by the legacy log."""

    if [e.returned_ids for e in entries] != result.returned_sequences:
        return False
    if {k: dict(v) for k, v in observed_final.items()} != result.final_table:
        return False
    for entry, trace in zip(entries, result.traces):
        if entry.change != trace.change:
            return False
    return True


def enumerate_seed_states(
    row_ids: Sequence[int],
    domains: Mapping[str, Sequence[object]],
    fixed_fields: Optional[Mapping[int, Mapping]] = None,
) -> list[dict[int, dict]]:
    """Enumerate every seed table allowed by finite per-field value domains.

    ``domains`` maps a field name to its finite set of possible old values.
    Rows share the same domain for a field; ``fixed_fields`` pins values that
    the log already establishes (e.g. the shared external ID).
    """

    fixed_fields = fixed_fields or {}
    field_names = list(domains)
    value_choices = [list(domains[name]) for name in field_names]
    per_row_combos = list(itertools.product(*value_choices))

    seeds: list[dict[int, dict]] = []
    for combo in itertools.product(per_row_combos, repeat=len(row_ids)):
        seed: dict[int, dict] = {}
        for doc_id, chosen in zip(row_ids, combo):
            row: dict = dict(fixed_fields.get(doc_id, {}))
            row.update(dict(zip(field_names, chosen)))
            seed[doc_id] = row
        seeds.append(seed)
    return seeds


def candidate_old_values(
    histories: Sequence[ReplayResult], step: int
) -> dict[str, object]:
    """List, per affected row and field, every old value still possible.

    Possibilities are gathered across all histories compatible with the log.
    A field that is absent before the operation in some histories contributes
    an explicit ``<absent>`` possibility, since "key missing" cannot be told
    apart from any concrete value from the log alone.
    """

    traces = [h.trace_at(step) for h in histories]
    affected = sorted({doc_id for trace in traces for doc_id in trace.returned_ids})
    fields = sorted(
        {
            name
            for trace in traces
            for old_row in trace.before.values()
            for name in old_row
        }
    )

    by_doc_id: dict[int, dict[str, object]] = {}
    for doc_id in affected:
        field_report: dict[str, object] = {}
        for name in fields:
            values: set = set()
            absent_possible = False
            for trace in traces:
                old_row = trace.before.get(doc_id, {})
                if name in old_row:
                    values.add(_hashable(old_row[name]))
                else:
                    absent_possible = True
            possibilities = sorted(values, key=repr)
            if absent_possible:
                possibilities.append("<absent>")
            field_report[name] = {
                "possible_values": possibilities,
                "absent_possible": absent_possible,
                "determined": len(possibilities) == 1,
                "unrecoverable": len(possibilities) > 1,
            }
        by_doc_id[doc_id] = field_report

    return {
        "step": step,
        "paths": sorted({trace.path for trace in traces}),
        "affected_doc_ids": affected,
        "by_doc_id": by_doc_id,
    }


def _hashable(value: object) -> object:
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def minimal_discriminating_observation(
    history_a: ReplayResult, history_b: ReplayResult, step: int
) -> list[dict[str, object]]:
    """Smallest pre-image observation separating two log-equivalent histories.

    Returns the differing ``(doc_id, field)`` pre-op values; observing any one
    of them is enough to decide which history actually happened.
    """

    before_a = history_a.trace_at(step).before
    before_b = history_b.trace_at(step).before
    probes: list[dict[str, object]] = []
    for doc_id in sorted(set(before_a) | set(before_b)):
        row_a = before_a.get(doc_id, {})
        row_b = before_b.get(doc_id, {})
        for name in sorted(set(row_a) | set(row_b)):
            value_a = row_a.get(name, "<absent>")
            value_b = row_b.get(name, "<absent>")
            if value_a != value_b:
                probes.append(
                    {
                        "doc_id": doc_id,
                        "field": name,
                        "history_a_old": value_a,
                        "history_b_old": value_b,
                    }
                )
    return probes


def inverse_patches(result: ReplayResult) -> list[dict[str, object]]:
    """Inverse patches, ordered last-write-first, for a full rollback."""

    patches: list[dict[str, object]] = []
    for trace in reversed(result.traces):
        if trace.path == "insert":
            patches.append(
                {
                    "step": trace.step,
                    "op": "remove",
                    "doc_ids": list(trace.returned_ids),
                }
            )
        else:
            for doc_id in trace.returned_ids:
                patches.append(
                    {
                        "step": trace.step,
                        "op": "restore",
                        "doc_id": doc_id,
                        "snapshot": copy.deepcopy(trace.before[doc_id]),
                    }
                )
    return patches


def repair_scope(result: ReplayResult, step: int) -> dict[str, object]:
    """Exact repair footprint of one step, derived from its before-image."""

    trace = result.trace_at(step)
    return {
        "step": step,
        "path": trace.path,
        "updated_rows": {
            doc_id: {
                "overwritten_fields": copy.deepcopy(trace.overwritten.get(doc_id, {})),
                "added_fields": list(trace.added_fields.get(doc_id, [])),
            }
            for doc_id in trace.returned_ids
            if trace.path != "insert"
        },
        "inserted_doc_ids": list(trace.returned_ids)
        if trace.path == "insert"
        else [],
    }


def rollback(table: Table, patches: Sequence[Mapping[str, object]]) -> None:
    """Apply inverse patches produced by :func:`inverse_patches` to a table.

    Restores replace the whole row (remove + insert) instead of merging, so
    fields that were absent in the before-image but added later are actually
    withdrawn -- a plain merge would leave such added keys behind.
    """

    for patch in patches:
        if patch["op"] == "remove":
            table.remove(doc_ids=list(patch["doc_ids"]))
        else:
            assert patch["op"] == "restore"
            doc_id = int(patch["doc_id"])  # type: ignore[arg-type]
            table.remove(doc_ids=[doc_id])
            snapshot = dict(patch["snapshot"])
            if snapshot:
                table.insert(Document(snapshot, doc_id=doc_id))
