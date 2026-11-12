"""
Segmented, resumable replay of bulk update rules.

This module backs :meth:`tinydb.table.Table.update_multiple_segmented`. It
is meant for one-shot, potentially huge rule sources (e.g. generators
streaming rules from an external system) that cannot or should not be held
in memory all at once.

A *rule* is a plain, JSON-serializable mapping::

    {
        'cond':   {'op': '==', 'path': 'char', 'value': 'a'},
        'fields': {'int': 2, 'flag': {'$set': True}},
    }

Each rule gets a stable ``number`` (its 1-based position in the source,
unless the rule mapping explicitly provides a ``number``). Rules are
applied row by row in the table's original document order; within a row,
rules run in source order so a later rule's condition sees the values that
earlier rules wrote on the same row -- exactly like
:meth:`~tinydb.table.Table.update_multiple` with the full list.

Only one segment of rules and one segment of documents is held in memory
at a time. The rule source is materialized into a local staging directory
up front and progress is tracked in a checkpoint file, so an interrupted
run can resume from the last confirmed row/rule boundary by calling
:meth:`~tinydb.table.Table.update_multiple_segmented` again with the same
``state_path``.

Rules that need Python callbacks (callable conditions or callable field
transforms) cannot be serialized and are therefore rejected here; use
:meth:`~tinydb.table.Table.update_multiple` for them.
"""

import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Optional

from .queries import Query, QueryLike

if TYPE_CHECKING:
    from .table import Table

__all__ = ('compile_condition', 'apply_field_ops', 'run_segmented_update')

#: Markers supported in field operations (see :func:`apply_field_ops`)
_FIELD_OP_MARKERS = ('$set', '$inc', '$unset')

_STATE_VERSION = 1


def _resolve_path(path: Any) -> Query:
    """
    Build a query for a serialized field path.

    ``path`` may be a dotted string (``'a.b.c'``) or a list of keys.
    """
    if isinstance(path, str):
        parts = path.split('.') if path else []
    else:
        parts = list(path)

    query = Query()
    for part in parts:
        query = query[part]

    if not parts:
        # An empty path addresses the value itself (e.g. elements of a
        # list tested by a nested 'any'/'all' condition)
        query = query.map(lambda value: value)

    return query


def compile_condition(spec: Mapping) -> QueryLike:
    """
    Compile a serializable condition spec into a query object.

    Supported specs:

    - Comparisons: ``{'op': '=='|'!='|'<'|'<='|'>'|'>=', 'path': p,
      'value': v}``
    - ``{'op': 'exists', 'path': p}``
    - ``{'op': 'matches'|'search', 'path': p, 'pattern': r, 'flags': 0}``
    - ``{'op': 'one_of', 'path': p, 'items': [...]}``
    - ``{'op': 'any'|'all', 'path': p, 'items': [...]}`` or
      ``{'op': 'any'|'all', 'path': p, 'cond': <nested spec>}``
    - ``{'op': 'and'|'or', 'conds': [<spec>, ...]}``
    - ``{'op': 'not', 'cond': <spec>}``
    - ``{'op': 'fragment', 'document': {...}}``
    - ``{'op': 'noop'}`` (matches everything)

    :param spec: the condition spec mapping
    :returns: a query object usable like any other TinyDB query
    """
    if not isinstance(spec, Mapping):
        raise TypeError(
            'Condition specs must be mappings; callable conditions cannot '
            'be serialized -- use Table.update_multiple for callback rules'
        )

    if 'op' not in spec:
        raise ValueError("Condition spec requires an 'op' key")

    op = spec['op']

    if op in ('and', 'or'):
        conds = [compile_condition(cond) for cond in spec.get('conds', ())]
        if not conds:
            raise ValueError(f"Condition op {op!r} requires non-empty 'conds'")
        result = conds[0]
        for cond in conds[1:]:
            result = result & cond if op == 'and' else result | cond
        return result

    if op == 'not':
        return ~compile_condition(spec['cond'])

    if op == 'noop':
        return Query().noop()

    if op == 'fragment':
        return Query().fragment(spec['document'])

    query = _resolve_path(spec.get('path', ()))

    if op == '==':
        return query == spec.get('value')
    if op == '!=':
        return query != spec.get('value')
    if op == '<':
        return query < spec.get('value')
    if op == '<=':
        return query <= spec.get('value')
    if op == '>':
        return query > spec.get('value')
    if op == '>=':
        return query >= spec.get('value')
    if op == 'exists':
        return query.exists()
    if op == 'matches':
        return query.matches(spec['pattern'], spec.get('flags', 0))
    if op == 'search':
        return query.search(spec['pattern'], spec.get('flags', 0))
    if op == 'one_of':
        return query.one_of(spec['items'])
    if op in ('any', 'all'):
        if 'cond' in spec:
            cond: Any = compile_condition(spec['cond'])
        else:
            cond = spec['items']
        return query.any(cond) if op == 'any' else query.all(cond)

    raise ValueError(f'Unknown condition op: {op!r}')


def apply_field_ops(doc: MutableMapping, fields: Mapping) -> None:
    """
    Apply a rule's field operations to a document in place.

    A plain value sets the field (same as ``Table.update`` with a mapping).
    A single-key mapping whose key is a marker performs an operation
    instead:

    - ``{'$set': value}``: set the field to ``value``
    - ``{'$inc': number}``: add ``number`` to the field (missing fields
      start at ``0``)
    - ``{'$unset': true}``: remove the field from the document
    """
    for key, value in fields.items():
        if isinstance(value, Mapping) and len(value) == 1:
            marker = next(iter(value))
            if marker in _FIELD_OP_MARKERS:
                if marker == '$set':
                    doc[key] = value['$set']
                elif marker == '$inc':
                    doc[key] = doc.get(key, 0) + value['$inc']
                else:  # '$unset'
                    doc.pop(key, None)
                continue

        doc[key] = value


def _normalize_rule(raw: Mapping, position: int) -> dict:
    """
    Validate a raw rule mapping and give it a stable number.
    """
    if not isinstance(raw, Mapping):
        raise TypeError(
            'Segmented rules must be mappings with "cond" and "fields" '
            'keys; callback-style rules (callables) cannot be serialized '
            '-- use Table.update_multiple for them'
        )

    cond = raw.get('cond')
    fields = raw.get('fields')

    if not isinstance(cond, Mapping):
        raise TypeError(
            'Rule "cond" must be a serializable condition spec (a mapping); '
            'callable conditions cannot be serialized -- use '
            'Table.update_multiple for callback rules'
        )

    if not isinstance(fields, Mapping):
        raise TypeError(
            'Rule "fields" must be a mapping of field operations; callable '
            'transforms cannot be serialized -- use Table.update_multiple '
            'for callback rules'
        )

    return {
        'number': raw.get('number', position),
        'cond': dict(cond),
        'fields': dict(fields),
    }


def _rule_segment_path(staging_dir: str, index: int) -> str:
    return os.path.join(staging_dir, f'rules-{index:06d}.jsonl')


def _stage_rules(
    rules: Iterable[Mapping],
    staging_dir: str,
    segment_size: int,
) -> tuple[int, list]:
    """
    Materialize the (possibly one-shot) rule source into per-segment JSONL
    files, assigning stable numbers in source order.

    :returns: the rule count and the rule numbers in source order
    """
    count = 0
    segment_index = 0
    seen_numbers = set()
    rule_order: list = []

    handle = open(_rule_segment_path(staging_dir, segment_index), 'w',
                  encoding='utf-8')
    try:
        for position, raw in enumerate(rules, start=1):
            rule = _normalize_rule(raw, position)

            if rule['number'] in seen_numbers:
                raise ValueError(
                    f"Duplicate rule number: {rule['number']!r}"
                )
            seen_numbers.add(rule['number'])
            rule_order.append(rule['number'])

            if count and count % segment_size == 0:
                handle.close()
                segment_index += 1
                handle = open(_rule_segment_path(staging_dir, segment_index),
                              'w', encoding='utf-8')

            try:
                line = json.dumps(rule)
            except TypeError as exc:
                raise TypeError(
                    f"Rule {rule['number']!r} is not JSON-serializable and "
                    f'cannot be staged for segmented replay: {exc}'
                ) from exc

            handle.write(line + '\n')
            count += 1
    finally:
        handle.close()

    return count, rule_order


def _load_rule_segment(staging_dir: str, index: int) -> list:
    """
    Load one segment of rules from the staging area and compile the
    conditions. Only this segment is held in memory.
    """
    segment = []
    with open(_rule_segment_path(staging_dir, index), encoding='utf-8') as fh:
        for line in fh:
            rule = json.loads(line)
            segment.append((
                rule['number'],
                compile_condition(rule['cond']),
                rule['fields'],
            ))
    return segment


def _write_checkpoint(state_path: str, state: dict) -> None:
    """
    Persist the checkpoint atomically (write temp file, then rename).
    """
    tmp_path = state_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as fh:
        json.dump(state, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, state_path)


def _append_receipts(receipts_path: str, hits: list) -> int:
    """
    Append one line per hit rule of a confirmed cell to the receipts spill
    file. Returns the number of lines written.
    """
    if not hits:
        return 0

    # Group the cell's hits per rule, preserving document order
    per_rule: dict = {}
    for number, doc_id in hits:
        per_rule.setdefault(number, []).append(doc_id)

    lines = 0
    with open(receipts_path, 'a', encoding='utf-8') as fh:
        for number, doc_ids in per_rule.items():
            fh.write(json.dumps({'rule': number, 'doc_ids': doc_ids}) + '\n')
            lines += 1
        fh.flush()
        os.fsync(fh.fileno())

    return lines


def _truncate_receipts(receipts_path: str, lines: int) -> None:
    """
    Drop any receipt lines beyond the confirmed checkpoint. Such lines may
    exist if the run was interrupted after spilling receipts but before
    confirming the cell.
    """
    if not os.path.exists(receipts_path):
        return

    with open(receipts_path, encoding='utf-8') as fh:
        kept = []
        for index, line in enumerate(fh):
            if index >= lines:
                break
            kept.append(line)

    with open(receipts_path, 'w', encoding='utf-8') as fh:
        fh.writelines(kept)


def _aggregate_receipts(receipts_path: str, rule_order: list) -> list:
    """
    Combine the spilled per-cell receipts into the final receipt list, one
    entry per rule in source order, with hit rows in document order.
    """
    hits_by_rule = {number: [] for number in rule_order}

    if os.path.exists(receipts_path):
        with open(receipts_path, encoding='utf-8') as fh:
            for line in fh:
                record = json.loads(line)
                hits_by_rule[record['rule']].extend(record['doc_ids'])

    return [
        {
            'rule': number,
            'doc_ids': hits_by_rule[number],
            'count': len(hits_by_rule[number]),
        }
        for number in rule_order
    ]


def run_segmented_update(
    table: 'Table',
    rules: Iterable[Mapping],
    *,
    rule_segment_size: int = 1000,
    doc_segment_size: int = 1000,
    state_path: Optional[str] = None,
    on_checkpoint: Optional[Callable[[int, int], None]] = None,
) -> list:
    """
    Apply a (possibly one-shot) source of serializable update rules to a
    table in segments, with checkpointing for resume after interruption.

    See :meth:`tinydb.table.Table.update_multiple_segmented` for the
    user-facing documentation.
    """
    if rule_segment_size < 1:
        raise ValueError('rule_segment_size must be at least 1')
    if doc_segment_size < 1:
        raise ValueError('doc_segment_size must be at least 1')

    resumable = state_path is not None
    if state_path is None:
        # Not resumable by the caller, but the machinery stays identical:
        # staging and checkpoint live in a temporary directory.
        staging_dir = tempfile.mkdtemp(prefix='tinydb-segmented-')
        state_path = os.path.join(staging_dir, 'checkpoint.json')
    else:
        staging_dir = state_path + '.staging'

    if resumable and os.path.exists(state_path):
        # Resume: reuse the staged rules, the snapshot of the original
        # document order and the confirmed row/rule boundary. Segment
        # sizes come from the checkpoint so cells line up with the
        # interrupted run.
        with open(state_path, encoding='utf-8') as fh:
            state = json.load(fh)

        staging_dir = state['staging_dir']
        rule_segment_size = state['rule_segment_size']
        doc_segment_size = state['doc_segment_size']
        receipts_path = os.path.join(staging_dir, 'receipts.jsonl')

        # Discard receipts that were spilled after the last confirmed
        # boundary (interrupted between spilling and checkpointing).
        _truncate_receipts(receipts_path, state['receipt_lines'])
    else:
        os.makedirs(staging_dir, exist_ok=True)
        receipts_path = os.path.join(staging_dir, 'receipts.jsonl')

        # Materialize the one-shot rule source into staged segments with
        # stable numbers before touching any documents.
        rule_count, rule_order = _stage_rules(rules, staging_dir,
                                              rule_segment_size)

        # Snapshot the original document order once; the whole run (and a
        # potential resume) iterates rows in exactly this order.
        doc_ids = [
            table.document_id_class(doc_id)
            for doc_id in table._read_table().keys()
        ]

        state = {
            'version': _STATE_VERSION,
            'staging_dir': staging_dir,
            'rule_count': rule_count,
            'rule_order': rule_order,
            'doc_ids': doc_ids,
            'rule_segment_size': rule_segment_size,
            'doc_segment_size': doc_segment_size,
            # Last confirmed (doc segment, rule segment) cell, row-major
            'confirmed': None,
            # Number of confirmed lines in the receipts spill file
            'receipt_lines': 0,
        }
        _write_checkpoint(state_path, state)

    doc_ids = state['doc_ids']
    rule_count = state['rule_count']
    rule_segment_count = (rule_count + rule_segment_size - 1) \
        // rule_segment_size
    doc_segments = [
        doc_ids[i:i + doc_segment_size]
        for i in range(0, len(doc_ids), doc_segment_size)
    ]
    confirmed = tuple(state['confirmed']) if state['confirmed'] else None

    try:
        for doc_segment_index, segment_doc_ids in enumerate(doc_segments):
            for rule_segment_index in range(rule_segment_count):
                cell = (doc_segment_index, rule_segment_index)
                if confirmed is not None and cell <= confirmed:
                    # Already confirmed by a previous, interrupted run
                    continue

                # Load one segment of rules and apply it to one segment of
                # documents, then persist before confirming the boundary
                rule_segment = _load_rule_segment(staging_dir,
                                                  rule_segment_index)

                hits: list = []

                def updater(
                    table_data: dict,
                    _ids=segment_doc_ids,
                    _rules=rule_segment,
                ):
                    for doc_id in _ids:
                        if doc_id not in table_data:
                            # Row vanished since the snapshot was taken
                            continue
                        doc = table_data[doc_id]
                        # Rules run in source order on the live row, so a
                        # later rule's condition sees values written by
                        # earlier rules on this same row
                        for number, cond, fields in _rules:
                            if cond(doc):
                                apply_field_ops(doc, fields)
                                hits.append((number, doc_id))

                table._update_table(updater)

                # Spill the cell's receipts, then confirm the boundary --
                # in this order, so a resume never loses confirmed
                # receipts and never double-counts them
                state['receipt_lines'] += _append_receipts(receipts_path,
                                                           hits)
                state['confirmed'] = list(cell)
                _write_checkpoint(state_path, state)
                confirmed = cell

                if on_checkpoint is not None:
                    on_checkpoint(doc_segment_index, rule_segment_index)

        receipts = _aggregate_receipts(receipts_path, state['rule_order'])
    except BaseException:
        # The checkpoint on disk reflects the last confirmed row/rule
        # boundary; a later call with the same ``state_path`` resumes
        # from there.
        if not resumable:
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    # The run completed: clean up the staging area and the checkpoint
    shutil.rmtree(staging_dir, ignore_errors=True)
    if resumable and os.path.exists(state_path):
        os.remove(state_path)

    return receipts
