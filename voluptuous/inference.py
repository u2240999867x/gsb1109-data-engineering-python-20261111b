"""Contract-oriented schema inference from accepted and rejected examples.

The classic :meth:`Schema.infer` builds a schema by taking the union of every
observed column, which can admit value combinations that never occur together
(for example a money row carrying a count row's unit).  The helpers in this
module instead search a finite set of candidate discriminator columns for the
contract with the fewest branches that still accepts every positive example
and rejects every negative example, using only the observed key presence and
JSON value types at the leaves.
"""

from __future__ import annotations

import itertools

from voluptuous import error as er
from voluptuous.schema_builder import Optional, Required, Schema


# JSON scalar type name -> voluptuous scalar node.
class _JsonNumber:
    """Validator for the JSON ``number`` type (``int``/``float``, no ``bool``)."""

    def __call__(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise er.TypeInvalid('expected number')
        return value

    def __repr__(self):
        return 'number'

    def __eq__(self, other):
        return isinstance(other, _JsonNumber)

    def __hash__(self):
        return hash(_JsonNumber)


JSON_NUMBER = _JsonNumber()

_SCALAR_NODES = {
    'null': type(None),
    'boolean': bool,
    'number': JSON_NUMBER,
    'string': str,
}


class ContractCandidate:
    """One minimum-branch contract found during inference."""

    __slots__ = ('discriminators', 'schema', 'branches')

    def __init__(self, discriminators, schema, branches):
        self.discriminators = tuple(discriminators)
        self.schema = schema
        self.branches = branches

    def __repr__(self):
        return 'ContractCandidate(discriminators=%r, branches=%d)' % (
            self.discriminators,
            self.branches,
        )


class InferredContract:
    """Result of a contract inference run.

    ``candidates`` holds every contract with the minimum number of branches.
    When more than one distinct contract exists, ``separating_record`` is a
    record built only from observed values that exactly one of the contracts
    accepts, so a human can decide which contract is correct.
    """

    __slots__ = (
        'candidates',
        'separating_record',
        'separating_record_accepted_by',
        'separating_record_rejected_by',
    )

    def __init__(
        self,
        candidates,
        separating_record=None,
        separating_record_accepted_by=None,
        separating_record_rejected_by=None,
    ):
        self.candidates = list(candidates)
        self.separating_record = separating_record
        self.separating_record_accepted_by = separating_record_accepted_by
        self.separating_record_rejected_by = separating_record_rejected_by

    @property
    def schema(self) -> Schema:
        return self.candidates[0].schema

    @property
    def branches(self) -> int:
        return self.candidates[0].branches

    @property
    def discriminators(self):
        return self.candidates[0].discriminators

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1

    def __repr__(self):
        return (
            'InferredContract(branches=%d, discriminators=%r, ambiguous=%r)'
            % (self.branches, self.discriminators, self.ambiguous)
        )


def _json_type(value):
    """Return the JSON type name of ``value`` (booleans are not numbers)."""
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'boolean'
    if isinstance(value, (int, float)):
        return 'number'
    if isinstance(value, str):
        return 'string'
    if isinstance(value, list):
        return 'array'
    if isinstance(value, dict):
        return 'object'
    return type(value).__name__


def _fingerprint(value):
    """Structural presence/type fingerprint used to spot contradictions."""
    kind = _json_type(value)
    if kind == 'object':
        return (
            'object',
            frozenset((key, _fingerprint(sub)) for key, sub in value.items()),
        )
    if kind == 'array':
        return (
            'array',
            tuple(
                sorted(
                    (_fingerprint(item) for item in value),
                    key=repr,
                )
            ),
        )
    return (kind,)


def _canonical(node):
    """Deterministic, comparable representation of a built schema node."""
    from voluptuous.validators import Any, Union
    from voluptuous.schema_builder import Marker

    if isinstance(node, dict):
        return (
            'object',
            frozenset(
                (_canonical(key), _canonical(value))
                for key, value in node.items()
            ),
        )
    if isinstance(node, list):
        return ('array', tuple(_canonical(item) for item in node))
    if isinstance(node, (Any, Union)):
        kind = 'union' if isinstance(node, Union) else 'any'
        return (
            kind,
            tuple(
                sorted(
                    (_canonical(item) for item in node.validators),
                    key=repr,
                )
            ),
        )
    if isinstance(node, Marker):
        return (type(node).__name__, _canonical(node.schema))
    if isinstance(node, _JsonNumber):
        return ('number',)
    if isinstance(node, type):
        return (node.__name__,)
    if callable(node):
        return (getattr(node, '__name__', repr(node)),)
    return ('literal', node)


def _union(nodes):
    """Combine leaf nodes, collapsing equal nodes and single alternatives."""
    from voluptuous.validators import Any

    unique = []
    seen = set()
    for node in nodes:
        key = _canonical(node)
        if key not in seen:
            seen.add(key)
            unique.append(node)
    if len(unique) == 1:
        return unique[0]
    unique.sort(key=lambda node: repr(_canonical(node)))
    return Any(*unique)


def _array_leaf(values):
    """Build an array node, keeping element types unknown for empty arrays."""
    elements = []
    saw_elements = False
    for value in values:
        if value:
            saw_elements = True
            elements.extend(value)
    if not saw_elements:
        return list
    return [_leaf_from_values(elements)]


def _dict_leaf(rows):
    """Merge observed dict rows into one branch dict node.

    Keys present in every row of the branch are marked :class:`Required`;
    keys only some rows carry are marked :class:`Optional`.
    """
    keys = []
    values_by_key = {}
    for row in rows:
        for key, value in row.items():
            if key not in values_by_key:
                values_by_key[key] = []
                keys.append(key)
            values_by_key[key].append(value)

    node = {}
    for key in keys:
        values = values_by_key[key]
        value_node = _leaf_from_values(values)
        present_in_all = all(key in row for row in rows)
        if present_in_all:
            node[Required(key)] = value_node
        else:
            node[Optional(key)] = value_node
    return node


def _leaf_from_values(values):
    """Build a leaf node accepting every observed value at one slot."""
    scalar_nodes = []
    array_values = []
    dict_values = []
    for value in values:
        kind = _json_type(value)
        if kind == 'array':
            array_values.append(value)
        elif kind == 'object':
            dict_values.append(value)
        else:
            scalar_nodes.append(_SCALAR_NODES[kind])

    nodes = list(scalar_nodes)
    if array_values:
        nodes.append(_array_leaf(array_values))
    if dict_values:
        nodes.append(_dict_leaf(dict_values))
    return _union(nodes)


def _discriminator_value(row, column):
    """Hashable partition key for one discriminator column."""
    if not isinstance(row, dict) or column not in row:
        return None
    value = row[column]
    if isinstance(value, (dict, list)):
        return _fingerprint(value)
    return value


def _signature(row, discriminators):
    return tuple(
        (column, _discriminator_value(row, column)) for column in discriminators
    )


def _build_node(positives, discriminators):
    """Build a (possibly branched) schema node for a discriminator subset.

    Multiple partitions are joined with :class:`Union` routed by the actual
    discriminator values, so a record is only ever checked against its own
    branch instead of being allowed to match any branch's shape.
    """
    from voluptuous.validators import Union

    if not all(isinstance(row, dict) for row in positives):
        return 1, _leaf_from_values(positives)

    groups = {}
    order = []
    for row in positives:
        signature = _signature(row, discriminators)
        if signature not in groups:
            groups[signature] = []
            order.append(signature)
        groups[signature].append(row)

    branch_nodes = [_dict_leaf(groups[signature]) for signature in order]
    if len(branch_nodes) == 1:
        return 1, branch_nodes[0]

    signature_by_id = {
        id(node): signature for node, signature in zip(branch_nodes, order)
    }

    def route(value, validators):
        if not isinstance(value, dict):
            return []
        target = _signature(value, discriminators)
        return [
            node for node in validators if signature_by_id.get(id(node)) == target
        ]

    return len(branch_nodes), Union(*branch_nodes, discriminant=route)


def _accepts(schema, value):
    try:
        schema(value)
    except er.Invalid:
        return False
    return True


def _find_contradictions(positives, negatives, discriminators):
    """Positive/negative pairs no admissible leaf contract can separate.

    Leaves may only use key presence (required/optional) and JSON types, never
    literal values.  A negative row is only unseparatable when it falls in the
    same finest discriminator partition (matching discriminator values) as a
    positive row and has the identical presence/type fingerprint, because then
    that partition's branch leaf cannot tell the two rows apart.  Rows in
    different partitions are already separated by branching.
    """
    positive_keys = [
        (_signature(row, discriminators), _fingerprint(row)) for row in positives
    ]
    contradictions = []
    for neg_index, negative in enumerate(negatives, start=1):
        neg_key = (_signature(negative, discriminators), _fingerprint(negative))
        for pos_index, pos_key in enumerate(positive_keys, start=1):
            if neg_key == pos_key:
                contradictions.append((pos_index, neg_index))
    return contradictions


def _observed_samples(positives):
    """Map each observed key to one sample value per observed JSON type."""
    samples = {}
    for row in positives:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            by_type = samples.setdefault(key, {})
            kind = _json_type(value)
            if kind not in by_type:
                by_type[kind] = value
    return samples


def _separating_record(left, right, positives):
    """Synthesize a record accepted by exactly one of two candidate schemas.

    Returns ``(record, accepted_side)`` where ``accepted_side`` is 0 for the
    left schema and 1 for the right schema, or ``None`` if no observed-value
    mutation distinguishes them.
    """
    if not all(isinstance(row, dict) for row in positives):
        return None

    samples = _observed_samples(positives)

    def verdict_differs(record):
        left_ok = _accepts(left, record)
        right_ok = _accepts(right, record)
        if left_ok == right_ok:
            return None
        return record, (0 if left_ok else 1)

    for base in positives:
        record = dict(base)

        for key in list(record):
            narrowed = dict(record)
            del narrowed[key]
            diff = verdict_differs(narrowed)
            if diff is not None:
                return diff

        for key, by_type in samples.items():
            if key in record:
                for kind, value in by_type.items():
                    if kind == _json_type(record[key]):
                        continue
                    changed = dict(record)
                    changed[key] = value
                    diff = verdict_differs(changed)
                    if diff is not None:
                        return diff
            else:
                for value in by_type.values():
                    widened = dict(record)
                    widened[key] = value
                    diff = verdict_differs(widened)
                    if diff is not None:
                        return diff

    return None


def infer_contract(positives, negatives, discriminators, schema_cls=Schema, **kwargs):
    """Infer the minimum-branch contract separating positives from negatives.

    :param positives: records that the inferred schema must accept.
    :param negatives: records that the inferred schema must reject.
    :param discriminators: ordered candidate column names allowed to define
        branches; every subset is tried, starting with the empty set.
    :param schema_cls: schema class to instantiate.
    :returns: :class:`InferredContract`.
    :raises er.ContradictoryExamplesError: if a negative row has the same
        observed presence/type fingerprint as a positive row.
    """
    positives = list(positives)
    negatives = list(negatives)
    discriminators = tuple(sorted(discriminators))

    if not positives:
        raise er.SchemaError('at least one positive example is required')

    contradictions = _find_contradictions(positives, negatives, discriminators)
    if contradictions:
        pairs = ', '.join(
            'positive #%d / negative #%d' % pair for pair in contradictions
        )
        raise er.ContradictoryExamplesError(
            'indistinguishable examples using only observed keys and JSON '
            'types: %s' % pairs,
            contradictions=contradictions,
        )

    # Extra keys are a core rejection mechanism, so contract schemas always
    # prevent extras regardless of any caller-provided ``extra`` option.
    kwargs['extra'] = 0

    # Build every subset once, ordered by fewest branches, then fewest columns.
    candidates = []
    for size in range(len(discriminators) + 1):
        for subset in itertools.combinations(discriminators, size):
            branch_count, node = _build_node(positives, subset)
            candidates.append((branch_count, size, subset, node))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    feasible = []
    best_branch_count = None
    for branch_count, _size, subset, node in candidates:
        if best_branch_count is not None and branch_count > best_branch_count:
            break
        schema = schema_cls(node, **kwargs)
        if not all(_accepts(schema, row) for row in positives):
            continue
        if not all(not _accepts(schema, row) for row in negatives):
            continue
        best_branch_count = branch_count
        feasible.append((subset, schema, branch_count, _canonical(node)))

    if not feasible:
        raise er.SchemaError(
            'no contract over the candidate discriminator columns separates '
            'the positives from the negatives'
        )

    unique = []
    seen = set()
    for subset, schema, branch_count, canonical in feasible:
        if canonical in seen:
            continue
        seen.add(canonical)
        unique.append(ContractCandidate(subset, schema, branch_count))

    for other_index, other in enumerate(unique[1:], start=1):
        separating = _separating_record(
            unique[0].schema, other.schema, positives
        )
        if separating is not None:
            record, accepted_side = separating
            if accepted_side == 0:
                accepted_index, rejected_index = 0, other_index
            else:
                accepted_index, rejected_index = other_index, 0
            return InferredContract(
                unique,
                separating_record=record,
                separating_record_accepted_by=accepted_index,
                separating_record_rejected_by=rejected_index,
            )

    return InferredContract(unique)
