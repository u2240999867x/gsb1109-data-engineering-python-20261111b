"""
Serializable query descriptions with contract-checked equivalence.

Filtering rules need to travel between processes together with partition
manifests, so this module provides a saveable (JSON-able) description
format for a decidable fragment of TinyDB queries:

* comparisons (``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``),
* boolean combinations (``and`` / ``or`` / ``not``),
* ``any`` / ``all`` over list fields against a finite set of scalars,
* nested fields, including keys that literally contain a dot.

A :class:`Catalog` declares the contract of the stored documents: finite
value domains, list length bounds, and whether fields may be missing, may
be ``None`` or may be empty lists.  Given two descriptions and a catalog,
:func:`compare_descriptions` decides whether both rules select the same
documents for every legal document.  If they differ, a minimal legal
document is returned as a difference witness; if they are equivalent, old
partition manifests can be reused after a description upgrade.  Rules that
only touch a subset of the contracted fields are still decided --
``unknown`` is only reported when a rule escapes the contract (undeclared
field) or the bounded document space exceeds ``max_documents``.

Queries that rely on plain Python test functions cannot be serialized;
:func:`describe_query` raises :class:`QueryNotSerializableError` pointing
at the source location of the offending function.
"""

import inspect
import json
import os
from collections.abc import Mapping
from heapq import heappop, heappush
from itertools import combinations, count, product
from math import comb, isfinite

from .queries import Query, QueryInstance
from .utils import FrozenDict

__all__ = (
    'DESCRIPTION_VERSION',
    'SUPPORTED_VERSIONS',
    'QueryDescriptionError',
    'UnsupportedQueryError',
    'QueryNotSerializableError',
    'ContractViolationError',
    'ScalarField',
    'ListField',
    'ObjectField',
    'Catalog',
    'RuleComparison',
    'describe_query',
    'compile_query',
    'validate_description',
    'compare_descriptions',
    'check_upgrade',
    'description_to_json',
    'description_from_json',
)

#: The description format version emitted by :func:`describe_query`.
DESCRIPTION_VERSION = 1

#: Description format versions understood by this module.
SUPPORTED_VERSIONS = (1,)

_COMPARISONS = ('==', '!=', '<', '<=', '>', '>=')
_ORDERING_COMPARISONS = ('<', '<=', '>', '>=')
_MEMBERSHIP_OPS = frozenset(('any', 'all', 'exists'))

#: Sentinel marking an absent field during document enumeration.
_ABSENT = object()


# --- Errors ------------------------------------------------------------------


class QueryDescriptionError(ValueError):
    """Base class for query description errors."""


class UnsupportedQueryError(QueryDescriptionError):
    """The query uses an operation outside the serializable fragment."""


class QueryNotSerializableError(QueryDescriptionError):
    """
    The query relies on a Python callable and cannot be serialized.

    ``func`` is the offending callable (if known) and ``location`` a
    ``"file:line"`` string pointing at its definition (if determinable).
    """

    def __init__(self, message, *, func=None, location=None):
        super().__init__(message)
        self.func = func
        self.location = location


class ContractViolationError(ValueError):
    """A document does not satisfy the catalog contract."""


# --- Small helpers -----------------------------------------------------------


def _sort_key(value):
    return (type(value).__name__, repr(value))


def _is_json_scalar(value):
    if value is None or isinstance(value, (bool, str, int)):
        return True
    if isinstance(value, float):
        return isfinite(value)
    return False


def _json_value_problem(value):
    if _is_json_scalar(value):
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            problem = _json_value_problem(item)
            if problem is not None:
                return problem
        return None
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                return 'mapping keys must be strings, got {!r}'.format(key)
            problem = _json_value_problem(item)
            if problem is not None:
                return problem
        return None
    return 'value {!r} is not JSON-serializable'.format(value)


def _thaw(value):
    if isinstance(value, FrozenDict):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    if isinstance(value, frozenset):
        return sorted((_thaw(item) for item in value), key=repr)
    return value


def _source_location(func):
    try:
        file = inspect.getsourcefile(func)
        _, line = inspect.getsourcelines(func)
    except (OSError, TypeError):
        return None
    if file is None:
        return None
    return '{}:{}'.format(file, line)


def _query_location(query):
    # Best effort origin for queries without a structured hash (e.g. built
    # via Query.map or a hand-written QueryInstance).  Locations inside
    # TinyDB itself are not useful to the caller, so they are dropped.
    test = getattr(query, '_test', None)
    if test is None:
        return None
    location = _source_location(test)
    if location is None:
        return None
    file = location.rsplit(':', 1)[0]
    if os.path.dirname(file) == os.path.dirname(os.path.abspath(__file__)):
        return None
    return location


def _dotted(path, key=None):
    parts = list(path)
    if key is not None:
        parts.append(key)
    return '.'.join(parts)


# --- Query -> description ----------------------------------------------------


def describe_query(query, *, version=DESCRIPTION_VERSION):
    """
    Convert a TinyDB query into a saveable description.

    The result is a JSON-able dict ``{'version': ..., 'root': ...}`` that
    can be shipped across processes together with a partition manifest and
    turned back into a query with :func:`compile_query`.

    :raises QueryNotSerializableError: the query uses a Python test
        function or is otherwise not cacheable; the error points at the
        source location of the offending callable when known.
    :raises UnsupportedQueryError: the query uses an operation outside the
        serializable fragment (regexes, fragments, ``any``/``all`` with a
        nested query, non-scalar sets).
    """
    if version not in SUPPORTED_VERSIONS:
        raise ValueError('unsupported description version: {!r}'
                         .format(version))
    if not isinstance(query, QueryInstance):
        raise TypeError('expected a QueryInstance, got {!r}'.format(query))
    return {'version': version, 'root': _describe_node(query._hash, query)}


def _describe_node(hashval, query):
    if hashval is None:
        location = _query_location(query)
        message = ('query cannot be serialized: it is not cacheable (built '
                   'with Query.map() or a custom QueryInstance); only '
                   'comparisons, boolean combinations, any/all over finite '
                   'scalar sets, exists and one_of are serializable')
        if location is not None:
            message += '; query test defined at {}'.format(location)
        raise QueryNotSerializableError(message, location=location)

    if hashval == ():
        return {'op': 'const', 'value': True}

    if hashval == (None,):
        raise QueryDescriptionError(
            'empty query (no field selected) cannot be serialized')

    op = hashval[0]

    if op == 'path':
        raise QueryDescriptionError(
            'query selects a field but performs no test on it')

    if op in _COMPARISONS:
        _, path, value = hashval
        return {
            'op': 'cmp',
            'path': _describe_path(path),
            'cmp': op,
            'value': _describe_comparison_value(op, value),
        }

    if op in ('and', 'or'):
        children = [_describe_node(child, query) for child in hashval[1]]
        flat = []
        for child in children:
            if child['op'] == op:
                flat.extend(child['args'])
            else:
                flat.append(child)
        # Deduplicate and sort so that logically identical queries produce
        # byte-identical descriptions regardless of frozenset ordering.
        unique = {}
        for child in flat:
            unique[json.dumps(child, sort_keys=True)] = child
        ordered = [unique[key] for key in sorted(unique)]
        if len(ordered) == 1:
            return ordered[0]
        return {'op': op, 'args': ordered}

    if op == 'not':
        arg = _describe_node(hashval[1], query)
        if arg == {'op': 'const', 'value': True}:
            return {'op': 'const', 'value': False}
        return {'op': 'not', 'arg': arg}

    if op == 'exists':
        return {'op': 'exists', 'path': _describe_path(hashval[1])}

    if op in ('any', 'all'):
        _, path, cond = hashval
        if isinstance(cond, QueryInstance) or callable(cond):
            raise UnsupportedQueryError(
                '{}() with a nested query condition is not serializable; '
                'use a finite list of scalars instead'.format(op))
        values = _thaw(cond)
        if not isinstance(values, list):
            raise UnsupportedQueryError(
                '{}() condition must be a list of scalars, got {!r}'
                .format(op, values))
        for value in values:
            if not _is_json_scalar(value):
                raise UnsupportedQueryError(
                    '{}() only supports finite sets of scalars, got {!r}'
                    .format(op, value))
        return {'op': op, 'path': _describe_path(path), 'values': values}

    if op == 'one_of':
        _, path, items = hashval
        values = _thaw(items)
        if not isinstance(values, list):
            raise UnsupportedQueryError(
                'one_of() items must be a list of scalars, got {!r}'
                .format(values))
        for value in values:
            if not _is_json_scalar(value):
                raise UnsupportedQueryError(
                    'one_of() only supports finite sets of scalars, got {!r}'
                    .format(value))
        return {'op': 'one_of', 'path': _describe_path(path),
                'values': values}

    if op == 'test':
        _, path, func, _args = hashval
        name = getattr(func, '__qualname__', None) or repr(func)
        location = _source_location(func)
        message = ('query uses Python test function {} on field {!r} which '
                   'cannot be serialized; re-express it with comparisons, '
                   'boolean operators, any/all, exists or one_of'
                   .format(name, list(path)))
        if location is not None:
            message += '; function defined at {}'.format(location)
        raise QueryNotSerializableError(message, func=func,
                                        location=location)

    if op in ('matches', 'search'):
        raise UnsupportedQueryError(
            'regular expression queries ({}) are not part of the '
            'serializable fragment'.format(op))

    if op == 'fragment':
        raise UnsupportedQueryError(
            'fragment() queries are not part of the serializable fragment')

    raise UnsupportedQueryError(
        'unsupported query structure: {!r}'.format(hashval))


def _describe_path(path):
    parts = list(path)
    for part in parts:
        if not isinstance(part, str):
            raise QueryNotSerializableError(
                'query path contains a non-string element {!r} (e.g. added '
                'via Query.map()) and cannot be serialized'.format(part))
    return parts


def _describe_comparison_value(op, value):
    thawed = _thaw(value)
    if op in _ORDERING_COMPARISONS:
        if not _is_json_scalar(thawed) or thawed is None:
            raise UnsupportedQueryError(
                'ordering comparison {} requires a JSON scalar other than '
                'None, got {!r}'.format(op, thawed))
        return thawed
    problem = _json_value_problem(thawed)
    if problem is not None:
        raise UnsupportedQueryError(
            'comparison value cannot be serialized: {}'.format(problem))
    return thawed


# --- Description validation and compilation ----------------------------------


def validate_description(description):
    """
    Check that a description is well-formed and supported.

    :raises QueryDescriptionError: if the description is malformed or uses
        an unsupported format version.
    """
    if not isinstance(description, Mapping):
        raise QueryDescriptionError(
            'description must be a mapping, got {!r}'.format(description))
    version = description.get('version')
    if version not in SUPPORTED_VERSIONS:
        raise QueryDescriptionError(
            'unsupported description version: {!r}'.format(version))
    if 'root' not in description:
        raise QueryDescriptionError('description has no "root" node')
    _validate_node(description['root'])


def _require_key(node, key):
    if key not in node:
        raise QueryDescriptionError(
            'description node {!r} is missing key {!r}'.format(node, key))
    return node[key]


def _validate_node(node):
    if not isinstance(node, Mapping) or 'op' not in node:
        raise QueryDescriptionError(
            'invalid description node: {!r}'.format(node))
    op = node['op']

    if op == 'const':
        if not isinstance(_require_key(node, 'value'), bool):
            raise QueryDescriptionError(
                'const node value must be a boolean: {!r}'.format(node))
        return

    if op in ('and', 'or'):
        args = _require_key(node, 'args')
        if not isinstance(args, list):
            raise QueryDescriptionError(
                '{} node args must be a list: {!r}'.format(op, node))
        for arg in args:
            _validate_node(arg)
        return

    if op == 'not':
        _validate_node(_require_key(node, 'arg'))
        return

    if op in ('cmp', 'any', 'all', 'exists', 'one_of'):
        _validate_path(_require_key(node, 'path'))
        if op == 'cmp':
            cmp_op = _require_key(node, 'cmp')
            if cmp_op not in _COMPARISONS:
                raise QueryDescriptionError(
                    'unknown comparison operator: {!r}'.format(cmp_op))
            value = _require_key(node, 'value')
            if cmp_op in _ORDERING_COMPARISONS:
                if not _is_json_scalar(value) or value is None:
                    raise QueryDescriptionError(
                        'ordering comparison {} requires a JSON scalar '
                        'other than None, got {!r}'.format(cmp_op, value))
            else:
                problem = _json_value_problem(value)
                if problem is not None:
                    raise QueryDescriptionError(
                        'comparison value is not serializable: {}'
                        .format(problem))
        elif op in ('any', 'all', 'one_of'):
            values = _require_key(node, 'values')
            if not isinstance(values, list):
                raise QueryDescriptionError(
                    '{} node values must be a list: {!r}'.format(op, node))
            for value in values:
                if not _is_json_scalar(value):
                    raise QueryDescriptionError(
                        '{} node values must be JSON scalars, got {!r}'
                        .format(op, value))
        return

    raise QueryDescriptionError(
        'unknown description operation: {!r}'.format(op))


def _validate_path(path):
    if not isinstance(path, list) or not path:
        raise QueryDescriptionError(
            'node path must be a non-empty list of keys: {!r}'.format(path))
    for part in path:
        if not isinstance(part, str):
            raise QueryDescriptionError(
                'node path elements must be strings: {!r}'.format(path))


def compile_query(description):
    """
    Turn a query description back into an executable TinyDB query.

    The description is validated first, so descriptions received from
    another process can be compiled safely.
    """
    validate_description(description)
    return _compile_node(description['root'])


def _compile_node(node):
    op = node['op']

    if op == 'const':
        query = Query().noop()
        return query if node['value'] else ~query

    if op in ('and', 'or'):
        children = [_compile_node(arg) for arg in node['args']]
        if not children:
            empty = Query().noop()
            return empty if op == 'and' else ~empty
        result = children[0]
        for child in children[1:]:
            result = result & child if op == 'and' else result | child
        return result

    if op == 'not':
        return ~_compile_node(node['arg'])

    query = Query()
    for part in node['path']:
        query = query[part]

    if op == 'cmp':
        cmp_op = node['cmp']
        value = node['value']
        if cmp_op == '==':
            return query == value
        if cmp_op == '!=':
            return query != value
        if cmp_op == '<':
            return query < value
        if cmp_op == '<=':
            return query <= value
        if cmp_op == '>':
            return query > value
        return query >= value

    if op == 'any':
        return query.any(list(node['values']))

    if op == 'all':
        return query.all(list(node['values']))

    if op == 'exists':
        return query.exists()

    # op == 'one_of'
    return query.one_of(list(node['values']))


def description_to_json(description):
    """Serialize a description to a JSON string for cross-process delivery."""
    validate_description(description)
    return json.dumps(description, sort_keys=True)


def description_from_json(payload):
    """Parse and validate a description received as a JSON string."""
    description = json.loads(payload)
    validate_description(description)
    return description


# --- Catalog: field contracts ------------------------------------------------


def _check_domain_values(values, what):
    checked = []
    for value in values:
        if value is None:
            raise ValueError(
                '{} must not contain None; use nullable=True to allow '
                'null values'.format(what))
        if not _is_json_scalar(value):
            raise ValueError(
                '{} must contain only JSON scalars, got {!r}'
                .format(what, value))
        checked.append(value)
    return checked


def _check_fields(fields):
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError(
            'fields must be a non-empty mapping of field name to contract')
    checked = {}
    for name, contract in fields.items():
        if not isinstance(name, str):
            raise ValueError(
                'field names must be strings, got {!r}'.format(name))
        if not isinstance(contract, (ScalarField, ListField, ObjectField)):
            raise ValueError(
                'invalid contract for field {!r}: {!r}'
                .format(name, contract))
        checked[name] = contract
    return checked


class ScalarField:
    """
    Contract for a scalar field with a finite value domain.

    :param domain: the finite set of legal values (JSON scalars).
    :param optional: whether the field may be missing from a document.
    :param nullable: whether the field may be ``None``.
    """

    def __init__(self, domain, *, optional=False, nullable=False):
        values = _check_domain_values(domain, 'domain')
        self.domain = tuple(sorted(set(values), key=_sort_key))
        self.optional = bool(optional)
        self.nullable = bool(nullable)
        if not self.domain and not self.optional and not self.nullable:
            raise ValueError(
                'scalar field has an empty domain but is neither optional '
                'nor nullable: no legal value exists')

    def __repr__(self):
        return ('ScalarField({!r}, optional={!r}, nullable={!r})'
                .format(list(self.domain), self.optional, self.nullable))


class ListField:
    """
    Contract for a list field over a finite set of scalar elements.

    :param elements: the finite set of legal list elements (JSON scalars).
    :param max_length: the upper bound on the list length.
    :param allow_empty: whether the empty list is a legal value.
    :param optional: whether the field may be missing from a document.
    :param nullable: whether the field may be ``None``.
    """

    def __init__(self, elements, *, max_length, allow_empty=True,
                 optional=False, nullable=False):
        if (not isinstance(max_length, int) or isinstance(max_length, bool)
                or max_length < 0):
            raise ValueError('max_length must be a non-negative integer')
        values = _check_domain_values(elements, 'elements')
        self.elements = tuple(sorted(set(values), key=_sort_key))
        self.max_length = max_length
        self.allow_empty = bool(allow_empty)
        self.optional = bool(optional)
        self.nullable = bool(nullable)
        has_legal_list = (self.allow_empty
                          or (self.elements and self.max_length > 0))
        if not has_legal_list and not self.optional and not self.nullable:
            raise ValueError(
                'list field allows no legal list but is neither optional '
                'nor nullable: no legal value exists')

    def __repr__(self):
        return ('ListField({!r}, max_length={!r}, allow_empty={!r}, '
                'optional={!r}, nullable={!r})'
                .format(list(self.elements), self.max_length,
                        self.allow_empty, self.optional, self.nullable))


class ObjectField:
    """
    Contract for a nested object field.

    Field names are matched literally, so a name containing a dot (e.g.
    ``"a.b"``) refers to a single key, not to a nested path.

    :param fields: mapping of field name to field contract.
    :param optional: whether the field may be missing from a document.
    """

    def __init__(self, fields, *, optional=False):
        self.fields = _check_fields(fields)
        self.optional = bool(optional)

    def __repr__(self):
        return 'ObjectField({!r}, optional={!r})'.format(self.fields,
                                                         self.optional)


class Catalog:
    """
    The contract of the stored documents.

    Declares every legal field together with its finite domain and list
    length bounds, so that the space of legal documents is bounded and
    rule equivalence becomes decidable.
    """

    def __init__(self, fields):
        self.fields = _check_fields(fields)

    def resolve(self, path):
        """
        Return the contract for a field path.

        Path segments are matched literally against declared field names.

        :raises KeyError: if the path is not declared in the catalog.
        """
        fields = self.fields
        contract = None
        for part in path:
            if fields is None or part not in fields:
                raise KeyError(tuple(path))
            contract = fields[part]
            fields = (contract.fields
                      if isinstance(contract, ObjectField) else None)
        return contract

    def validate(self, document):
        """
        Check a document against the contract.

        :raises ContractViolationError: if the document is not legal.
        """
        if not isinstance(document, Mapping):
            raise ContractViolationError(
                'document must be a mapping, got {!r}'.format(document))
        _validate_document(self.fields, document, ())

    def is_legal(self, document):
        """Return whether a document satisfies the contract."""
        try:
            self.validate(document)
        except ContractViolationError:
            return False
        return True


def _validate_document(fields, document, path):
    for key in document:
        if key not in fields:
            raise ContractViolationError(
                'undeclared field {}'.format(_dotted(path, key)))
    for name, contract in fields.items():
        if name not in document:
            if not contract.optional:
                raise ContractViolationError(
                    'required field {} is missing'
                    .format(_dotted(path, name)))
            continue
        _validate_value(contract, document[name], path + (name,))


def _validate_value(contract, value, path):
    where = _dotted(path)
    if isinstance(contract, ScalarField):
        if value is None:
            if not contract.nullable:
                raise ContractViolationError(
                    'field {} may not be null'.format(where))
        elif value not in contract.domain:
            raise ContractViolationError(
                'field {} value {!r} is outside its domain'
                .format(where, value))
    elif isinstance(contract, ListField):
        if value is None:
            if not contract.nullable:
                raise ContractViolationError(
                    'field {} may not be null'.format(where))
            return
        if not isinstance(value, list):
            raise ContractViolationError(
                'field {} must be a list, got {!r}'.format(where, value))
        if len(value) > contract.max_length:
            raise ContractViolationError(
                'field {} exceeds its maximum length of {}'
                .format(where, contract.max_length))
        if not value and not contract.allow_empty:
            raise ContractViolationError(
                'field {} may not be an empty list'.format(where))
        for element in value:
            if element not in contract.elements:
                raise ContractViolationError(
                    'field {} element {!r} is outside its domain'
                    .format(where, element))
    else:  # ObjectField
        if not isinstance(value, Mapping):
            raise ContractViolationError(
                'field {} must be a mapping, got {!r}'.format(where, value))
        _validate_document(contract.fields, value, path)


# --- Equivalence checking ----------------------------------------------------


class RuleComparison:
    """
    The outcome of comparing two query descriptions under a catalog.

    ``status`` is one of:

    * ``'equivalent'`` -- both rules select the same documents for every
      legal document; manifests built with the old rule can be reused.
    * ``'different'`` -- the rules differ; ``witness`` is a minimal legal
      document on which they disagree.
    * ``'unknown'`` -- equivalence could not be decided; ``reason``
      explains why (undeclared field, document space too large, ...).
    """

    EQUIVALENT = 'equivalent'
    DIFFERENT = 'different'
    UNKNOWN = 'unknown'

    def __init__(self, status, *, witness=None, reason=None):
        if status not in (self.EQUIVALENT, self.DIFFERENT, self.UNKNOWN):
            raise ValueError('invalid comparison status: {!r}'.format(status))
        self.status = status
        self.witness = witness
        self.reason = reason

    @property
    def equivalent(self):
        """Whether both rules select the same documents."""
        return self.status == self.EQUIVALENT

    @property
    def reusable(self):
        """Whether manifests built with the old rule can be reused."""
        return self.status == self.EQUIVALENT

    def __repr__(self):
        if self.status == self.DIFFERENT:
            return 'RuleComparison(different, witness={!r})'.format(
                self.witness)
        if self.status == self.UNKNOWN:
            return 'RuleComparison(unknown, reason={!r})'.format(self.reason)
        return 'RuleComparison(equivalent)'


def compare_descriptions(old, new, catalog, *, max_documents=100000):
    """
    Decide whether two query descriptions select the same documents.

    Both descriptions are compiled and evaluated over every document that
    is legal under ``catalog``.  Only fields referenced by the rules take
    part in the enumeration, so rules over a subset of the contracted
    fields are still decided; unreferenced fields are fixed to their
    cheapest legal state.  Documents on which a rule raises ``TypeError``
    (e.g. an ordering comparison against a null value, which TinyDB does
    not catch) are treated as not selected by that rule.

    :param max_documents: upper bound on the bounded document space; if
        the space exceeds it, the result is ``unknown``.
    :return: a :class:`RuleComparison`.
    """
    if not isinstance(catalog, Catalog):
        raise TypeError('catalog must be a Catalog instance, got {!r}'
                        .format(catalog))
    old_query = compile_query(old)
    new_query = compile_query(new)

    referenced = (_collect_paths(old['root'])
                  | _collect_paths(new['root']))

    for path, _op in sorted(referenced):
        try:
            catalog.resolve(path)
        except KeyError:
            return RuleComparison(
                RuleComparison.UNKNOWN,
                reason='field {!r} is not declared in the catalog'
                       .format(list(path)))

    top_names = {path[0] for path, _op in referenced}
    varying = []
    total = 1
    for name in sorted(top_names):
        contract = catalog.fields[name]
        membership_only = False
        if isinstance(contract, ListField):
            ops = {op for path, op in referenced if path == (name,)}
            membership_only = ops <= _MEMBERSHIP_OPS
        total *= _count_states(contract, membership_only)
        if total > max_documents:
            return RuleComparison(
                RuleComparison.UNKNOWN,
                reason='document space exceeds {} combinations; refine the '
                       'catalog or raise max_documents'
                       .format(max_documents))
        varying.append((name, contract, membership_only))

    base = {}
    for name, contract in catalog.fields.items():
        if name in top_names:
            continue
        state = _cheapest_state(contract)
        if state is not _ABSENT:
            base[name] = state

    return _search_witness(old_query, new_query, varying, base)


def check_upgrade(old_description, new_description, catalog, **kwargs):
    """
    Decide whether old partition manifests survive a description upgrade.

    When upgrading filtering rules to a new description version, call this
    with the old and the new description: if the result is equivalent, the
    old manifests can be reused unchanged; if it is different, ``witness``
    is a minimal legal document on which the selection changed.
    """
    return compare_descriptions(old_description, new_description, catalog,
                                **kwargs)


def _collect_paths(node):
    found = set()

    def visit(current):
        op = current['op']
        if op in ('and', 'or'):
            for arg in current['args']:
                visit(arg)
        elif op == 'not':
            visit(current['arg'])
        elif op == 'const':
            return
        else:
            found.add((tuple(current['path']), op))

    visit(node)
    return found


def _count_states(contract, membership_only):
    total = 1 if contract.optional else 0
    if isinstance(contract, ScalarField):
        total += (1 if contract.nullable else 0) + len(contract.domain)
    elif isinstance(contract, ListField):
        if contract.nullable:
            total += 1
        total += _count_lists(contract, membership_only)
    else:  # ObjectField
        subtotal = 1
        for sub in contract.fields.values():
            subtotal *= _count_states(sub, False)
        total += subtotal
    return total


def _count_lists(contract, membership_only):
    total = 0
    for length in range(contract.max_length + 1):
        if length == 0:
            if contract.allow_empty:
                total += 1
        elif membership_only:
            total += comb(len(contract.elements), length)
        else:
            total += len(contract.elements) ** length
    return total


def _field_states(contract, membership_only):
    states = []
    if contract.optional:
        states.append(_ABSENT)
    if isinstance(contract, ScalarField):
        if contract.nullable:
            states.append(None)
        states.extend(contract.domain)
    elif isinstance(contract, ListField):
        if contract.nullable:
            states.append(None)
        states.extend(_list_states(contract, membership_only))
    else:  # ObjectField
        names = list(contract.fields)
        sub_states = [_field_states(contract.fields[name], False)
                      for name in names]
        documents = []
        for combo in product(*sub_states):
            documents.append({name: value
                              for name, value in zip(names, combo)
                              if value is not _ABSENT})
        documents.sort(key=_document_key)
        states.extend(documents)
    return states


def _list_states(contract, membership_only):
    elements = list(contract.elements)
    states = []
    for length in range(contract.max_length + 1):
        if length == 0:
            if contract.allow_empty:
                states.append([])
        elif membership_only:
            # any/all/exists only observe set membership, so duplicates and
            # orderings of the same subset cannot change the outcome.
            for combo in combinations(elements, length):
                states.append(list(combo))
        else:
            for combo in product(elements, repeat=length):
                states.append(list(combo))
    return states


def _cheapest_state(contract):
    if contract.optional:
        return _ABSENT
    if isinstance(contract, ScalarField):
        if contract.nullable:
            return None
        return contract.domain[0]
    if isinstance(contract, ListField):
        if contract.nullable:
            return None
        if contract.allow_empty:
            return []
        return [contract.elements[0]]
    # ObjectField
    return {name: state
            for name, sub in contract.fields.items()
            if (state := _cheapest_state(sub)) is not _ABSENT}


def _state_cost(state):
    if state is _ABSENT:
        return (0, 0)
    if isinstance(state, list):
        return (1, len(state))
    if isinstance(state, Mapping):
        present, list_length, _ = _document_key(state)
        return (present + 1, list_length)
    return (1, 0)


def _document_key(document):
    present = 0
    list_length = 0
    for value in document.values():
        present += 1
        if isinstance(value, list):
            list_length += len(value)
        elif isinstance(value, Mapping):
            sub_present, sub_list_length, _ = _document_key(value)
            present += sub_present
            list_length += sub_list_length
    return (present, list_length,
            json.dumps(document, sort_keys=True))


def _safe_match(query, document):
    try:
        return bool(query(document))
    except TypeError:
        # TinyDB propagates TypeError for e.g. ordering comparisons against
        # null values; for selection purposes such documents count as not
        # matched by the rule.
        return False


def _search_witness(old_query, new_query, varying, base):
    names = [name for name, _, _ in varying]
    states = [_field_states(contract, membership_only)
              for _, contract, membership_only in varying]
    costs = [[_state_cost(state) for state in field_states]
             for field_states in states]

    def cost_of(indices):
        present = 0
        list_length = 0
        for field_costs, index in zip(costs, indices):
            present += field_costs[index][0]
            list_length += field_costs[index][1]
        return (present, list_length)

    start = tuple(0 for _ in varying)
    serial = count()
    heap = [(cost_of(start), next(serial), start)]
    seen = {start}
    best = None
    best_cost = None

    # Best-first search over the bounded document space: documents are
    # visited in non-decreasing cost order, so the first disagreement found
    # at the cheapest cost level is a minimal legal witness.
    while heap:
        cost, _, indices = heappop(heap)
        if best_cost is not None and cost > best_cost:
            break
        document = dict(base)
        for name, field_states, index in zip(names, states, indices):
            value = field_states[index]
            if value is not _ABSENT:
                document[name] = value
        old_match = _safe_match(old_query, document)
        new_match = _safe_match(new_query, document)
        if old_match != new_match:
            if (best is None
                    or _document_key(document) < _document_key(best)):
                best = document
                best_cost = cost
        for position in range(len(indices)):
            if indices[position] + 1 < len(states[position]):
                neighbor = list(indices)
                neighbor[position] += 1
                neighbor = tuple(neighbor)
                if neighbor not in seen:
                    seen.add(neighbor)
                    heappush(heap,
                             (cost_of(neighbor), next(serial), neighbor))

    if best is not None:
        return RuleComparison(RuleComparison.DIFFERENT, witness=best)
    return RuleComparison(RuleComparison.EQUIVALENT)
