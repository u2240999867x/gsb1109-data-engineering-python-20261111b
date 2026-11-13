"""
Differential regression: explicit column getters vs. pydash shorthand iteratees.

A column mapping was migrated from explicit getter functions to pydash shorthand
iteratees: path strings for grouping, match dicts and ``[path, value]`` pairs for
filtering. Every row is read twice by the pipeline, once by :func:`pydash.filter_`
and once by :func:`pydash.group_by`, so each shorthand must replay cleanly on the
same row.

These tests compare the two implementations on more than return values: every row
is instrumented, so the exact source positions read and their order are checked
alongside the final grouped row ids. Inputs include literal dot keys, nested keys
that repeat the same name, and consuming fields that hand out the next queued
value on every read.

A small exhaustive space of paths and row shapes is swept, and a greedy shrinker
reduces each divergence class to a minimal witness:

- ``WRONG_COLUMN``: the shorthand reads different source positions than the getter.
- ``EXTRA_READ``: the shorthand reads the same positions but more times.
- ``MISSING_AS_EMPTY``: the shorthand turns a missing or exhausted column into an
  empty value instead of failing.
"""

from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import dataclass

import pytest

import pydash as _


parametrize = pytest.mark.parametrize

WRONG_COLUMN = "wrong-column"
EXTRA_READ = "extra-read"
MISSING_AS_EMPTY = "missing-as-empty"
DIVERGENCE_CLASSES = (WRONG_COLUMN, EXTRA_READ, MISSING_AS_EMPTY)

TARGET = 1


class ExhaustedError(Exception):
    """A consuming field was read more times than it has values."""


class TrackedDict(dict):
    """A dict that records every read as ``(row_id, key)`` in a shared log."""

    def __init__(self, data, log, row_id):
        super().__init__()
        self._log = log
        self.row_id = row_id
        for key, value in data.items():
            self[key] = wrap_value(value, log, row_id)

    def __getitem__(self, key):
        self._log.append((self.row_id, key))
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._log.append((self.row_id, key))
        return super().get(key, default)


def wrap_value(value, log, row_id):
    if isinstance(value, dict):
        return TrackedDict(value, log, row_id)
    return value


class ConsumingRow(Mapping):
    """A row whose fields hand out the next queued value on every read."""

    def __init__(self, queues, log, row_id):
        self._queues = {key: deque(values) for key, values in queues.items()}
        self._log = log
        self.row_id = row_id

    def __getitem__(self, key):
        self._log.append((self.row_id, key))
        if key not in self._queues:
            raise KeyError(key)
        if not self._queues[key]:
            raise ExhaustedError(key)
        return self._queues[key].popleft()

    def __iter__(self):
        return iter(self._queues)

    def __len__(self):
        return len(self._queues)


def tracked(data):
    return ("tracked", data)


def consuming(queues):
    return ("consuming", queues)


def build_rows(row_specs, log):
    rows = []
    for row_id, (kind, payload) in enumerate(row_specs):
        if kind == "tracked":
            rows.append(TrackedDict(payload, log, row_id))
        else:
            rows.append(ConsumingRow(payload, log, row_id))
    return rows


@dataclass(frozen=True)
class ColumnMapping:
    """One column mapping: the intended key sequence plus its shorthand forms."""

    keys: tuple
    target: object
    filter_form: object
    group_form: object


@dataclass
class Outcome:
    groups: object
    log: list
    error: object


def explicit_getter(keys):
    def getter(row):
        value = row
        for key in keys:
            value = value[key]
        return value

    return getter


def explicit_predicate(mapping):
    getter = explicit_getter(mapping.keys)

    def predicate(row):
        return getter(row) == mapping.target

    return predicate


def run_pipeline(row_specs, mapping, style):
    """Filter then group, returning grouped row ids and the read log."""
    log = []
    rows = build_rows(row_specs, log)
    if style == "explicit":
        predicate = explicit_predicate(mapping)
        getter = explicit_getter(mapping.keys)
    else:
        predicate = mapping.filter_form
        getter = mapping.group_form
    try:
        survivors = _.filter_(rows, predicate)
        grouped = _.group_by(survivors, getter)
    except Exception as exc:  # noqa: BLE001 - the divergence is the data
        return Outcome(None, log, exc)
    groups = {key: [row.row_id for row in members] for key, members in grouped.items()}
    return Outcome(groups, log, None)


def is_submultiset(smaller, bigger):
    remaining = Counter(bigger)
    for entry in smaller:
        if remaining[entry] <= 0:
            return False
        remaining[entry] -= 1
    return True


def classify(explicit, shorthand):
    """Classify the divergence between the two pipeline outcomes, if any."""
    divergence = None
    if explicit.error is None and shorthand.error is None:
        if explicit.groups != shorthand.groups:
            divergence = WRONG_COLUMN
        elif explicit.log != shorthand.log:
            if is_submultiset(explicit.log, shorthand.log):
                divergence = EXTRA_READ
            else:
                divergence = WRONG_COLUMN
    elif explicit.error is not None and shorthand.error is None:
        divergence = MISSING_AS_EMPTY
    elif explicit.error is None and shorthand.error is not None:
        # The shorthand over-read a consuming field and exhausted it.
        divergence = EXTRA_READ
    return divergence


def case_class(mapping, row_specs):
    explicit = run_pipeline(row_specs, mapping, "explicit")
    shorthand = run_pipeline(row_specs, mapping, "shorthand")
    return classify(explicit, shorthand)


#
# Exhaustive space of mappings and row shapes.
#


def dotted(keys, escape=False):
    parts = []
    for key in keys:
        part = str(key)
        if escape:
            part = part.replace(".", r"\.")
        parts.append(part)
    return ".".join(parts)


def nested_source(keys, target):
    source = target
    for key in reversed(keys):
        source = {key: source}
    return source


def group_forms(keys):
    # NOTE: list(keys) with two keys is a matches-property pair to group_by, not a
    # deep path; it is kept on purpose as a migration trap to detect.
    forms = [dotted(keys), list(keys)]
    escaped = dotted(keys, escape=True)
    if escaped != forms[0]:
        forms.append(escaped)
    return forms


def filter_forms(keys, target):
    forms = [
        {dotted(keys): target},  # match dict on the naive dotted key (literal read)
        nested_source(keys, target),  # match dict mirroring the intended structure
        [dotted(keys), target],  # pair on the naive dotted key (path read)
        [list(keys), target],  # pair on the explicit key list (faithful)
    ]
    unique = []
    for form in forms:
        if form not in unique:
            unique.append(form)
    return unique


KEY_SEQS = [("a",), ("b",), ("a", "b"), ("a.b",), ("a", "a"), ("0",), (0,)]


def enumerate_mappings():
    for keys in KEY_SEQS:
        for group_form in group_forms(keys):
            for filter_form in filter_forms(keys, TARGET):
                yield ColumnMapping(keys, TARGET, filter_form, group_form)


ALL_MAPPINGS = list(enumerate_mappings())

ROW_SETS = {
    "flat-hit": [tracked({"a": 1})],
    "flat-mixed": [tracked({"a": 1}), tracked({"a": 2}), tracked({"a": 1})],
    "nested-hit": [tracked({"a": {"b": 1}})],
    "nested-miss": [tracked({"a": {"b": 2}})],
    "literal-dot": [tracked({"a.b": 1})],
    "dot-and-nested": [tracked({"a.b": 1, "a": {"b": 2}})],
    "same-name": [tracked({"a": {"a": 1}})],
    "missing": [tracked({"x": 1})],
    "empty": [tracked({})],
    "int-key": [tracked({0: 1})],
    "str-key": [tracked({"0": 1})],
    "consuming-replay": [consuming({"a": [1, 1]})],
    "consuming-short": [consuming({"a": [1]})],
}


@dataclass
class Case:
    mapping: ColumnMapping
    row_set: str
    row_specs: list
    divergence: object


def sweep():
    cases = []
    for mapping in ALL_MAPPINGS:
        for row_set, row_specs in ROW_SETS.items():
            divergence = case_class(mapping, row_specs)
            cases.append(Case(mapping, row_set, row_specs, divergence))
    return cases


#
# Greedy shrinker: reduce rows and mapping while the divergence class survives.
#


def form_rank(form):
    if isinstance(form, str):
        return (0, len(form))
    if isinstance(form, list):
        return (1, len(form))
    return (2, len(form))


def mapping_complexity(mapping):
    return (
        len(mapping.keys),
        form_rank(mapping.group_form),
        form_rank(mapping.filter_form),
    )


def row_neighbors(row_specs):
    for index, (kind, payload) in enumerate(row_specs):
        if len(row_specs) > 1:
            yield row_specs[:index] + row_specs[index + 1 :]
        for key in payload:
            trimmed = {k: v for k, v in payload.items() if k != key}
            yield row_specs[:index] + [(kind, trimmed)] + row_specs[index + 1 :]
        if kind == "tracked":
            for key, value in payload.items():
                if isinstance(value, dict) and value:
                    emptied = dict(payload)
                    emptied[key] = {}
                    yield row_specs[:index] + [(kind, emptied)] + row_specs[index + 1 :]
        if kind == "consuming":
            for key, values in payload.items():
                if len(values) > 1:
                    shortened = dict(payload)
                    shortened[key] = values[:-1]
                    yield row_specs[:index] + [(kind, shortened)] + row_specs[index + 1 :]


def mapping_neighbors(mapping):
    base = mapping_complexity(mapping)
    for candidate in ALL_MAPPINGS:
        if mapping_complexity(candidate) < base:
            yield candidate


def neighbors(mapping, row_specs):
    for candidate_rows in row_neighbors(row_specs):
        yield mapping, candidate_rows
    for candidate_mapping in mapping_neighbors(mapping):
        yield candidate_mapping, row_specs


def shrink(mapping, row_specs, target_class):
    """Greedily minimize rows and mapping while `target_class` reproduces."""
    while True:
        for candidate_mapping, candidate_rows in neighbors(mapping, row_specs):
            if case_class(candidate_mapping, candidate_rows) == target_class:
                mapping, row_specs = candidate_mapping, candidate_rows
                break
        else:
            return mapping, row_specs


def shrink_all(target_class):
    """Shrink every divergent case of `target_class` to its local minimum."""
    witnesses = []
    for case in sweep():
        if case.divergence != target_class:
            continue
        witness = shrink(case.mapping, case.row_specs, target_class)
        if witness not in witnesses:
            witnesses.append(witness)
    return witnesses


#
# Replay and equivalence tests.
#


@parametrize("style", ["explicit", "shorthand"])
def test_filter_then_group_replays_rows_in_order(style):
    mapping = ColumnMapping(("a",), TARGET, {"a": 1}, "a")
    outcome = run_pipeline(ROW_SETS["flat-mixed"], mapping, style)
    assert outcome.error is None
    assert outcome.groups == {1: [0, 2]}
    # Filter reads every row in order, then the group pass re-reads each survivor.
    assert outcome.log == [(0, "a"), (1, "a"), (2, "a"), (0, "a"), (2, "a")]


@parametrize(
    "keys,filter_form,group_form,row_set",
    [
        (("a",), {"a": 1}, "a", "flat-hit"),
        (("a",), ["a", 1], ["a"], "flat-mixed"),
        (("a", "b"), {"a": {"b": 1}}, "a.b", "nested-hit"),
        (("a", "b"), ["a.b", 1], "a.b", "nested-hit"),
        (("a", "b"), [["a", "b"], 1], "a.b", "nested-miss"),
        (("a.b",), {"a.b": 1}, ["a.b"], "literal-dot"),
        (("a.b",), [["a.b"], 1], r"a\.b", "literal-dot"),
        (("a", "a"), {"a": {"a": 1}}, "a.a", "same-name"),
        (("0",), {"0": 1}, ["0"], "str-key"),
        ((0,), {0: 1}, [0], "int-key"),
        (("a",), {"a": 1}, "a", "consuming-replay"),
    ],
)
def test_faithful_shorthands_match_explicit(keys, filter_form, group_form, row_set):
    mapping = ColumnMapping(keys, TARGET, filter_form, group_form)
    explicit = run_pipeline(ROW_SETS[row_set], mapping, "explicit")
    shorthand = run_pipeline(ROW_SETS[row_set], mapping, "shorthand")
    assert explicit.error is None
    assert shorthand.error is None
    assert shorthand.groups == explicit.groups
    assert shorthand.log == explicit.log


@parametrize("filter_form", [{"a": 1}, ["a", 1]])
def test_consuming_field_replays_once_per_pass(filter_form):
    mapping = ColumnMapping(("a",), TARGET, filter_form, "a")
    outcome = run_pipeline(ROW_SETS["consuming-replay"], mapping, "shorthand")
    assert outcome.error is None
    assert outcome.groups == {1: [0]}
    # One read for the filter pass and one for the group pass, never more.
    assert outcome.log == [(0, "a"), (0, "a")]


#
# Divergence class tests.
#


def test_literal_dot_key_reads_wrong_column():
    mapping = ColumnMapping(("a.b",), TARGET, {"a.b": 1}, "a.b")
    explicit = run_pipeline(ROW_SETS["literal-dot"], mapping, "explicit")
    shorthand = run_pipeline(ROW_SETS["literal-dot"], mapping, "shorthand")
    assert classify(explicit, shorthand) == WRONG_COLUMN
    assert explicit.groups == {1: [0]}
    # The path string splits the literal key and groups the row under None.
    assert shorthand.groups == {None: [0]}
    assert explicit.log == [(0, "a.b"), (0, "a.b")]
    assert shorthand.log == [(0, "a.b"), (0, "a")]


def test_literal_dot_key_with_nested_decoy_reads_decoy_value():
    mapping = ColumnMapping(("a.b",), TARGET, {"a.b": 1}, "a.b")
    explicit = run_pipeline(ROW_SETS["dot-and-nested"], mapping, "explicit")
    shorthand = run_pipeline(ROW_SETS["dot-and-nested"], mapping, "shorthand")
    assert classify(explicit, shorthand) == WRONG_COLUMN
    assert explicit.groups == {1: [0]}
    # The shorthand walks into the nested decoy and groups by its value.
    assert shorthand.groups == {2: [0]}
    assert shorthand.log == [(0, "a.b"), (0, "a"), (0, "b")]


def test_pair_form_splits_literal_dot_key_but_match_dict_does_not():
    row_specs = ROW_SETS["literal-dot"]
    match_dict = ColumnMapping(("a.b",), TARGET, {"a.b": 1}, ["a.b"])
    pair = ColumnMapping(("a.b",), TARGET, ["a.b", 1], ["a.b"])
    assert run_pipeline(row_specs, match_dict, "shorthand").groups == {1: [0]}
    # The pair form parses "a.b" as a path, misses, and drops the row.
    assert run_pipeline(row_specs, pair, "shorthand").groups == {}


def test_numeric_string_key_costs_one_extra_read():
    mapping = ColumnMapping((0,), TARGET, {"0": 1}, "0")
    explicit = run_pipeline(ROW_SETS["int-key"], mapping, "explicit")
    shorthand = run_pipeline(ROW_SETS["int-key"], mapping, "shorthand")
    assert classify(explicit, shorthand) == EXTRA_READ
    assert explicit.groups == shorthand.groups == {1: [0]}
    # The dict int-key fallback probes "0" before 0, once per pipeline pass.
    assert explicit.log == [(0, 0), (0, 0)]
    assert shorthand.log == [(0, "0"), (0, 0), (0, "0"), (0, 0)]


def test_missing_column_is_grouped_as_empty():
    mapping = ColumnMapping(("a",), TARGET, {"a": 1}, "a")
    explicit = run_pipeline(ROW_SETS["missing"], mapping, "explicit")
    shorthand = run_pipeline(ROW_SETS["missing"], mapping, "shorthand")
    assert classify(explicit, shorthand) == MISSING_AS_EMPTY
    assert isinstance(explicit.error, KeyError)
    # The shorthand silently drops the row instead of failing.
    assert shorthand.groups == {}


def test_exhausted_consuming_field_is_grouped_as_empty():
    mapping = ColumnMapping(("a",), TARGET, {"a": 1}, "a")
    explicit = run_pipeline(ROW_SETS["consuming-short"], mapping, "explicit")
    shorthand = run_pipeline(ROW_SETS["consuming-short"], mapping, "shorthand")
    assert classify(explicit, shorthand) == MISSING_AS_EMPTY
    assert isinstance(explicit.error, ExhaustedError)
    # The field cannot be re-read for grouping, so the row lands under None.
    assert shorthand.groups == {None: [0]}


#
# Exhaustive sweep and shrinking.
#


def test_exhaustive_sweep_covers_every_divergence_class():
    cases = sweep()
    observed = {case.divergence for case in cases}
    assert observed <= {None, WRONG_COLUMN, EXTRA_READ, MISSING_AS_EMPTY}
    for divergence in DIVERGENCE_CLASSES:
        assert divergence in observed
    # Faithful encodings never diverge on the plain row shapes.
    faithful = ColumnMapping(("a", "b"), TARGET, {"a": {"b": 1}}, "a.b")
    assert case_class(faithful, ROW_SETS["nested-hit"]) is None


def test_shrunken_witnesses_are_minimal():
    for divergence in DIVERGENCE_CLASSES:
        witnesses = shrink_all(divergence)
        assert witnesses
        for mapping, row_specs in witnesses:
            assert case_class(mapping, row_specs) == divergence
            for candidate_mapping, candidate_rows in neighbors(mapping, row_specs):
                assert case_class(candidate_mapping, candidate_rows) != divergence


def test_shrunken_wrong_column_witnesses():
    assert (
        ColumnMapping(("a.b",), TARGET, ["a.b", 1], "a.b"),
        [tracked({"a.b": 1})],
    ) in shrink_all(WRONG_COLUMN)


def test_shrunken_extra_read_witnesses():
    assert (
        ColumnMapping((0,), TARGET, ["0", 1], "0"),
        [tracked({0: 1})],
    ) in shrink_all(EXTRA_READ)


def test_shrunken_missing_as_empty_witnesses():
    assert (
        ColumnMapping(("a",), TARGET, ["a", 1], "a"),
        [tracked({})],
    ) in shrink_all(MISSING_AS_EMPTY)
