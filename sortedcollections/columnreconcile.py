"""Column-order reconciliation for schema migration.

A data dictionary moves from an old field set to a new one. Some old columns
are renamed, some are split into a consecutive run of new columns, and every
regional template only records the partial column order it uses. This module
reconciles those partial orders into the fields that can be claimed
consistently.

An *interpretation* of an old column is the tuple of new columns it becomes:
a one-element tuple for a rename and a multi-element tuple for a split. The
caller declares every candidate interpretation. The same old column must keep
the same interpretation in every template, so each old column is one global
variable; templates only supply ordering evidence.

A template contributes two kinds of evidence from the relative order of the
old columns it contains and the new columns they map to:

- order: the anchor blocks of two old columns must appear in the same order
  as the old columns themselves;
- adjacency: two old columns that are consecutive in the template's old order
  must map to back-to-back anchor blocks in the template's new order.

Claims are also injective: a new column can only be claimed once.

When several interpretations stay feasible, only the old columns whose every
feasible interpretation lands in the same place are migrated automatically.
The rest stay ambiguous and two distinct full mappings are returned as
witnesses. If nothing is feasible, every inclusion-minimal group of template
constraints behind the conflict is returned: with one core, dropping any
single member restores feasibility, while several independent cores each
need one member relaxed.

Example::

    >>> result = reconcile_columns(
    ...     ['a', 'b'], ['x', 'y', 'z'],
    ...     {'a': [('x',)], 'b': [('y',), ('z',)]},
    ...     [{'name': 't1', 'old': ['a', 'b'], 'new': ['z', 'x', 'y']}],
    ... )
    >>> result.feasible
    True
    >>> result.consistent
    {'a': ('x',), 'b': ('y',)}

Without the template the choice between ``y`` and ``z`` stays open::

    >>> result = reconcile_columns(
    ...     ['a', 'b'], ['x', 'y', 'z'],
    ...     {'a': [('x',)], 'b': [('y',), ('z',)]},
    ... )
    >>> result.ambiguous
    ('b',)
    >>> len(result.witness_mappings)
    2

Contradictory templates yield a minimal conflict core instead::

    >>> result = reconcile_columns(
    ...     ['a', 'b'], ['x', 'y', 'z'],
    ...     {'a': [('x',), ('z',)], 'b': [('y',)]},
    ...     [{'name': 't1', 'old': ['a', 'b'], 'new': ['z', 'y', 'x']},
    ...      {'name': 't2', 'old': ['a', 'b'], 'new': ['x', 'y', 'z']}],
    ... )
    >>> result.feasible
    False
    >>> sorted(template.name for template in result.conflict)
    ['t1', 't2']

"""

__all__ = [
    'ColumnReconcileError',
    'TemplateOrder',
    'ReconcileResult',
    'reconcile_columns',
]


class ColumnReconcileError(ValueError):
    """Raised when reconciliation inputs are malformed."""


class TemplateOrder:
    """A template's partial old and new column orders.

    Attributes:
        name: template label used in conflict reports;
        index: zero-based position of the template among the submissions;
        old: old columns in this template's local order;
        new: new columns in this template's local order.

    """

    # pylint: disable=too-few-public-methods

    def __init__(self, name, index, old, new):
        self.name = name
        self.index = index
        self.old = tuple(old)
        self.new = tuple(new)
        self.old_columns = frozenset(self.old)
        self.new_pos = {column: pos for pos, column in enumerate(self.new)}

    @property
    def key(self):
        "Identity used to merge repeated submissions of the same template."
        return self.old, self.new

    def __repr__(self):
        return f'TemplateOrder({self.name!r}, {self.old!r}, {self.new!r})'


class ReconcileResult:
    """Outcome of :func:`reconcile_columns`.

    Attributes:
        feasible: whether at least one global assignment exists;
        consistent: mapping of old columns to the new-column tuple claimed in
            every feasible assignment; these are safe to migrate;
        ambiguous: old columns whose claim differs between feasible
            assignments;
        witness_mappings: two distinct full mappings when ambiguous, else an
            empty tuple;
        conflict: templates participating in any minimal conflict. With a
            single minimal conflict, dropping any one member restores
            feasibility; with several, one member of each core must be
            relaxed. Empty when the problem is feasible or the conflict comes
            from anchor collisions no template can fix;
        conflict_cores: inclusion-minimal infeasible template groups, each
            given as a tuple of :class:`TemplateOrder` objects;
        reasons: human-readable explanations of the conflict, one per member
            of a single core, or one per core when several overlap;
        templates: the deduplicated template orders actually considered;
        solution_count: number of feasible assignments found.

    """

    # pylint: disable=too-few-public-methods,too-many-arguments
    def __init__(self, consistent, ambiguous, witness_mappings, feasible,
                 conflict, conflict_cores, reasons, templates,
                 solution_count):
        self.consistent = dict(consistent)
        self.ambiguous = tuple(ambiguous)
        self.witness_mappings = tuple(witness_mappings)
        self.feasible = feasible
        self.conflict = tuple(conflict)
        self.conflict_cores = tuple(
            tuple(core) for core in conflict_cores)
        self.reasons = tuple(reasons)
        self.templates = tuple(templates)
        self.solution_count = solution_count

    @property
    def mapping(self):
        "New dict view of the fields safe to migrate automatically."
        return dict(self.consistent)

    def __repr__(self):
        state = 'feasible' if self.feasible else 'conflicting'
        return (f'ReconcileResult({state}, consistent={self.consistent!r}, '
                f'ambiguous={self.ambiguous!r})')


def reconcile_columns(old_fields, new_fields, candidates, templates=()):
    """Reconcile old and new column orders across partial templates.

    Args:
        old_fields: ordered iterable of old dictionary columns;
        new_fields: ordered iterable of new dictionary columns;
        candidates: mapping of old column to its candidate interpretations,
            each interpretation being a new column name (rename) or an
            iterable of new column names (split);
        templates: iterable of template orders. Each item is a mapping with
            ``old``/``new`` keys (optional ``name``), a ``(old, new)`` pair
            or a ``(name, old, new)`` triple. Repeated submissions with the
            same old and new order count once.

    Returns:
        ReconcileResult.

    Raises:
        ColumnReconcileError: if a candidate or template references unknown
            fields, an interpretation is empty, or a template is malformed.

    """
    old_order = _dedupe_fields(old_fields)
    new_order = _dedupe_fields(new_fields)
    old_set, new_set = frozenset(old_order), frozenset(new_order)
    interpretations = _normalize_candidates(candidates, old_set, new_set)
    template_orders = _normalize_templates(templates, old_set, new_set)

    solution_count, intersection, witnesses = _collect_solutions(
        interpretations, template_orders)

    if solution_count:
        ambiguous = [old for old in interpretations if old not in intersection]
        return ReconcileResult(
            consistent=intersection,
            ambiguous=ambiguous,
            witness_mappings=witnesses if solution_count > 1 else (),
            feasible=True,
            conflict=(),
            conflict_cores=(),
            reasons=(),
            templates=template_orders,
            solution_count=solution_count,
        )

    cores, reasons = _minimal_conflicts(interpretations, template_orders)
    conflict = tuple(dict.fromkeys(
        template for core in cores for template in core))
    return ReconcileResult(
        consistent={},
        ambiguous=list(interpretations),
        witness_mappings=(),
        feasible=False,
        conflict=conflict,
        conflict_cores=cores,
        reasons=reasons,
        templates=template_orders,
        solution_count=0,
    )


def _dedupe_fields(fields):
    "Return fields in first-seen order, collapsing duplicate entries."
    ordered = []
    seen = set()
    for field in fields:
        if field not in seen:
            seen.add(field)
            ordered.append(field)
    return ordered


def _normalize_candidates(candidates, old_set, new_set):
    "Turn caller input into old column -> tuple of interpretation tuples."
    interpretations = {}
    for old, groups in candidates.items():
        if old not in old_set:
            raise ColumnReconcileError(
                f'candidate old column {old!r} is not in old_fields')
        if isinstance(groups, str):
            groups = [groups]
        normalized = []
        for group in groups:
            if isinstance(group, str):
                group = (group,)
            else:
                group = tuple(group)
            if not group:
                raise ColumnReconcileError(
                    f'interpretation of {old!r} must not be empty')
            for new_column in group:
                if new_column not in new_set:
                    raise ColumnReconcileError(
                        f'candidate new column {new_column!r} is not in '
                        f'new_fields (for old column {old!r})')
            if group not in normalized:
                normalized.append(group)
        if not normalized:
            raise ColumnReconcileError(
                f'old column {old!r} has no candidate interpretations')
        interpretations[old] = tuple(normalized)
    return interpretations


def _normalize_templates(templates, old_set, new_set):
    "Validate templates, attach names and merge repeated submissions."
    orders = []
    seen = set()
    for index, item in enumerate(templates):
        order = _coerce_template(item, index, old_set, new_set)
        if order.key in seen:
            continue
        seen.add(order.key)
        orders.append(order)
    return orders


def _coerce_template(item, index, old_set, new_set):
    "Accept mappings, pairs and triples as template orders."
    if isinstance(item, TemplateOrder):
        name, old, new = item.name, item.old, item.new
    elif isinstance(item, dict):
        name = item.get('name')
        old = tuple(item.get('old', ()))
        new = tuple(item.get('new', ()))
    else:
        try:
            sequence = tuple(item)
        except TypeError as exc:
            raise ColumnReconcileError(
                f'template #{index} is not a mapping, pair or triple') from exc
        if len(sequence) == 2:
            name, old, new = None, sequence[0], sequence[1]
        elif len(sequence) == 3:
            name, old, new = sequence
        else:
            raise ColumnReconcileError(
                f'template #{index} must be a (old, new) pair or a '
                f'(name, old, new) triple')
        old, new = tuple(old), tuple(new)

    for field in old:
        if field not in old_set:
            raise ColumnReconcileError(
                f'template #{index} references unknown old column {field!r}')
    for field in new:
        if field not in new_set:
            raise ColumnReconcileError(
                f'template #{index} references unknown new column {field!r}')
    if name is None:
        name = f'template-{index}'
    return TemplateOrder(name, index, old, new)


def _collect_solutions(interpretations, templates):
    "Enumerate feasible assignments, keeping the intersection and two views."
    solution_count = 0
    intersection = None
    witnesses = []
    for assignment in _feasible_assignments(interpretations, templates):
        solution_count += 1
        if intersection is None:
            intersection = dict(assignment)
        else:
            for old in list(intersection):
                if assignment.get(old) != intersection[old]:
                    del intersection[old]
        if len(witnesses) < 2:
            witnesses.append(dict(assignment))
    return solution_count, intersection or {}, witnesses


def _feasible_assignments(interpretations, templates):
    """Yield feasible assignments as old column -> interpretation tuples.

    Template checks are restricted to the newly assigned column to prune the
    search tree without rechecking settled pairs.
    """
    assignment = {}
    owner = {}
    fixed = {old: groups[0] for old, groups in interpretations.items()
             if len(groups) == 1}
    uncertain = sorted(
        (old for old, groups in interpretations.items() if len(groups) > 1),
        key=lambda old: (len(interpretations[old]), old))

    for old, group in fixed.items():
        clash = _claim(group, owner)
        if clash is not None:
            return
        assignment[old] = group

    if _violations(templates, assignment, None):
        return

    def _search(depth):
        "Backtrack over uncertain columns, pruning failed branches."
        if depth == len(uncertain):
            yield dict(assignment)
            return
        old = uncertain[depth]
        for group in interpretations[old]:
            clash = _claim(group, owner)
            if clash is not None:
                continue
            assignment[old] = group
            if not _violations(templates, assignment, old):
                yield from _search(depth + 1)
            del assignment[old]
            _release(group, owner)

    yield from _search(0)


def _claim(group, owner):
    "Record anchor ownership; return the first colliding new column or None."
    for new_column in group:
        if new_column in owner:
            return new_column
    for new_column in group:
        owner[new_column] = True
    return None


def _release(group, owner):
    "Undo :func:`_claim` for a successfully placed interpretation."
    for new_column in group:
        del owner[new_column]


def _violations(templates, assignment, only_old):
    "Return violated template constraints as (template, kind, a, b) tuples."
    found = []
    for template in templates:
        for kind, a, b in _template_violations(template, assignment, only_old):
            found.append((template, kind, a, b))
    return found


def _template_violations(template, assignment, only_old):
    "Check order and adjacency evidence supplied by one template."
    blocks = {}
    for old, group in assignment.items():
        if old not in template.old_columns:
            continue
        positions = sorted(template.new_pos[column] for column in group
                           if column in template.new_pos)
        if positions:
            blocks[old] = positions

    old_index = {column: pos for pos, column in enumerate(template.old)}
    olds = list(blocks)
    violations = []
    for i, first in enumerate(olds):
        for second in olds[i + 1:]:
            if only_old is not None and only_old not in (first, second):
                continue
            earlier, later = first, second
            if old_index[earlier] > old_index[later]:
                earlier, later = later, earlier
            earlier_pos = blocks[earlier]
            later_pos = blocks[later]
            if max(earlier_pos) >= min(later_pos):
                violations.append(('order', earlier, later))
                continue
            if old_index[later] - old_index[earlier] == 1:
                if max(earlier_pos) + 1 != min(later_pos):
                    violations.append(('adjacent', earlier, later))
    return violations


def _minimal_conflicts(interpretations, templates):
    "Return every inclusion-minimal infeasible template group and reasons."
    if next(_feasible_assignments(interpretations, ()), None) is None:
        return (), _collision_reasons(interpretations)

    minimal = set()
    memo = {}

    def _is_feasible(index_tuple):
        "Memoized feasibility check for a subset of templates."
        if index_tuple not in memo:
            group = [templates[i] for i in index_tuple]
            memo[index_tuple] = next(
                _feasible_assignments(interpretations, group), None
            ) is not None
        return memo[index_tuple]

    def _search(indices):
        "Collect inclusion-minimal infeasible index groups."
        index_set = frozenset(indices)
        if any(core <= index_set for core in minimal):
            return
        reduced = False
        for position, index in enumerate(indices):
            trial = indices[:position] + indices[position + 1:]
            if not _is_feasible(trial):
                _search(trial)
                reduced = True
        if not reduced:
            minimal.add(index_set)

    _search(tuple(range(len(templates))))

    cores = tuple(
        tuple(templates[i] for i in sorted(core))
        for core in sorted(minimal, key=lambda core: (len(core), sorted(
            templates[i].name for i in core))))
    if len(cores) == 1:
        core = cores[0]
        reasons = []
        for template in core:
            witness = next(_feasible_assignments(
                interpretations,
                [item for item in core if item is not template]))
            reasons.extend(_explain(template, witness))
        return cores, reasons

    reasons = []
    for number, core in enumerate(cores, start=1):
        names = ', '.join(template.name for template in core)
        reasons.append(f'conflict core {number}: relax one of {names}')
    return cores, reasons


def _collision_reasons(interpretations):
    "Describe anchor collisions that remain even after dropping templates."
    reasons = []
    olds = list(interpretations)
    for i, first in enumerate(olds):
        for second in olds[i + 1:]:
            if all(set(group_a) & set(group_b)
                   for group_a in interpretations[first]
                   for group_b in interpretations[second]):
                reasons.append(
                    f'{first!r} and {second!r} cannot both be placed: every '
                    f'candidate interpretation shares a new column'
                )
    if not reasons:
        reasons.append('no assignment satisfies the candidate claims without '
                       'template evidence (a group of columns claims fewer '
                       'new columns than it has members)')
    return reasons


def _explain(template, witness):
    "Turn a template's violations under a witness into readable reasons."
    reasons = []
    for kind, earlier, later in _template_violations(template, witness, None):
        if kind == 'adjacent':
            wording = 'must map to back-to-back columns but do not'
        else:
            wording = 'must keep their old relative order but do not'
        reasons.append(
            f'template {template.name!r}: {earlier!r} and {later!r} '
            f'{wording}'
        )
    return reasons
