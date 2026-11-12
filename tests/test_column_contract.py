"""Regression tests for building a column directory from formatted SQL.

A data warehouse may feed the *formatted* text of a query back into the same
component that derives a column directory from the original statement.  The
column contract of a query therefore has to survive a formatting round trip.

The contract derived here is intentionally independent of any single token
layout.  Every projection row is a tuple::

    (level, ordinal, output_name, alias, input_refs)

* ``level`` counts nested projections, the outer SELECT is level 0.
* ``ordinal`` is the 1-based position of the output column in its SELECT.
* ``output_name`` is the alias when present, otherwise the real name
  (``'*'`` for a wildcard, ``None`` when no name can be derived).
* ``alias`` is the explicit or implicit alias, ``None`` when absent.
* ``input_refs`` is a tuple of ``(parent, column)`` pairs, one per input
  column reference in textual order; ``parent`` is the qualifying name or
  ``None``.  Quoted identifiers are recorded without their quotes.

Two rows that print the same output name but live on different levels are
distinct rows and must never be merged.  When formatting only moves tokens
around the contract stays identical; when a formatting option changes the
binding itself (for example stripping a comment that sat between a column and
its would-be alias) the divergence is pinned together with a deterministically
shrunk query that still mismatches the same source column.
"""

import re

import pytest

import sqlparse
from sqlparse import sql, tokens as T
from sqlparse.utils import remove_quotes

# Keywords that end the SELECT projection while walking statement tokens.
_STOP_KEYWORDS = frozenset({
    'FROM', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'LIMIT', 'UNION',
    'INTERSECT', 'EXCEPT', 'WINDOW', 'FETCH', 'INTO',
})
_NAME_TYPES = (T.Name, T.Literal.String.Symbol)
_TRAILING_CLAUSES = ('WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT')


def refs(*pairs):
    """Build an ``input_refs`` tuple from ``(parent, column)`` pairs."""
    return tuple(pairs)


def _is_subquery(group):
    return any(
        token.ttype is T.Keyword.DML and token.normalized == 'SELECT'
        for token in group.flatten())


def _projection_items(stmt):
    """Return the projection tokens of a single SELECT in textual order."""
    start = 0
    for idx, token in enumerate(stmt.tokens):
        if token.ttype is T.Keyword.DML and token.normalized == 'SELECT':
            start = idx + 1
            break

    items = []
    for token in stmt.tokens[start:]:
        if token.is_keyword and token.normalized in _STOP_KEYWORDS:
            break
        if isinstance(token, sql.IdentifierList):
            items.extend(token.get_identifiers())
        elif isinstance(token, (sql.Identifier, sql.Parenthesis,
                                sql.Function)):
            items.append(token)
        elif token.ttype is T.Wildcard:
            items.append(token)
    return items


def _alias_group(identifier):
    groups = [token for token in identifier.tokens
              if isinstance(token, sql.Identifier)]
    return groups[-1] if groups else None


def _expression_region(projection):
    """Tokens making up the expression, without the trailing alias."""
    if not isinstance(projection, sql.Identifier):
        return []
    cut = len(projection.tokens)
    as_idx, as_keyword = projection.token_next_by(m=(T.Keyword, 'AS'))
    if as_keyword is not None:
        cut = as_idx
    else:
        alias = _alias_group(projection)
        if alias is not None:
            cut = projection.token_index(alias)
    return projection.tokens[:cut]


def _flatten_expression(region):
    for token in region:
        # A nested subquery contributes its own contract on the next level.
        if isinstance(token, sql.Parenthesis) and _is_subquery(token):
            continue
        if token.is_group:
            yield from token.flatten()
        else:
            yield token


def _input_refs(projection):
    if projection.ttype is T.Wildcard:
        return ()
    if isinstance(projection, sql.Parenthesis) and _is_subquery(projection):
        return ()

    flat = list(_flatten_expression(_expression_region(projection)))
    result = []
    for idx, token in enumerate(flat):
        if token.ttype not in _NAME_TYPES:
            continue
        following = flat[idx + 1] if idx + 1 < len(flat) else None
        if following is not None and (
                following.match(T.Punctuation, '(')
                or following.match(T.Punctuation, '.')):
            # Function name or qualifier of a dotted name, not an input column.
            continue
        parent = None
        if idx >= 2 and flat[idx - 1].match(T.Punctuation, '.'):
            qualifier = flat[idx - 2]
            if qualifier.ttype in _NAME_TYPES:
                parent = remove_quotes(qualifier.value)
        result.append((parent, remove_quotes(token.value)))
    return tuple(result)


def _nested_select(projection):
    tokens = projection.tokens if isinstance(
        projection, (sql.Identifier, sql.Parenthesis)) else []
    for group in tokens:
        if isinstance(group, sql.Parenthesis) and _is_subquery(group):
            return group
    return None


def column_contract(stmt, level=0, rows=None):
    """Extract the labelled column contract of a parsed SELECT statement."""
    if rows is None:
        rows = []
    for ordinal, projection in enumerate(_projection_items(stmt), 1):
        if isinstance(projection, sql.Identifier):
            output_name = projection.get_name()
            alias = projection.get_alias()
        elif projection.ttype is T.Wildcard:
            output_name, alias = '*', None
        else:
            output_name, alias = None, None
        rows.append((level, ordinal, output_name, alias,
                     _input_refs(projection)))
        nested = _nested_select(projection)
        if nested is not None:
            column_contract(nested, level + 1, rows)
    return tuple(rows)


def _all_refs(contract):
    return {ref for row in contract for ref in row[4]}


def focal_columns(raw_contract, formatted_contract):
    """Source columns referenced on both sides of a mismatched projection.

    Rows are paired by ``(level, ordinal)``; only rows whose contract changed
    participate, and the focal source columns are the input references shared
    by the raw and formatted representation of that same projection slot.
    """
    formatted_rows = {(row[0], row[1]): row for row in formatted_contract}
    focal = set()
    for raw_row in raw_contract:
        formatted_row = formatted_rows.get((raw_row[0], raw_row[1]))
        if formatted_row is not None and raw_row != formatted_row:
            focal |= set(raw_row[4]) & set(formatted_row[4])
    return focal


def _contracts_for(sql_text, options):
    raw_contract = column_contract(sqlparse.parse(sql_text)[0])
    formatted = sqlparse.format(sql_text, **options)
    formatted_contract = column_contract(sqlparse.parse(formatted)[0])
    return raw_contract, formatted_contract, formatted


# ---------------------------------------------------------------------------
# Deterministic shrinking of a diverging query.
# ---------------------------------------------------------------------------

_PROJECTION_RE = re.compile(r'(?is)^(\s*SELECT\s+)')
_FROM_RE = re.compile(r'(?i)FROM\b')


def _skip_comment(sql_text, index):
    """Return the index just past a comment starting at *index*."""
    if sql_text.startswith('/*', index):
        end = sql_text.find('*/', index + 2)
        return len(sql_text) if end == -1 else end + 2
    if sql_text.startswith('--', index):
        end = sql_text.find('\n', index + 2)
        return len(sql_text) if end == -1 else end + 1
    return index + 1


def _top_level_commas(body):
    """Indices of commas that sit outside quotes, comments and parentheses."""
    commas = []
    depth = 0
    quote = None
    index = 0
    while index < len(body):
        char = body[index]
        if quote is not None:
            if char == quote:
                quote = None
            index += 1
            continue
        if body.startswith(('/*', '--'), index):
            index = _skip_comment(body, index)
            continue
        if char in ("'", '"', '`'):
            quote = char
        elif char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif char == ',' and depth == 0:
            commas.append(index)
        index += 1
    return commas


def _split_projection(sql_text):
    """Split a SELECT into head, projection body and FROM tail.

    Parentheses, quotes and comments are respected, so a FROM inside a nested
    subquery does not terminate the outer projection.
    """
    match = _PROJECTION_RE.match(sql_text)
    if match is None:
        return None
    head_end = match.end()
    depth = 0
    quote = None
    index = head_end
    from_start = None
    while index < len(sql_text):
        char = sql_text[index]
        if quote is not None:
            if char == quote:
                quote = None
            index += 1
            continue
        if sql_text.startswith(('/*', '--'), index):
            index = _skip_comment(sql_text, index)
            continue
        if char in ("'", '"', '`'):
            quote = char
            index += 1
            continue
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif depth == 0 and _FROM_RE.match(sql_text, index):
            if not sql_text[index - 1].isalnum():
                from_start = index
                break
        index += 1
    if from_start is None:
        return None
    return (sql_text[:head_end], sql_text[head_end:from_start],
            sql_text[from_start:])


def _without_projection_item(sql_text, index):
    split = _split_projection(sql_text)
    if split is None:
        return None
    head, body, tail = split
    commas = _top_level_commas(body)
    if len(commas) == 0:
        return None
    if index == 0:
        end = commas[0] + 1
        while end < len(body) and body[end] == ' ':
            end += 1
        new_body = body[end:]
    else:
        start = commas[index - 1]
        end = commas[index] + 1 if index < len(commas) else len(body)
        new_body = body[:start] + body[end:]
    separator = ' ' if new_body and not new_body[-1].isspace() else ''
    return head + new_body + separator + tail


def _drop_trailing_clause(sql_text):
    for keyword in _TRAILING_CLAUSES:
        match = re.search(r'\s+' + keyword + r'\b', sql_text,
                          flags=re.IGNORECASE)
        if match is not None:
            return sql_text[:match.start()]
    return None


def _still_diverges(sql_text, options, focal):
    raw_contract, formatted_contract, _ = _contracts_for(sql_text, options)
    if raw_contract == formatted_contract:
        return False
    shared = _all_refs(raw_contract) & _all_refs(formatted_contract)
    return focal <= shared


def shrink_divergence(sql_text, options, focal):
    """Remove clauses and sibling columns while keeping the divergence.

    Every accepted reduction must still produce a contract mismatch and must
    still reference every focal source column, so the shrunk query pins the
    same binding defect as the original one.
    """
    current = sql_text
    changed = True
    while changed:
        changed = False
        candidate = _drop_trailing_clause(current)
        if candidate is not None and _still_diverges(
                candidate, options, focal):
            current = candidate
            changed = True
            continue
        split = _split_projection(current)
        if split is None:
            continue
        item_count = len(_top_level_commas(split[1])) + 1
        for index in range(item_count):
            candidate = _without_projection_item(current, index)
            if candidate is not None and _still_diverges(
                    candidate, options, focal):
                current = candidate
                changed = True
                break
    return current


# ---------------------------------------------------------------------------
# Independent expectations for small, source-labelled SELECT queries.
# ---------------------------------------------------------------------------

CONTRACT_CASES = [
    pytest.param(
        'simple_select',
        'SELECT a, b FROM t1',
        (
            (0, 1, 'a', None, refs((None, 'a'))),
            (0, 2, 'b', None, refs((None, 'b'))),
        ),
        id='simple_select'),
    pytest.param(
        'explicit_and_implicit_alias',
        'SELECT a AS x, b y FROM t1',
        (
            (0, 1, 'x', 'x', refs((None, 'a'))),
            (0, 2, 'y', 'y', refs((None, 'b'))),
        ),
        id='aliases'),
    pytest.param(
        'quoted_identifiers_lose_quotes_in_contract',
        'SELECT "a", "b" AS "c" FROM t1',
        (
            (0, 1, 'a', None, refs((None, 'a'))),
            (0, 2, 'c', 'c', refs((None, 'b'))),
        ),
        id='quoted_identifiers'),
    pytest.param(
        'qualified_names_keep_parent',
        'SELECT t1.a, t2.b FROM t1, t2',
        (
            (0, 1, 'a', None, refs(('t1', 'a'))),
            (0, 2, 'b', None, refs(('t2', 'b'))),
        ),
        id='qualified_names'),
    pytest.param(
        'newline_acts_as_implicit_alias_separator',
        'SELECT a\nx FROM t1',
        ((0, 1, 'x', 'x', refs((None, 'a'))),),
        id='newline_alias'),
    pytest.param(
        'comma_inside_string_does_not_split_columns',
        "SELECT 'a, b' AS x FROM t1",
        ((0, 1, 'x', 'x', ()),),
        id='string_with_comma'),
    pytest.param(
        'string_at_alias_position_is_not_an_identifier',
        "SELECT a 'x' FROM t1",
        ((0, 1, 'a', None, refs((None, 'a'))),),
        id='string_at_binding_position'),
    pytest.param(
        'string_expression_with_implicit_alias',
        "SELECT 'x' y FROM t1",
        ((0, 1, 'y', None, refs((None, 'y'))),),
        id='string_expression_implicit_alias'),
    pytest.param(
        'hint_comment_is_kept_outside_the_contract',
        'SELECT /*+ INDEX(t1 idx) */ a FROM t1',
        ((0, 1, 'a', None, refs((None, 'a'))),),
        id='hint_comment'),
    pytest.param(
        'function_argument_is_an_input_column',
        'SELECT count(a) AS n FROM t1',
        ((0, 1, 'n', 'n', refs((None, 'a'))),),
        id='function_argument'),
    pytest.param(
        'expression_tracks_each_reference_in_order',
        'SELECT f(a) + b AS z FROM t1',
        ((0, 1, 'z', 'z', refs((None, 'a'), (None, 'b'))),),
        id='expression_references'),
    pytest.param(
        'wildcard_has_no_input_refs',
        'SELECT * FROM t1',
        ((0, 1, '*', None, ()),),
        id='wildcard'),
    pytest.param(
        'qualified_wildcard_stays_a_wildcard',
        'SELECT t1.* FROM t1',
        ((0, 1, '*', None, ()),),
        id='qualified_wildcard'),
    pytest.param(
        'nested_projection_same_name_is_a_separate_level',
        'SELECT a, (SELECT a FROM t2) AS a FROM t1',
        (
            (0, 1, 'a', None, refs((None, 'a'))),
            (0, 2, 'a', 'a', ()),
            (1, 1, 'a', None, refs((None, 'a'))),
        ),
        id='nested_same_name'),
    pytest.param(
        'nested_projection_with_alias',
        'SELECT a, (SELECT b FROM t2) s FROM t1',
        (
            (0, 1, 'a', None, refs((None, 'a'))),
            (0, 2, 's', 's', ()),
            (1, 1, 'b', None, refs((None, 'b'))),
        ),
        id='nested_aliased_subquery'),
    pytest.param(
        'two_levels_of_nested_projections',
        'SELECT (SELECT (SELECT c FROM t3) s2 FROM t2) s1 FROM t1',
        (
            (0, 1, 's1', 's1', ()),
            (1, 1, 's2', 's2', ()),
            (2, 1, 'c', None, refs((None, 'c'))),
        ),
        id='double_nested_projection'),
    pytest.param(
        'cte_definition_is_not_an_output_column',
        'WITH w AS (SELECT a FROM t2) SELECT a FROM w',
        ((0, 1, 'a', None, refs((None, 'a'))),),
        id='cte_not_in_projection'),
]


@pytest.mark.parametrize('label, sql_text, expected', CONTRACT_CASES)
def test_raw_column_contract_matches_expectation(label, sql_text, expected):
    assert column_contract(sqlparse.parse(sql_text)[0]) == expected


FORMAT_OPTIONS = [
    pytest.param({'reindent': True}, id='reindent'),
    pytest.param({'reindent_aligned': True}, id='reindent_aligned'),
    pytest.param({'keyword_case': 'upper'}, id='keyword_case'),
]


@pytest.mark.parametrize('label, sql_text, expected', CONTRACT_CASES)
@pytest.mark.parametrize('options', FORMAT_OPTIONS)
def test_column_contract_survives_formatting(label, sql_text, expected,
                                             options):
    raw_contract, formatted_contract, formatted = _contracts_for(
        sql_text, options)
    assert raw_contract == expected
    assert formatted_contract == expected


def test_same_output_name_on_different_levels_is_not_merged():
    sql_text = 'SELECT a, (SELECT a FROM t2) AS a FROM t1'
    contract = column_contract(sqlparse.parse(sql_text)[0])

    outer_rows = [row for row in contract if row[0] == 0 and row[2] == 'a']
    inner_rows = [row for row in contract if row[0] == 1 and row[2] == 'a']
    assert len(outer_rows) == 2
    assert len(inner_rows) == 1
    # The outer alias column hides its binding behind the subquery; the inner
    # column still binds directly to the source column of the same name.
    assert outer_rows[0][4] == refs((None, 'a'))
    assert outer_rows[1][4] == ()
    assert inner_rows[0][4] == refs((None, 'a'))
    assert len({(row[0], row[1]) for row in contract}) == len(contract)


def test_contract_preserves_column_order_and_each_binding():
    sql_text = 'SELECT t1.a, t2.b, a FROM t1, t2'
    contract = column_contract(sqlparse.parse(sql_text)[0])
    assert [row[1] for row in contract] == [1, 2, 3]
    assert [row[2] for row in contract] == ['a', 'b', 'a']
    assert [row[4] for row in contract] == [
        refs(('t1', 'a')), refs(('t2', 'b')), refs((None, 'a'))]

    formatted = sqlparse.format(sql_text, reindent=True)
    assert column_contract(sqlparse.parse(formatted)[0]) == contract


def test_newline_alias_becomes_space_but_contract_is_unchanged():
    raw_sql = 'SELECT a\nx FROM t1'
    formatted = sqlparse.format(raw_sql, reindent=True)
    # The newline that separated column and alias is normalised to a space,
    # but both spellings still describe the same alias binding.
    assert 'a\nx' in raw_sql
    assert 'a x' in formatted and 'a\nx' not in formatted
    assert column_contract(sqlparse.parse(formatted)[0]) == \
        column_contract(sqlparse.parse(raw_sql)[0])


def test_hint_survives_strip_comments_without_contract_change():
    sql_text = 'SELECT /*+ INDEX(t1 idx) */ a FROM t1'
    formatted = sqlparse.format(sql_text, strip_comments=True)
    assert '/*+ INDEX(t1 idx) */' in formatted
    assert column_contract(sqlparse.parse(formatted)[0]) == \
        column_contract(sqlparse.parse(sql_text)[0])


TYPOGRAPHY_CASES = [
    pytest.param('missing_spaces', 'SELECT a,b FROM t1',
                 'SELECT a, b FROM t1', id='missing_spaces'),
    pytest.param('extra_spaces', 'SELECT a,   b   FROM t1',
                 'SELECT a, b FROM t1', id='extra_spaces'),
    pytest.param('newline_before_from', 'SELECT a, b\nFROM t1',
                 'SELECT a, b FROM t1', id='newline_before_from'),
    pytest.param('spaced_alias', 'SELECT a  AS  x FROM t1',
                 'SELECT a AS x FROM t1', id='spaced_alias'),
]


@pytest.mark.parametrize('label, sql_text, compact_sql', TYPOGRAPHY_CASES)
def test_token_only_differences_are_recorded_as_layout(label, sql_text,
                                                       compact_sql):
    raw_contract = column_contract(sqlparse.parse(sql_text)[0])
    compact_contract = column_contract(sqlparse.parse(compact_sql)[0])
    # The wording differs only at the token/layout level; the column contract
    # is exactly the same.
    assert sql_text != compact_sql
    assert raw_contract == compact_contract


# ---------------------------------------------------------------------------
# Divergences: comments placed where binding is decided.
# ---------------------------------------------------------------------------

DIVERGENCE_CASES = [
    pytest.param(
        'block_comment_between_column_and_alias',
        'SELECT a /* c */ x, b FROM t1',
        {'strip_comments': True},
        (
            (0, 1, 'a', 'a', refs((None, 'a'))),
            (0, 2, 'x', None, refs((None, 'x'))),
            (0, 3, 'b', None, refs((None, 'b'))),
        ),
        (
            (0, 1, 'x', 'x', refs((None, 'a'))),
            (0, 2, 'b', None, refs((None, 'b'))),
        ),
        {(None, 'a')},
        'SELECT a /* c */ x FROM t1',
        id='block_comment_alias_position'),
    pytest.param(
        'block_comment_with_trailing_where_clause',
        'SELECT a /* c */ x, b FROM t1 WHERE b > 1',
        {'strip_comments': True},
        (
            (0, 1, 'a', 'a', refs((None, 'a'))),
            (0, 2, 'x', None, refs((None, 'x'))),
            (0, 3, 'b', None, refs((None, 'b'))),
        ),
        (
            (0, 1, 'x', 'x', refs((None, 'a'))),
            (0, 2, 'b', None, refs((None, 'b'))),
        ),
        {(None, 'a')},
        'SELECT a /* c */ x FROM t1',
        id='block_comment_plus_where'),
    pytest.param(
        'line_comment_between_column_and_alias',
        'SELECT a -- c\n x, b FROM t1',
        {'strip_comments': True},
        (
            (0, 1, 'a', 'a', refs((None, 'a'))),
            (0, 2, 'x', None, refs((None, 'x'))),
            (0, 3, 'b', None, refs((None, 'b'))),
        ),
        (
            (0, 1, 'x', 'x', refs((None, 'a'))),
            (0, 2, 'b', None, refs((None, 'b'))),
        ),
        {(None, 'a')},
        'SELECT a -- c\n x FROM t1',
        id='line_comment_alias_position'),
    pytest.param(
        'comment_inside_nested_projection',
        'SELECT (SELECT a /* c */ x FROM t2) s, b FROM t1',
        {'strip_comments': True},
        (
            (0, 1, 's', 's', ()),
            (1, 1, 'a', 'a', refs((None, 'a'))),
            (1, 2, 'x', None, refs((None, 'x'))),
            (0, 2, 'b', None, refs((None, 'b'))),
        ),
        (
            (0, 1, 's', 's', ()),
            (1, 1, 'x', 'x', refs((None, 'a'))),
            (0, 2, 'b', None, refs((None, 'b'))),
        ),
        {(None, 'a')},
        'SELECT (SELECT a /* c */ x FROM t2) s FROM t1',
        id='nested_comment_alias_position'),
    pytest.param(
        'trailing_comment_turns_into_implicit_alias',
        'SELECT a, b -- c\n FROM t1',
        {'strip_comments': True},
        (
            (0, 1, 'a', None, refs((None, 'a'))),
            (0, 2, 'b', 'b', refs((None, 'b'))),
        ),
        (
            (0, 1, 'a', None, refs((None, 'a'))),
            (0, 2, 'b', None, refs((None, 'b'))),
        ),
        {(None, 'b')},
        'SELECT b -- c\n FROM t1',
        id='trailing_comment_alias'),
]


@pytest.mark.parametrize(
    'label, sql_text, options, expected_raw, expected_formatted, focal, '
    'expected_minimal',
    DIVERGENCE_CASES)
def test_formatting_can_change_binding_contract(label, sql_text, options,
                                                expected_raw,
                                                expected_formatted, focal,
                                                expected_minimal):
    raw_contract, formatted_contract, formatted = _contracts_for(
        sql_text, options)
    assert raw_contract == expected_raw
    assert formatted_contract == expected_formatted
    assert raw_contract != formatted_contract
    assert focal_columns(raw_contract, formatted_contract) == focal
    # The comment text really was removed, so the divergence is caused by the
    # formatting option rather than by ordinary whitespace changes.
    assert '/* c */' not in formatted and '-- c' not in formatted


@pytest.mark.parametrize(
    'label, sql_text, options, expected_raw, expected_formatted, focal, '
    'expected_minimal',
    DIVERGENCE_CASES)
def test_divergence_shrinks_to_same_source_column(label, sql_text, options,
                                                  expected_raw,
                                                  expected_formatted, focal,
                                                  expected_minimal):
    minimal = shrink_divergence(sql_text, options, focal)

    assert minimal == expected_minimal
    assert len(minimal) <= len(sql_text)
    minimal_raw, minimal_formatted, _ = _contracts_for(minimal, options)
    assert minimal_raw != minimal_formatted
    assert focal <= (_all_refs(minimal_raw) & _all_refs(minimal_formatted))
    # Shrinking only removes sibling columns or trailing clauses.
    assert len(minimal) <= len(sql_text)
