#
# Copyright (C) 2009-2020 the sqlparse authors and contributors
# <see AUTHORS file>
#
# This module is part of python-sqlparse and is released under
# the BSD License: https://opensource.org/licenses/BSD-3-Clause

import unicodedata

import pytest

from sqlparse.layout import Cell, ProjectionLayout, layout_select


def display_width(text):
    width = 0
    for char in text:
        if char == '\t':
            width += 8 - width % 8
        elif unicodedata.combining(char):
            continue
        elif unicodedata.category(char) == 'Cc':
            continue
        else:
            width += 2 if unicodedata.east_asian_width(char) in ('W', 'F') \
                else 1
    return width


def test_basic_projections_and_source_offsets():
    sql_text = 'SELECT a, b FROM t'
    layout = layout_select(sql_text, 80)
    assert layout.text == 'a, b'
    first, second = layout.projections
    assert first.text == 'a' and second.text == 'b'
    assert sql_text[first.start:first.end] == 'a'
    assert sql_text[second.start:second.end] == 'b'
    assert layout.projection_at(0, 0) is first
    assert layout.projection_at(0, 3) is second
    # The separator space never claims to be a field.
    assert layout.projection_at(0, 2) is None


def test_wildcard_and_single_projection():
    layout = layout_select('SELECT * FROM t', 80)
    assert len(layout.projections) == 1
    assert layout.projections[0].text == '*'
    assert layout.text == '*'
    assert layout.projection_at(0, 0).index == 0


@pytest.mark.parametrize('quantifier', ['DISTINCT', 'ALL'])
def test_select_quantifiers(quantifier):
    layout = layout_select(
        f'SELECT {quantifier} a, b FROM t', 80)
    assert [p.text for p in layout.projections] == ['a', 'b']


def test_projection_without_from_clause():
    layout = layout_select('SELECT a + 1 AS next_value', 80)
    assert layout is not None
    assert layout.projections[0].text == 'a + 1 AS next_value'


def test_case_expression_is_one_projection():
    layout = layout_select(
        'SELECT CASE WHEN a = 1 THEN b END AS z FROM t', 80)
    projection = layout.projections[0]
    assert projection.text == 'CASE WHEN a = 1 THEN b END AS z'
    # Every non-separator cell points back at the single field.
    assert all(
        cell.projection_index in (None, 0) for cell in layout.cells)


def test_multiline_source_expression_offsets():
    sql_text = 'select a,\n  b + c as total\nfrom t'
    layout = layout_select(sql_text, 80)
    first, second = layout.projections
    assert sql_text[first.start:first.end] == 'a'
    source_span = sql_text[second.start:second.end]
    assert source_span.split() == ['b', '+', 'c', 'as', 'total']
    assert not source_span[0].isspace()
    assert not source_span[-1].isspace()


def test_wide_character_both_cells_point_to_same_projection():
    layout = layout_select("SELECT '名' AS n FROM t", 80)
    wide_cells = [cell for cell in layout.cells if cell.char == '名']
    assert len(wide_cells) == 1
    wide = wide_cells[0]
    assert wide.width == 2
    first_column = wide.column
    projection = layout.projections[0]
    assert layout.projection_at(0, first_column) is projection
    assert layout.projection_at(0, first_column + 1) is projection
    assert layout.cells_at(0, first_column + 1)[0] is wide
    # The rendered text keeps one code point for the wide character.
    assert layout.lines[0][first_column] == '名'


def test_wide_character_advances_two_columns():
    layout = layout_select("SELECT '名前' AS n FROM t", 80)
    quote, first, second, closing = (
        cell for cell in layout.cells if cell.char in ("'", '名', '前'))
    assert first.column == 1 and first.width == 2
    assert second.column == 3 and second.width == 2
    assert closing.column == 5


def test_combining_mark_is_zero_width_and_attached_to_base():
    # Explicit combining acute accent: 'e' + U+0301.
    sql_text = "SELECT 'café' AS s FROM t"
    layout = layout_select(sql_text, 80)
    base = next(cell for cell in layout.cells if cell.char == 'e')
    mark = next(cell for cell in layout.cells
                if cell.char == '́')
    assert mark.width == 0
    assert mark.column == base.column
    assert mark.projection_index == base.projection_index == 0
    assert layout.projection_at(0, base.column) is layout.projections[0]
    assert any(cell.char == '́' and cell.width == 0
               for cell in layout.cells_at(0, base.column))


def test_tab_expands_against_current_column():
    layout = layout_select("SELECT 'a\tb' AS s FROM t", 80, tab_width=4)
    assert layout.text == "'a  b' AS s"
    expanded = [cell for cell in layout.cells
                if cell.column in (2, 3) and cell.char == ' ']
    assert len(expanded) == 2
    assert all(cell.projection_index == 0 for cell in expanded)
    # A tab after a double-width character lands on the same tab stops.
    wide = layout_select("SELECT '名\tb' AS s FROM t", 80, tab_width=4)
    assert wide.text == "'名 b' AS s"
    assert wide.projection_at(0, 4) is wide.projections[0]


def test_indentation_and_separators_are_not_fields():
    layout = layout_select('SELECT alpha, beta FROM t', 10)
    assert layout.text == 'alpha,\n  beta'
    indentation = [cell for cell in layout.cells
                   if cell.line == 1 and cell.column < 2]
    assert indentation and all(
        cell.char == ' ' and cell.projection_index is None
        for cell in indentation)
    assert layout.projection_at(1, 0) is None
    # On a line that fits, the space following the comma is separator
    # whitespace and must not masquerade as a field.
    wide = layout_select('SELECT alpha, beta FROM t', 80)
    separator = next(cell for cell in wide.cells
                     if cell.char == ' ' and cell.projection_index is None)
    assert wide.projection_at(separator.line, separator.column) is None


def test_every_rendered_line_respects_width():
    layout = layout_select(
        'SELECT alpha, beta, gamma, delta FROM t', 12)
    for line in layout.lines:
        assert display_width(line) <= 12


def test_wrapping_keeps_projections_intact_when_possible():
    # At width 7 a greedy layout would put "xx" on the first line and tear
    # the second projection apart; the planner must prefer the break
    # before that projection even though both options use two lines.
    layout = layout_select('SELECT a, xx yy FROM t', 7)
    assert layout.text == 'a,\n  xx yy'
    assert all(
        run.line == 1 or run.projection_index == 0
        for run in layout.runs())


def test_avoiding_a_split_outranks_line_count():
    # Packing "bb" onto the first line would allow two lines total but
    # tear the second projection apart; the planner spends a third line
    # so every projection stays on a single line.
    layout = layout_select('SELECT aaaa, bb cc, dd FROM t', 9)
    assert len(layout.lines) == 3
    projection_lines = {}
    for run in layout.runs():
        projection_lines.setdefault(
            run.projection_index, set()).add(run.line)
    assert all(len(lines) == 1 for lines in projection_lines.values())
    assert layout.text == 'aaaa,\n  bb cc,\n  dd'


def test_wrapping_minimizes_line_count():
    layout = layout_select('SELECT one, two, three FROM t', 12)
    # Three fields cannot share one line, but two lines must suffice.
    assert len(layout.lines) == 2
    for line in layout.lines:
        assert display_width(line) <= 12


def test_multiline_projection_maps_every_fragment_back():
    layout = layout_select(
        'SELECT amount_in_cents AS amount, currency FROM t', 20)
    assert len(layout.lines) == 2
    for line, line_text in enumerate(layout.lines):
        for column, char in enumerate(line_text):
            if char.isspace():
                continue
            projection = layout.projection_at(line, column)
            assert projection is not None
    # The first projection spans both lines and both fragments resolve
    # back to the same field.
    assert {run.projection_index for run in layout.runs()
            if run.projection_index == 0} == {0}
    assert len([run for run in layout.runs()
                if run.projection_index == 0]) == 2


def test_unbreakable_atom_overflows_and_is_preserved():
    layout = layout_select('SELECT abcdefghij, x FROM t', 6)
    assert layout.lines[0].startswith('abcdefghij,')
    overflow = layout.overflowing_runs()
    assert len(overflow) == 1
    run = overflow[0]
    assert run.projection_index == 0
    assert run.start_column == 0 and run.end_column == 11
    overflowed = layout.overflowed_cells()
    assert {cell.column for cell in overflowed} == {6, 7, 8, 9, 10}
    assert all(cell.projection_index == 0 for cell in overflowed)
    # The following projection still gets a clean continuation line.
    assert layout.lines[1] == '  x'


def test_function_call_keeps_tight_spacing():
    layout = layout_select('SELECT count(*), id FROM t', 80)
    assert layout.text == 'count(*), id'
    assert 'count (*)' not in layout.text


def test_runs_merge_wide_cells_and_break_on_indentation():
    layout = layout_select("SELECT '名前', id FROM t", 8)
    runs = layout.runs()
    first = runs[0]
    assert first.projection_index == 0
    assert first.start_column == 0 and first.end_column == 7
    assert all(run.line == 1 for run in runs[1:])


def test_unsupported_statements_return_none():
    unsupported = [
        'UPDATE t SET a = 1',
        'WITH cte AS (SELECT 1) SELECT a FROM cte',
        'SELECT a UNION SELECT b',
        'INSERT INTO t VALUES (1)',
        'SELECT a FROM t WHERE b = 1',
        'SELECT a FROM t ORDER BY a',
        'SELECT a FROM t JOIN u ON t.id = u.id',
    ]
    for sql_text in unsupported:
        assert layout_select(sql_text, 80) is None, sql_text


def test_supported_from_variants():
    assert layout_select('SELECT a FROM t', 80).text == 'a'
    assert layout_select('SELECT a FROM s.t', 80).text == 'a'
    assert layout_select('SELECT a;', 80).text == 'a'


def test_multiple_statements_are_rejected():
    with pytest.raises(ValueError):
        layout_select('SELECT a FROM t; SELECT b FROM t', 80)


@pytest.mark.parametrize('width', [0, -1])
def test_invalid_width_raises(width):
    with pytest.raises(ValueError):
        layout_select('SELECT a FROM t', width)


def test_cells_are_frozen_and_layout_is_tuple_backed():
    layout = layout_select('SELECT a, b FROM t', 80)
    cell = layout.cells[0]
    assert isinstance(cell, Cell)
    with pytest.raises(AttributeError):
        cell.column = 3
    assert isinstance(layout, ProjectionLayout)
    assert isinstance(layout.projections, tuple)
    assert isinstance(layout.cells, tuple)


def test_source_at_returns_projection_span_in_original_text():
    sql_text = 'SELECT alpha, beta FROM t'
    layout = layout_select(sql_text, 80)
    span = layout.source_at(0, 2)
    assert span == (layout.projections[0].start, layout.projections[0].end)
    assert sql_text[span[0]:span[1]] == 'alpha'
    assert layout.source_at(0, 6) is None


def test_embedded_newline_literal_forces_render_break():
    sql_text = "SELECT 'line1\nline2' AS multiline FROM t"
    layout = layout_select(sql_text, 40)
    assert layout.lines[0] == "'line1"
    assert layout.lines[1].startswith("  line2'")
    # Both fragments belong to the single projection.
    first = layout.projection_at(0, 1)
    second = layout.projection_at(1, 3)
    assert first is second is layout.projections[0]
    # The continuation indent is not a field.
    assert layout.projection_at(1, 0) is None


def test_crlf_source_is_normalized_without_extra_breaks():
    layout = layout_select('SELECT a\r\n, b FROM t', 80)
    assert layout.text == 'a, b'
    assert '\r' not in layout.text


def test_full_width_character_occupies_two_cells():
    layout = layout_select("SELECT 'Ａ' AS fullwidth FROM t", 80)
    wide = next(cell for cell in layout.cells if cell.char == 'Ａ')
    assert wide.width == 2
    assert layout.projection_at(0, wide.column) is layout.projections[0]
    assert layout.projection_at(
        0, wide.column + 1) is layout.projections[0]


def test_comment_text_maps_within_its_projection():
    layout = layout_select('SELECT a-- note\n, b FROM t', 80)
    assert [p.text for p in layout.projections] == [
        'a-- note', 'b']
    assert layout.projections[0].text.startswith('a--')


def test_bare_literal_is_a_projection():
    layout = layout_select("SELECT '名前' FROM t", 80)
    assert len(layout.projections) == 1
    assert layout.projections[0].text == "'名前'"


def test_wide_unbreakable_atom_reports_cells_past_margin():
    layout = layout_select("SELECT '名前' FROM t", 3)
    assert layout.lines[0] == "'名前'"
    overflow = layout.overflowing_runs()
    assert len(overflow) == 1
    run = overflow[0]
    assert run.projection_index == 0
    assert run.end_column == 6
    overflowed_columns = {cell.column for cell in layout.overflowed_cells()}
    assert overflowed_columns == {3, 5}
    # The first wide character still fits and resolves to the field.
    assert layout.projection_at(0, 1) is layout.projections[0]
    assert layout.projection_at(0, 2) is layout.projections[0]
