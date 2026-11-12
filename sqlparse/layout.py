#
# Copyright (C) 2009-2020 the sqlparse authors and contributors
# <see AUTHORS file>
#
# This module is part of python-sqlparse and is released under
# the BSD License: https://opensource.org/licenses/BSD-3-Clause

"""Fixed-width layout with projection mapping for SELECT statements.

The formatter in this module renders the projection list of a simple
``SELECT ... FROM ...`` statement onto a fixed-width grid and remembers,
for every display cell, which output field (projection) the cell belongs
to and where that field occurs in the original SQL text.  This lets a
fixed-width preview jump back from a clicked display column to the
corresponding output expression.

Display widths follow terminal conventions rather than Python character
counts: East-Asian wide and full-width characters occupy two cells,
zero-width combining marks occupy none, and tabs expand against the
column they are actually rendered on.
"""

import unicodedata
from dataclasses import dataclass

from sqlparse import sql, tokens as T

__all__ = [
    'Cell',
    'Projection',
    'ProjectionLayout',
    'Run',
    'layout_select',
]


def _char_width(char):
    """Display width of *char* in fixed-width cells.

    Tabs are expanded by the caller, newlines start a new line and every
    other control character is treated as zero-width.  Combining marks
    have zero width and attach to the preceding base character.
    """
    if char == '\t' or char in '\r\n':
        return 0
    if unicodedata.combining(char):
        return 0
    if unicodedata.category(char) == 'Cc':
        return 0
    return 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1


@dataclass(frozen=True)
class Projection:
    """One output field of the rendered SELECT statement."""

    index: int
    text: str
    source_token: object
    start: int
    end: int


@dataclass(frozen=True)
class Cell:
    """A single fixed-width display cell."""

    line: int
    column: int
    char: str
    width: int
    projection_index: int | None

    @property
    def projection(self):
        return self.projection_index is not None


@dataclass(frozen=True)
class Run:
    """A contiguous run of cells belonging to one projection on one line."""

    line: int
    start_column: int
    end_column: int
    projection_index: int


def _normalize_text(group):
    """Collapse whitespace runs of a token group into single spaces.

    Line breaks are preserved (as ``\\n``) because they force a render
    break, while CRLF and lone CR endings are normalized to LF.
    """
    parts = []
    for token in group.flatten():
        if token.is_whitespace:
            value = token.value.replace('\r\n', '\n').replace('\r', '\n')
            if '\n' in value:
                if parts and not parts[-1].endswith('\n'):
                    parts.append('\n')
            elif parts and not parts[-1].endswith(' '):
                parts.append(' ')
        else:
            value = token.value.replace('\r\n', '\n').replace('\r', '\n')
            parts.append(value)
    return ''.join(parts).strip()


def _split_atoms(text):
    """Split projection text into atoms that may never be torn apart.

    Whitespace runs are the only regular break points; quoted literals,
    bracketed names and balanced parenthesised expressions stay whole even
    when they contain spaces or commas.
    """
    atoms = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char.isspace():
            end = index + 1
            while end < length and text[end].isspace():
                end += 1
            atoms.append(text[index:end])
            index = end
        elif char in ('"', "'", '`'):
            quote = char
            end = index + 1
            while end < length:
                if text[end] == quote:
                    if quote == '"' and end + 1 < length and text[end + 1] == '"':
                        end += 2
                        continue
                    end += 1
                    break
                end += 1
            atoms.append(text[index:end])
            index = end
        elif char in '([':
            close = ')' if char == '(' else ']'
            depth = 0
            end = index
            while end < length:
                inner = text[end]
                if inner in ('"', "'", '`'):
                    quote = inner
                    end += 1
                    while end < length and text[end] != quote:
                        end += 1
                    end += 1
                    continue
                if inner == char:
                    depth += 1
                elif inner == close:
                    depth -= 1
                    if depth == 0:
                        end += 1
                        break
                end += 1
            atoms.append(text[index:end])
            index = end
        elif char.isalnum() or char in '_.$':
            end = index + 1
            while end < length and (text[end].isalnum() or text[end] in '_.$'):
                end += 1
            atoms.append(text[index:end])
            index = end
        else:
            atoms.append(char)
            index += 1
    return atoms


def _extract_projections(stmt, source_lengths):
    select_index, select_kw = stmt.token_next_by(t=T.Keyword.DML)
    if select_kw is None or select_kw.normalized != 'SELECT':
        return None
    if stmt.token_first(skip_cm=True) is not select_kw:
        return None

    next_index, token = stmt.token_next(select_index)
    while token is not None and token.ttype in T.Keyword \
            and token.normalized in ('DISTINCT', 'ALL'):
        next_index, token = stmt.token_next(next_index)
    if token is None or token.is_whitespace:
        return None

    if isinstance(token, sql.IdentifierList):
        groups = list(token.get_identifiers())
    elif token.ttype is T.Wildcard or isinstance(
            token, (sql.Identifier, sql.Function, sql.Case, sql.Parenthesis)):
        groups = [token]
    elif not token.is_group and not token.is_whitespace \
            and token.ttype not in T.Keyword \
            and token.ttype is not T.Punctuation:
        # A bare leaf field such as a literal, a name or a wildcard.
        groups = [token]
    else:
        return None
    if not groups or any(
            not any(not leaf.is_whitespace for leaf in group.flatten())
            for group in groups):
        return None

    # The supported subset ends on the projection list or on a single
    # ``FROM <table>`` clause; WHERE, joins, set operators and the like
    # put the statement outside the agreed SELECT subset.
    tail_index, tail = stmt.token_next(next_index)
    while tail is not None and (tail.is_whitespace
                                or tail.match(T.Punctuation, ';')):
        tail_index, tail = stmt.token_next(tail_index)
    if tail is not None:
        if not (tail.ttype in T.Keyword and tail.normalized == 'FROM'):
            return None
        table_index, table = stmt.token_next(tail_index)
        if table is None or table.is_whitespace:
            return None
        tail_index, tail = stmt.token_next(table_index)
        while tail is not None and (tail.is_whitespace
                                    or tail.match(T.Punctuation, ';')):
            tail_index, tail = stmt.token_next(tail_index)
        if tail is not None:
            return None

    projections = []
    for index, group in enumerate(groups):
        leaves = [leaf for leaf in group.flatten() if not leaf.is_whitespace]
        start = source_lengths[id(leaves[0])][0]
        end = source_lengths[id(leaves[-1])][1]
        projections.append(Projection(
            index=index,
            text=_normalize_text(group),
            source_token=group,
            start=start,
            end=end,
        ))
    return projections


class ProjectionLayout:
    """Rendered projection list plus the cell-to-field mapping."""

    def __init__(self, text, width, tab_width, projections, cells, lines):
        self.text = text
        self.width = width
        self.tab_width = tab_width
        self.projections = tuple(projections)
        self.cells = tuple(cells)
        self.lines = tuple(lines)
        # A wide cell is reachable through each column it occupies and a
        # zero-width mark through the column of its base character.
        self._cell_index = {}
        for cell in self.cells:
            occupied = range(cell.column, cell.column + cell.width) \
                if cell.width else (cell.column,)
            for column in occupied:
                self._cell_index.setdefault(
                    (cell.line, column), []).append(cell)

    def __str__(self):
        return self.text

    def cells_at(self, line, column):
        """All cells rendered at (*line*, *column*), widest first.

        A double-width character contributes one cell spanning two
        columns (so both columns resolve to it); a zero-width mark shares
        the column of its base character.  Indentation and inter-field
        separator cells are included with ``projection_index`` set to
        ``None``.
        """
        cells = self._cell_index.get((line, column), [])
        return sorted(cells, key=lambda cell: cell.width, reverse=True)

    def projection_at(self, line, column):
        """The :class:`Projection` occupying a display cell, else ``None``.

        Both cells of a double-width character resolve to the same
        projection; indentation and separator whitespace resolve to
        ``None`` and never masquerade as a field.
        """
        for cell in self.cells_at(line, column):
            if cell.projection_index is not None:
                return self.projections[cell.projection_index]
        return None

    def source_at(self, line, column):
        """The ``(start, end)`` source span at a display cell, else ``None``.

        The offsets index into the original SQL text and cover the
        projection expression found at (*line*, *column*).
        """
        projection = self.projection_at(line, column)
        if projection is None:
            return None
        return projection.start, projection.end

    def runs(self, line=None):
        """Contiguous same-projection cell runs, one per line fragment."""
        result = []
        current = None
        for cell in self.cells:
            if line is not None and cell.line != line:
                continue
            if cell.projection_index is None or cell.width == 0:
                current = None
                continue
            end_column = cell.column + cell.width
            if current is not None \
                    and current['line'] == cell.line \
                    and current['projection_index'] == cell.projection_index \
                    and current['end_column'] == cell.column:
                current['end_column'] = end_column
            else:
                current = {
                    'line': cell.line,
                    'start_column': cell.column,
                    'end_column': end_column,
                    'projection_index': cell.projection_index,
                }
                result.append(current)
        return [Run(**item) for item in result]

    def overflowing_runs(self):
        """Runs of cells extending past the configured width.

        An atom that cannot be split and does not fit on a line is kept
        intact; this method reports the runs that occupy columns beyond
        the right margin.
        """
        return [run for run in self.runs()
                if run.end_column > self.width]

    def overflowed_cells(self):
        """The individual cells occupying columns past the right margin.

        These belong to an unbreakable atom that was preserved as-is; a
        double-width cell is included when either of its two columns lies
        beyond ``width``.
        """
        return [cell for cell in self.cells
                if cell.projection_index is not None
                and cell.column + cell.width > self.width]


def _plan_breaks(content, intra_before, space_before, width, indent_width,
                 tab_width):
    """Choose line breaks via dynamic programming.

    The cost is lexicographic: the number of breaks that tear a
    projection apart is minimised first, then the number of lines.  A
    lone atom is always feasible, even when it overflows the margin.
    Segment widths are accumulated while extending each segment, which
    keeps the whole search quadratic rather than cubic.
    """
    count = len(content)
    dp = [None] * (count + 1)
    previous = [None] * (count + 1)
    dp[0] = (0, 0)
    for start in range(count):
        if dp[start] is None:
            continue
        line_number = dp[start][1]
        column = indent_width if line_number else 0
        split = 1 if start > 0 and intra_before[start] else 0
        for end in range(start + 1, count + 1):
            index = end - 1
            if index > start and space_before[index]:
                column += 1
            for char in content[index][0]:
                if char == '\t':
                    column += tab_width - column % tab_width
                else:
                    column += _char_width(char)
            if column > width and end - start > 1:
                # Width grows monotonically, so longer segments from
                # this start cannot fit either.
                break
            candidate = (dp[start][0] + split, line_number + 1)
            if dp[end] is None or candidate < dp[end]:
                dp[end] = candidate
                previous[end] = start
    if previous[count] is None:
        # Defensive: extraction rejects empty projection lists, which are
        # the only input without a feasible plan.
        return [0]
    breaks = []
    end = count
    while end > 0:
        start = previous[end]
        breaks.append(start)
        end = start
    breaks.reverse()
    return breaks


def layout_select(sql_text, width, *, indent_width=2, tab_width=8):
    """Render the SELECT projection list of *sql_text* on a fixed grid.

    :param sql_text: a SQL string containing a single SELECT statement
    :param width: target line width in display columns
    :param indent_width: continuation-line indentation in columns
    :param tab_width: tab stop distance used when expanding tabs
    :returns: a :class:`ProjectionLayout`, or ``None`` when the statement
        is not part of the supported SELECT subset
    """
    width = int(width)
    indent_width = int(indent_width)
    tab_width = int(tab_width)
    if width < 1:
        raise ValueError('width requires a positive integer')
    if indent_width < 0 or tab_width < 1:
        raise ValueError('invalid indent or tab width')

    # Imported lazily so importing this module cannot cycle with the
    # package initializer.
    from sqlparse import parse

    statements = parse(sql_text)
    if len(statements) != 1:
        raise ValueError('layout_select expects a single statement')
    stmt = statements[0]

    source_lengths = {}
    offset = 0
    for leaf in stmt.flatten():
        source_lengths[id(leaf)] = (offset, offset + len(leaf.value))
        offset += len(leaf.value)

    projections = _extract_projections(stmt, source_lengths)
    if projections is None:
        return None

    # Flatten projections into content atoms; the separating comma hugs
    # the last atom of the field it terminates.  Whitespace atoms become
    # explicit separator flags so boundaries that had no whitespace
    # (e.g. function calls) stay tight.
    content = []
    intra_before = []
    space_before = []
    for position, projection in enumerate(projections):
        pending_space = position > 0
        for atom in _split_atoms(projection.text):
            if atom.isspace():
                pending_space = True
                continue
            content.append((atom, projection.index))
            intra_before.append(
                len(content) > 1
                and content[-2][1] is not None
                and content[-2][1] == projection.index)
            space_before.append(pending_space)
            pending_space = False
        if position + 1 < len(projections):
            atom_text, projection_index = content[-1]
            content[-1] = (atom_text + ',', projection_index)

    breaks = _plan_breaks(
        content, intra_before, space_before, width, indent_width,
        tab_width)

    cells = []
    rendered_lines = []
    line = 0
    column = 0
    line_chars = []

    def emit(char, projection_index, char_width):
        cells.append(Cell(line, column, char, char_width, projection_index))
        line_chars.append(char)
        return column + char_width

    def start_line(indent):
        nonlocal line, column, line_chars
        rendered_lines.append(''.join(line_chars))
        line_chars = []
        line += 1
        column = 0
        for _ in range(indent):
            column = emit(' ', None, 1)

    segment_start = 0
    for segment_number, segment_end in enumerate(
            breaks[1:] + [len(content)]):
        if segment_number:
            start_line(indent_width)
        for index in range(segment_start, segment_end):
            atom, projection_index = content[index]
            if index > segment_start and space_before[index]:
                separator_projection = projection_index \
                    if intra_before[index] else None
                column = emit(' ', separator_projection, 1)
            for char in atom:
                if char == '\t':
                    spaces = tab_width - column % tab_width
                    for _ in range(spaces):
                        column = emit(' ', projection_index, 1)
                elif char == '\n':
                    # An embedded line break (e.g. inside a string
                    # literal) starts a fresh, indented render line so
                    # the grid coordinates stay consistent.
                    start_line(indent_width)
                else:
                    char_width = _char_width(char)
                    if char_width == 0 and column:
                        # A zero-width mark attaches to the base
                        # character in the preceding cell.
                        cells.append(Cell(
                            line, column - 1, char, 0, projection_index))
                        line_chars.append(char)
                    else:
                        column = emit(char, projection_index, char_width)
        segment_start = segment_end
    rendered_lines.append(''.join(line_chars))

    return ProjectionLayout(
        text='\n'.join(rendered_lines),
        width=width,
        tab_width=tab_width,
        projections=projections,
        cells=cells,
        lines=rendered_lines,
    )
