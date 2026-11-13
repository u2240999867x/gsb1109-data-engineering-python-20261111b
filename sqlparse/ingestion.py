"""Turn SQL dumps into a row manifest.

The functions here interpret ``INSERT`` statements using a single quoting
profile and report three things at once:

* the actual target table of every row,
* the raw source range (character offsets and line/column) of every column,
* the lines that could not be ingested, kept apart as residual lines.

They are the profile-aware counterpart of :func:`sqlparse.split` and
:func:`sqlparse.parse`: splitting, formatting followed by parsing, and
reading a stream directly all go through the same profile, so they produce
the same row list.  Only the offsets differ when the text was reformatted.
"""

import re
from dataclasses import dataclass, field, replace
from io import TextIOBase
from typing import Any

from sqlparse import profiles, quotes, sql
from sqlparse import tokens as T

__all__ = [
    'BoundaryDiff',
    'ColumnSpan',
    'IngestManifest',
    'IngestStream',
    'IngestedRow',
    'ResidualLine',
    'StatementBoundary',
    'compare_profiles',
    'ingest',
    'ingest_stream',
    'statement_boundaries',
]

# Prefixes that may grow into a delimiter when the next chunk arrives.
_DOLLAR_PREFIX = re.compile(
    r'\$(?:[_A-ZÀ-Ü][\w$]*)?\$?', re.IGNORECASE)
_DOLLAR_DELIMITER = re.compile(
    r'\$(?:[_A-ZÀ-Ü]\w*)?\$', re.IGNORECASE)


@dataclass(frozen=True)
class ColumnSpan:
    """One column value of an ingested row.

    ``start`` and ``end`` are exclusive character offsets into the text that
    was analyzed.  ``line`` and ``column`` are 1-based positions of ``start``.
    ``raw`` is the exact source slice including quotes; ``value`` is the
    decoded value according to the quoting profile.
    """

    index: int
    name: str | None
    raw: str
    value: Any
    start: int
    end: int
    line: int
    column: int


@dataclass(frozen=True)
class IngestedRow:
    """A complete ``INSERT`` row."""

    index: int
    table: str
    schema: str | None
    statement_index: int
    line: int
    columns: tuple[ColumnSpan, ...]

    @property
    def values(self):
        """Decoded values in column order."""
        return tuple(column.value for column in self.columns)


@dataclass(frozen=True)
class ResidualLine:
    """A source line that was not ingested.

    ``reason`` is one of ``unsupported_statement``, ``no_values``,
    ``lexical_error``, ``column_count_mismatch`` or ``empty_row``.
    """

    text: str
    line: int
    start: int
    end: int
    reason: str
    statement_index: int


@dataclass(frozen=True)
class StatementBoundary:
    """Source range of one parsed statement."""

    index: int
    start: int
    end: int
    line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class BoundaryDiff:
    """Difference in statement boundaries between two quoting profiles.

    ``first_differing_line`` is the 1-based source line where the two
    readings start to disagree, or ``None`` when the boundaries are equal.
    """

    profile_a: str
    profile_b: str
    first_differing_line: int | None
    statement_count_a: int
    statement_count_b: int
    detail: str | None = None

    @property
    def same(self):
        return self.first_differing_line is None

    @property
    def differs(self):
        return not self.same


@dataclass(frozen=True)
class IngestManifest:
    """Result of :func:`ingest`."""

    profile: str
    source: str | None
    text: str
    rows: tuple[IngestedRow, ...]
    residual: tuple[ResidualLine, ...]
    statements: tuple[StatementBoundary, ...] = field(default=())

    def row_list(self):
        """Return ``(table, values)`` tuples, the stable row list."""
        return tuple((row.table, row.values) for row in self.rows)

    @property
    def quarantined(self):
        """Alias for :attr:`residual`."""
        return self.residual

    def to_dict(self):
        """Return a JSON-serialisable summary of the manifest."""
        return {
            'profile': self.profile,
            'source': self.source,
            'rows': [
                {
                    'index': row.index,
                    'table': row.table,
                    'schema': row.schema,
                    'line': row.line,
                    'values': list(row.values),
                    'columns': [
                        {
                            'index': column.index,
                            'name': column.name,
                            'raw': column.raw,
                            'value': column.value,
                            'start': column.start,
                            'end': column.end,
                            'line': column.line,
                            'column': column.column,
                        }
                        for column in row.columns
                    ],
                }
                for row in self.rows
            ],
            'residual': [
                {
                    'text': line.text,
                    'line': line.line,
                    'start': line.start,
                    'end': line.end,
                    'reason': line.reason,
                }
                for line in self.residual
            ],
        }


class _LineIndex:
    """Offsets at which each 1-based source line starts."""

    __slots__ = ('_position', 'starts')

    def __init__(self):
        self.starts = [0]
        self._position = 0

    def append(self, chunk):
        chunk = str(chunk)
        position = self._position
        length = len(chunk)
        index = 0
        while index < length:
            char = chunk[index]
            if char == '\r':
                step = 2 if chunk.startswith('\r\n', index) else 1
                index += step
                position += step
                self.starts.append(position)
            elif char == '\n':
                index += 1
                position += 1
                self.starts.append(position)
            else:
                index += 1
                position += 1
        self._position = position

    def position(self, offset):
        line = 1
        low, high = 0, len(self.starts) - 1
        while low <= high:
            middle = (low + high) // 2
            if self.starts[middle] <= offset:
                line = middle + 1
                low = middle + 1
            else:
                high = middle - 1
        return line, offset - self.starts[line - 1] + 1

    def line_start(self, line):
        return self.starts[line - 1] if line <= len(self.starts) else None


class _SourceMap:
    """Translate character offsets to 1-based line/column pairs."""

    __slots__ = ('lines', 'text')

    def __init__(self, text, lines=None):
        self.text = text
        self.lines = lines if lines is not None else _LineIndex()
        if lines is None:
            self.lines.append(text)

    def position(self, offset):
        return self.lines.position(min(max(offset, 0), len(self.text)))

    def line_text(self, line):
        start = self.lines.line_start(line)
        if start is None:
            return ''
        end = self.lines.line_start(line + 1)
        if end is None:
            end = len(self.text)
        return self.text[start:end].rstrip('\r\n')

    @property
    def line_count(self):
        return len(self.lines.starts)


def _parse(text, profile):
    import sqlparse

    return sqlparse.parse(text, profile=profile)


def _statement_spans(text, profile):
    """Return ``(start, end, statement)`` for each parsed statement."""
    spans = []
    cursor = 0
    for statement in _parse(text, profile):
        raw = str(statement)
        start = text.find(raw, cursor)
        if start < 0:
            stripped = raw.strip()
            start = text.find(stripped, cursor) if stripped else -1
            if start >= 0:
                raw = stripped
        if start < 0:
            start = cursor
        end = min(start + len(raw), len(text))
        spans.append((start, end, statement))
        cursor = max(end, start)
    return spans


def _leaf_offsets(statement):
    """Map leaf token ids to their offsets inside ``str(statement)``."""
    offsets = {}
    position = 0
    for leaf in statement.flatten():
        offsets[id(leaf)] = (position, position + len(leaf.value))
        position += len(leaf.value)
    return offsets


def _tokens_span(tokens_, offsets):
    """Return the source span of a sequence of sibling tokens."""
    leaves = []
    for token in tokens_:
        if token.is_group:
            leaves.extend(token.flatten())
        else:
            leaves.append(token)
    if not leaves:
        return None
    first = offsets.get(id(leaves[0]))
    last = offsets.get(id(leaves[-1]))
    if first is None or last is None:
        return None
    return first[0], last[1]


def _content_span(statement, offsets):
    """Return the span of the statement's first and last real token."""
    leaves = list(statement.flatten())
    content = [
        leaf for leaf in leaves
        if not (leaf.is_whitespace or leaf.ttype in T.Comment
                or leaf.ttype is T.Newline)
    ]
    if not content:
        return None
    return offsets[id(content[0])][0], offsets[id(content[-1])][1]


def _unquote(value, profile):
    if value is None:
        return None
    if len(value) >= 2 and value[0] == '[' and value[-1] == ']':
        return value[1:-1]
    return profiles.unquote_identifier(value, profile)


def _parenthesis_items(parenthesis):
    items = []
    for token in parenthesis.tokens[1:-1]:
        if token.is_whitespace or token.match(T.Punctuation, ','):
            continue
        if isinstance(token, sql.IdentifierList):
            items.extend(token.get_identifiers())
        else:
            items.append(token)
    return items


def _column_name(token, profile):
    if isinstance(token, (sql.Identifier, sql.Function)):
        name = token.get_name()
        if name is None:
            name = token.get_real_name()
    else:
        name = token.value
    return _unquote(name, profile)


def _target_table(statement, profile):
    """Return ``(schema, table, declared_columns)``."""
    into_idx, into = statement.token_next_by(m=(T.Keyword, 'INTO'))
    if into is None:
        return None, None, None
    _, target = statement.token_next(into_idx, skip_ws=True, skip_cm=True)
    if target is None:
        return None, None, None

    if isinstance(target, sql.Function):
        return (
            _unquote(target.get_parent_name(), profile),
            _unquote(target.get_real_name(), profile),
            [_column_name(item, profile) for item in target.get_parameters()],
        )

    if isinstance(target, sql.Identifier):
        schema = _unquote(target.get_parent_name(), profile)
        table = _unquote(target.get_real_name(), profile)
        function = target.token_next_by(i=sql.Function)[1]
        parenthesis = target.token_next_by(i=sql.Parenthesis)[1]
        columns = None
        if function is not None:
            parameters = function.get_parameters()
            columns = [_column_name(item, profile) for item in parameters]
        elif parenthesis is not None:
            columns = [
                _column_name(item, profile)
                for item in _parenthesis_items(parenthesis)
            ]
        if columns is not None and not columns:
            columns = None
        return schema, table, columns

    if isinstance(target, sql.TokenList):
        return (
            _unquote(target.get_parent_name(), profile),
            _unquote(target.get_real_name(), profile),
            None,
        )

    return None, _unquote(target.value, profile), None


def _values_parentheses(statement):
    values = statement.token_next_by(i=sql.Values)[1]
    if values is None:
        return []
    return [token for token in values.tokens
            if isinstance(token, sql.Parenthesis)]


def _split_values(parenthesis):
    """Split a row tuple into lists of top-level value tokens."""
    items = []
    current = []
    for token in parenthesis.tokens[1:-1]:
        if isinstance(token, sql.IdentifierList):
            children = token.tokens
        else:
            children = [token]
        for child in children:
            if child.match(T.Punctuation, ','):
                if current:
                    items.append(current)
                current = []
            elif child.is_whitespace:
                continue
            else:
                current.append(child)
    if current:
        items.append(current)
    return [tuple(item) for item in items]


def _decode(item, raw, profile):
    if len(item) != 1:
        return raw
    token = item[0]
    if isinstance(token, sql.TokenList):
        return raw
    ttype = token.ttype
    if ttype in T.String:
        return profiles.decode_string(raw, profile)
    if ttype in T.Number.Integer:
        try:
            return int(raw)
        except ValueError:
            return raw
    if ttype in T.Number.Float:
        try:
            return float(raw)
        except ValueError:
            return raw
    if ttype in T.Number.Hexadecimal:
        try:
            return int(raw, 16)
        except ValueError:
            return raw
    if ttype in T.Keyword:
        upper = raw.upper()
        if upper == 'NULL':
            return None
        if upper == 'TRUE':
            return True
        if upper == 'FALSE':
            return False
    if ttype in T.Name and raw[:1] in ('`', '['):
        return _unquote(raw, profile)
    return raw


def _has_lexical_error(statement):
    return any(leaf.ttype is T.Error for leaf in statement.flatten())


def _item_is_complete(item):
    """Return false if a value token contains an unterminated quote."""
    for token in item:
        if token.is_group:
            if not _item_is_complete(token.tokens):
                return False
            continue
        if token.ttype in T.String and len(token.value) >= 2:
            quote = token.value[0]
            if quote in ("'", '"') and token.value[-1] != quote:
                return False
    return True


def _statement_is_empty(statement):
    return all(
        leaf.is_whitespace
        or leaf.ttype in T.Comment
        or leaf.ttype is T.Newline
        for leaf in statement.flatten()
    )


def _residual_lines(source_map, start, end, reason, statement_index):
    """Build residual entries for the non-empty lines in [start, end)."""
    if end <= start:
        return []
    start_line = source_map.position(start)[0]
    end_line = source_map.position(end - 1)[0]
    result = []
    for line in range(start_line, end_line + 1):
        if line > source_map.line_count:
            break
        text = source_map.line_text(line)
        if not text.strip():
            continue
        offset = source_map.lines.line_start(line)
        result.append(ResidualLine(
            text=text,
            line=line,
            start=offset,
            end=offset + len(text),
            reason=reason,
            statement_index=statement_index,
        ))
    return result


def _parse_statement(statement, start, statement_index, profile, source_map):
    """Return ``(rows, residual)`` for one parsed statement."""
    rows = []
    residual = []
    if _statement_is_empty(statement):
        return rows, residual

    offsets = _leaf_offsets(statement)
    content = _content_span(statement, offsets)
    if content is None:
        return rows, []
    content_start = start + content[0]
    content_end = start + content[1]

    if statement.get_type() not in ('INSERT', 'REPLACE'):
        return rows, _residual_lines(
            source_map, content_start, content_end,
            'unsupported_statement', statement_index)

    if _has_lexical_error(statement):
        return rows, _residual_lines(
            source_map, content_start, content_end,
            'lexical_error', statement_index)

    schema, table, declared_columns = _target_table(statement, profile)
    if table is None:
        return rows, _residual_lines(
            source_map, content_start, content_end,
            'no_values', statement_index)

    parentheses = _values_parentheses(statement)
    if not parentheses:
        return rows, _residual_lines(
            source_map, content_start, content_end,
            'no_values', statement_index)

    statement_text = str(statement)
    qualified = table if schema is None else f'{schema}.{table}'
    for parenthesis in parentheses:
        items = _split_values(parenthesis)
        tuple_span = _tokens_span([parenthesis], offsets)
        first_offset = tuple_span[0] if tuple_span is not None else content[0]
        tuple_start = start + first_offset
        tuple_end = start + (tuple_span[1] if tuple_span is not None
                             else content[1])
        tuple_line, _ = source_map.position(tuple_start)
        if not items:
            residual.extend(_residual_lines(
                source_map, tuple_start, tuple_end,
                'empty_row', statement_index))
            continue
        if not all(_item_is_complete(item) for item in items):
            residual.extend(_residual_lines(
                source_map, tuple_start, tuple_end,
                'lexical_error', statement_index))
            continue
        if declared_columns is not None \
                and len(items) != len(declared_columns):
            residual.extend(_residual_lines(
                source_map, tuple_start, tuple_end,
                'column_count_mismatch', statement_index))
            continue

        columns = []
        for index, item in enumerate(items):
            span = _tokens_span(item, offsets)
            if span is None:
                continue
            raw_start, raw_end = span
            raw = statement_text[raw_start:raw_end]
            absolute_start = start + raw_start
            absolute_end = start + raw_end
            line, column = source_map.position(absolute_start)
            value = _decode(item, raw, profile)
            name = (declared_columns[index]
                    if declared_columns is not None else None)
            columns.append(ColumnSpan(
                index=index,
                name=name,
                raw=raw,
                value=value,
                start=absolute_start,
                end=absolute_end,
                line=line,
                column=column,
            ))
        rows.append(IngestedRow(
            index=0,
            table=qualified,
            schema=schema,
            statement_index=statement_index,
            line=tuple_line,
            columns=tuple(columns),
        ))
    return rows, residual


def _read_text(value, encoding=None):
    if isinstance(value, TextIOBase):
        value = value.read()
    if isinstance(value, bytes):
        value = value.decode(encoding or 'utf-8')
    return str(value)


def _build_manifest(text, active, source):
    source_map = _SourceMap(text)
    rows = []
    residual = []
    statements = []
    spans = _statement_spans(text, active)
    for index, (start, end, statement) in enumerate(spans):
        offsets = _leaf_offsets(statement)
        content = _content_span(statement, offsets)
        first = start + content[0] if content is not None else start
        last = start + content[1] if content is not None else end
        statement_start = source_map.position(first)[0]
        statement_end_line = source_map.position(max(first, last - 1))[0]
        statements.append(StatementBoundary(
            index=index,
            start=start,
            end=end,
            line=statement_start,
            end_line=statement_end_line,
            text=str(statement),
        ))
        parsed_rows, parsed_residual = _parse_statement(
            statement, start, index, active, source_map)
        rows.extend(parsed_rows)
        residual.extend(parsed_residual)
    return IngestManifest(
        profile=active.name,
        source=source,
        text=text,
        rows=tuple(
            replace(row, index=index) for index, row in enumerate(rows)
        ),
        residual=tuple(
            sorted(residual, key=lambda item: (item.line, item.start))
        ),
        statements=tuple(statements),
    )


def ingest(sql_text, profile=None, source=None, encoding=None):
    """Parse *sql_text* and return an :class:`IngestManifest`.

    :param sql_text: SQL text, bytes or a file-like object.
    :param profile: Quoting profile name or :class:`QuoteProfile`.
    :param source: Optional source label recorded in the manifest.
    :param encoding: Encoding used to decode bytes.
    """
    text = _read_text(sql_text, encoding)
    active = profiles.get_profile(profile)
    return _build_manifest(text, active, source)


def statement_boundaries(text, profile=None, encoding=None):
    """Return the statement boundaries of *text* under *profile*."""
    return list(ingest(text, profile=profile, encoding=encoding).statements)


def _first_divergence(text, boundaries_a, boundaries_b):
    """Line where the two statement lists first disagree.

    Walks the boundaries in source order.  While both lists agree, the
    compared offsets coincide.  The first pair of offsets that differs is
    the split point, and the line of the earlier of the two is the first
    source line where the readings diverge.
    """
    from sqlparse.ingestion import _SourceMap

    source_map = _SourceMap(text)
    index_a = index_b = 0
    while index_a < len(boundaries_a) and index_b < len(boundaries_b):
        first = boundaries_a[index_a]
        second = boundaries_b[index_b]
        if (first.start, first.end) == (second.start, second.end):
            index_a += 1
            index_b += 1
            continue
        offset = min(first.end, second.end)
        return source_map.position(offset)[0], index_a, first, second
    if index_a < len(boundaries_a):
        extra = boundaries_a[index_a]
        return extra.line, index_a, extra, None
    if index_b < len(boundaries_b):
        extra = boundaries_b[index_b]
        return extra.line, index_b, None, extra
    return None, None, None, None


def compare_profiles(text, profile_a, profile_b, encoding=None):
    """Compare statement boundaries under two quoting profiles.

    ``first_differing_line`` is the first source line at which the split
    points of the two readings differ -- usually the line where a
    semicolon moves into or out of a string value.
    """
    active_a = profiles.get_profile(profile_a)
    active_b = profiles.get_profile(profile_b)
    text = _read_text(text, encoding)
    boundaries_a = statement_boundaries(text, active_a)
    boundaries_b = statement_boundaries(text, active_b)
    if [(item.start, item.end) for item in boundaries_a] \
            == [(item.start, item.end) for item in boundaries_b]:
        return BoundaryDiff(
            profile_a=active_a.name,
            profile_b=active_b.name,
            first_differing_line=None,
            statement_count_a=len(boundaries_a),
            statement_count_b=len(boundaries_b),
        )
    line, index, first, second = _first_divergence(
        text, boundaries_a, boundaries_b)
    detail = f'statement {index} has no counterpart'
    if first is not None and second is not None:
        detail = (
            f'statement {index} ends at {first.end} under {active_a.name} '
            f'and at {second.end} under {active_b.name}')
    return BoundaryDiff(
        profile_a=active_a.name,
        profile_b=active_b.name,
        first_differing_line=line,
        statement_count_a=len(boundaries_a),
        statement_count_b=len(boundaries_b),
        detail=detail,
    )


def _shift_manifest(manifest, base, line_index, statement_offset,
                    row_offset):
    """Rebase a manifest parsed from a slice onto the whole stream."""
    rows = []
    for row in manifest.rows:
        columns = []
        for column in row.columns:
            line, column_number = line_index.position(base + column.start)
            columns.append(replace(
                column,
                start=base + column.start,
                end=base + column.end,
                line=line,
                column=column_number,
            ))
        first = columns[0].start if columns else base
        line, _ = line_index.position(first)
        rows.append(replace(
            row,
            index=row_offset + row.index,
            statement_index=row.statement_index + statement_offset,
            line=line,
            columns=tuple(columns),
        ))
    residual = []
    for item in manifest.residual:
        line, _ = line_index.position(base + item.start)
        residual.append(replace(
            item,
            line=line,
            start=base + item.start,
            end=base + item.end,
            statement_index=item.statement_index + statement_offset,
        ))
    statements = []
    # Statement lines were computed from the slice's content tokens, not
    # from its raw start, so shift them by the line delta instead of
    # looking up absolute offsets.
    line_delta = line_index.position(base)[0] - 1
    for boundary in manifest.statements:
        statements.append(replace(
            boundary,
            index=boundary.index + statement_offset,
            start=base + boundary.start,
            end=base + boundary.end,
            line=boundary.line + line_delta,
            end_line=boundary.end_line + line_delta,
        ))
    return rows, residual, statements


class IngestStream:
    """Incremental ingestion of one SQL stream.

    Each instance owns its own quoting state.  Two streams can therefore
    be fed alternately while both are in the middle of a quoted literal
    without borrowing each other's state.

    :param profile: Quoting profile name or :class:`QuoteProfile`.
    :param source: Optional source label stored in the final manifest.
    """

    def __init__(self, profile=None, source=None):
        self.profile = profiles.get_profile(profile)
        self.source = source
        self.rows = []
        self.residual = []
        self.statements = []
        self._text = ''
        self._line_index = _LineIndex()
        self._scan_pos = 0
        self._statement_start = 0
        self._statement_count = 0
        self._quote = quotes.QuoteState(self.profile)
        self._line_comment = False
        self._block_comment = False
        self._dollar_tag = None
        self._backtick = False
        self._backtick_escape = False
        self._paren_depth = 0
        self._pending_separator = False
        self._rows_emitted = 0
        self._finished = False

    def feed(self, chunk):
        """Consume *chunk* and return the rows completed by it."""
        if self._finished:
            raise ValueError('cannot feed a finished stream')
        chunk = _read_text(chunk)
        if not chunk:
            return []
        self._text += chunk
        self._line_index.append(chunk)
        if self._pending_separator:
            self._absorb_separator(chunk)
        return self._scan()

    def _absorb_separator(self, chunk):
        """Attach horizontal whitespace after a just-emitted statement.

        The statement splitter keeps spaces and tabs that follow a
        semicolon with that statement.  When a chunk ends right after the
        semicolon those characters have not arrived yet, so the boundary is
        extended once they do.
        """
        run = 0
        while run < len(chunk) and chunk[run] in ' \t\f\v':
            run += 1
        if run and self.statements:
            last = self.statements[-1]
            if last.end == self._statement_start:
                self.statements[-1] = replace(
                    last,
                    end=last.end + run,
                    text=last.text + chunk[:run],
                )
                self._statement_start += run
                self._scan_pos = self._statement_start
        # A chunk of nothing but whitespace may be followed by more of it,
        # so keep waiting in that case.
        self._pending_separator = run == len(chunk)

    def _scan(self):
        """Emit any statement prefix that is now known to be complete."""
        completed = []
        text = self._text
        while True:
            end = self._find_candidate(text, self._scan_pos)
            if end is None:
                return completed
            segment = text[self._statement_start:end]
            if not self._is_complete(segment):
                self._scan_pos = end
                continue
            completed.extend(self._emit(end))
            text = self._text
            if self._statement_start == len(text):
                # Nothing but horizontal whitespace can still follow this
                # semicolon; absorb it from the next chunk in feed().
                self._pending_separator = True
                return completed

    def _find_candidate(self, text, index):
        """Return the offset after the next unquoted, top-level semicolon.

        Scanner state and ``_scan_pos`` always move together: a chunk may
        end in the middle of a literal, comment or dollar-quoted body, and
        the next call has to continue exactly where this one stopped.  A
        trailing character that may be the first half of a multi-character
        delimiter is left unconsumed so the next chunk can complete it.
        """
        length = len(text)
        while index < length:
            char = text[index]

            if self._block_comment:
                close = text.find('*/', index)
                if close < 0:
                    break
                index = close + 2
                self._block_comment = False
                continue

            if self._line_comment:
                newline = text.find('\n', index)
                if newline < 0:
                    break
                index = newline + 1
                self._line_comment = False
                continue

            if self._dollar_tag is not None:
                close = text.find(self._dollar_tag, index)
                if close < 0:
                    break
                index = close + len(self._dollar_tag)
                self._dollar_tag = None
                continue

            if self._backtick:
                if self._backtick_escape:
                    self._backtick_escape = False
                    index += 1
                    continue
                if self.profile.backtick_escapes and char == '\\':
                    self._backtick_escape = True
                    index += 1
                    continue
                if char == '`':
                    if index == length - 1:
                        break
                    if text.startswith('``', index):
                        # A doubled backtick is an escaped backtick and does
                        # not close the identifier.
                        index += 2
                        continue
                    self._backtick = False
                    index += 1
                    continue
                index += 1
                continue

            if self._quote.step(char):
                index += 1
                continue

            if char == '-' and text.startswith('--', index):
                self._line_comment = True
                index += 2
                continue
            if char == '#' and text.startswith('# ', index):
                self._line_comment = True
                index += 2
                continue
            if char == '/' and text.startswith('/*', index):
                self._block_comment = True
                index += 2
                continue
            if char in '-#/' and index == length - 1:
                break
            if char == '`':
                self._backtick = True
                index += 1
                continue
            if char == '$':
                match = _DOLLAR_DELIMITER.match(text, index)
                if match is not None:
                    self._dollar_tag = match.group()
                    index = match.end()
                    continue
                tail = text[index:]
                if _DOLLAR_PREFIX.fullmatch(tail):
                    break
            if char == '(':
                self._paren_depth += 1
            elif char == ')':
                self._paren_depth = max(0, self._paren_depth - 1)
            elif char == ';' and self._paren_depth == 0:
                end = index + 1
                # The statement splitter keeps horizontal whitespace that
                # follows the semicolon with the statement, but starts a new
                # statement at a newline.  Mirror that here.
                while end < length and text[end] in ' \t\f\v':
                    end += 1
                self._scan_pos = end
                return end
            index += 1
        self._scan_pos = index
        return None

    def _is_complete(self, segment):
        """True if the splitter would end the statement at *segment*."""
        from sqlparse import lexer
        from sqlparse.engine.statement_splitter import StatementSplitter

        splitter = StatementSplitter()
        list(splitter.process(
            lexer.tokenize_with_profile(segment, self.profile)))
        return (splitter.consume_ws and bool(splitter.tokens)
                and not splitter._block_stack)

    def _emit(self, end):
        base = self._statement_start
        segment = self._text[base:end]
        manifest = ingest(segment, profile=self.profile, source=self.source)
        rows, residual, statements = _shift_manifest(
            manifest, base, self._line_index, self._statement_count,
            len(self.rows))
        for row in rows:
            self.rows.append(row)
        self.residual.extend(residual)
        self.statements.extend(statements)
        completed = list(rows)
        self._rows_emitted += len(completed)
        self._statement_count += len(manifest.statements)
        self._statement_start = end
        self._scan_pos = end
        return completed

    def finish(self):
        """Flush the remaining buffer and return newly completed rows.

        The incremental scanner is a fast path for reporting rows as soon
        as they close.  On malformed input it can pair delimiters slightly
        differently from the full lexer, so the final manifest is rebuilt
        from the complete text.  That guarantees the same rows, residual
        lines and statement boundaries as :func:`ingest` for the same
        profile, independent of how the stream was chunked.
        """
        if self._finished:
            return []
        self._finished = True
        manifest = ingest(
            self._text, profile=self.profile, source=self.source)
        completed = list(manifest.rows[self._rows_emitted:])
        self.rows = list(manifest.rows)
        self.residual = list(manifest.residual)
        self.statements = list(manifest.statements)
        self._statement_count = len(self.statements)
        self._statement_start = len(self._text)
        return completed

    def manifest(self):
        """Return the manifest collected so far."""
        if not self._finished:
            return self._partial_manifest()
        return IngestManifest(
            profile=self.profile.name,
            source=self.source,
            text=self._text,
            rows=tuple(self.rows),
            residual=tuple(
                sorted(self.residual,
                       key=lambda item: (item.line, item.start))
            ),
            statements=tuple(self.statements),
        )

    def _partial_manifest(self):
        """Return what incremental scanning has established so far."""
        residual = []
        for item in self.residual:
            start = self._line_index.line_start(item.line)
            text = ''
            end = start
            if start is not None:
                next_start = self._line_index.line_start(item.line + 1)
                if next_start is None:
                    next_start = len(self._text)
                text = self._text[start:next_start].rstrip('\r\n')
                end = start + len(text)
            residual.append(replace(
                item, text=text, start=start or 0, end=end))
        return IngestManifest(
            profile=self.profile.name,
            source=self.source,
            text=self._text,
            rows=tuple(self.rows),
            residual=tuple(
                sorted(residual, key=lambda item: (item.line, item.start))
            ),
            statements=tuple(self.statements),
        )


def ingest_stream(profile=None, source=None):
    """Create an :class:`IngestStream` for *profile*."""
    return IngestStream(profile=profile, source=source)
