#
# Copyright (C) 2009-2020 the sqlparse authors and contributors
# <see AUTHORS file>
#
# This module is part of python-sqlparse and is released under
# the BSD License: https://opensource.org/licenses/BSD-3-Clause

"""Verbatim regions and output-to-source range mapping.

A *verbatim region* marks a span of the original SQL that must survive
formatting byte for byte: neither keyword casing, whitespace stripping,
re-indentation nor newline conversion touches it.  Regions are opaque to
the grouping stage, so they may start in the middle of a ``SELECT`` list
and reach into a subquery as long as the brackets they contain stay
balanced; brackets outside a region keep participating in grouping.

The :class:`SourceMap` returned alongside the formatted text projects any
output position back onto the half-open character range of the original
input.  Projection is offset based -- the same column text occurring
several times is never located by string search -- and unmappable or
ambiguous positions raise :class:`~sqlparse.exceptions.SQLParseError`
with a line/column diagnostic.
"""

from bisect import bisect_right
from dataclasses import dataclass

from sqlparse import tokens as T
from sqlparse.exceptions import SQLParseError

#: Default inline marker pair.  The characters are not part of any ordinary
#: SQL identifier, so an unmarked statement behaves exactly as before; the
#: markers are consumed (never emitted) once a region is opened.
DEFAULT_OPEN_MARKER = '@@VBEGIN@@'
DEFAULT_CLOSE_MARKER = '@@VEND@@'


@dataclass(frozen=True, slots=True)
class Directive:
    """A resolved verbatim region in the *original* text.

    ``open_*``/``close_*`` bracket the marker text itself, ``content_*``
    the span that is actually preserved.  For an explicit range all four
    boundaries coincide with the range ends.
    """

    open_start: int
    open_end: int
    content_start: int
    content_end: int
    close_start: int
    close_end: int


@dataclass(frozen=True, slots=True)
class Segment:
    """One emitted output chunk and the source range it derives from."""

    out_start: int
    out_end: int
    src_start: int
    src_end: int
    verbatim: bool


def line_col(text, offset):
    """Return 1-based ``(line, column)`` for *offset* in *text*."""
    if offset < 0 or offset > len(text):
        raise SQLParseError(
            f'Position {offset} is outside the text (length {len(text)}).')
    line = text.count('\n', 0, offset) + 1
    column = offset - (text.rfind('\n', 0, offset) + 1) + 1
    return line, column


# ---------------------------------------------------------------------------
# Quote/comment aware scanning
# ---------------------------------------------------------------------------

def _skip_quoted(text, pos, quote):
    """Return the position just past a string/quoted identifier starting
    at *pos*.  Backslash and doubled-quote escapes are honored; an
    unterminated quote runs to the end of the text."""
    length = len(text)
    pos += 1
    while pos < length:
        char = text[pos]
        if char == '\\':
            pos += 2
            continue
        if char == quote:
            if pos + 1 < length and text[pos + 1] == quote:
                pos += 2
                continue
            return pos + 1
        pos += 1
    return length


def _dollar_end(text, pos):
    """Return the position past a dollar-quoted literal opening at *pos*,
    or ``None`` if the opener is malformed or never closed.

    Mirrors the lexer: the tag is empty (``$$...$$``) or starts with a
    letter/underscore followed by word characters, and the opener is not
    preceded by a word character, a quote or another ``$``.
    """
    length = len(text)
    if pos > 0 and (text[pos - 1].isalnum() or text[pos - 1] in '_"$'):
        return None
    close = text.find('$', pos + 1)
    if close == -1:
        return None
    tag = text[pos:close + 1]
    body = tag[1:-1]
    if body:
        if not (body[0] == '_' or body[0].isalpha()):
            return None
        if not all(ch.isalnum() or ch == '_' for ch in body):
            return None
    end = text.find(tag, close + 1)
    return None if end == -1 else end + len(tag)


def iter_code_spans(text):
    """Yield ``(start, end)`` ranges of *text* lying outside strings,
    quoted identifiers, dollar-quoted bodies and comments.

    The state machine mirrors the lexer's delimiter handling, so a marker
    hidden inside a quote or comment is treated as literal text rather
    than as a region boundary.
    """
    length = len(text)
    pos = 0
    span_start = 0
    while pos < length:
        char = text[pos]
        nxt = text[pos + 1] if pos + 1 < length else ''

        if char == '-' and nxt == '-':
            yield span_start, pos
            end = text.find('\n', pos)
            pos = length if end == -1 else end
            span_start = pos
            continue
        if char == '#' and nxt == ' ':
            yield span_start, pos
            end = text.find('\n', pos)
            pos = length if end == -1 else end
            span_start = pos
            continue
        if char == '/' and nxt == '*':
            yield span_start, pos
            end = text.find('*/', pos + 2)
            pos = length if end == -1 else end + 2
            span_start = pos
            continue
        if char in "'\"`":
            yield span_start, pos
            pos = _skip_quoted(text, pos, char)
            span_start = pos
            continue
        if char == '[':
            # Only the quoted-name shape the lexer recognizes ([...], no
            # nested brackets, not preceded by a word char or ']') hides
            # markers; a bare array index stays ordinary code.
            prev = text[pos - 1] if pos > 0 else ''
            is_quoted_name = (
                prev == ''
                or (prev != ']' and not prev.isalnum() and prev != '_'))
            close = text.find(']', pos + 1)
            inner = text[pos + 1:close] if close != -1 else None
            if (is_quoted_name and inner is not None
                    and '[' not in inner and ']' not in inner):
                yield span_start, pos
                pos = close + 1
                span_start = pos
                continue
        if char == '$':
            end = _dollar_end(text, pos)
            if end is not None:
                yield span_start, pos
                pos = end
                span_start = pos
                continue
        pos += 1
    if span_start < length:
        yield span_start, length


# ---------------------------------------------------------------------------
# Marker resolution and validation
# ---------------------------------------------------------------------------

def parse_markers(text, open_marker, close_marker):
    """Find inline region markers and return resolved :class:`Directive`
    objects in source order.  Markers inside strings or comments are
    literal text, not markers."""
    if not open_marker or not close_marker:
        raise SQLParseError('verbatim markers must be non-empty strings.')
    if open_marker == close_marker:
        raise SQLParseError(
            'verbatim open and close markers must differ.')

    directives = []
    open_stack = []
    for span_start, span_end in iter_code_spans(text):
        span = text[span_start:span_end]
        cursor = 0
        while cursor < len(span):
            open_at = span.find(open_marker, cursor)
            close_at = span.find(close_marker, cursor)
            if open_at == -1 and close_at == -1:
                break
            if close_at == -1 or (open_at != -1 and open_at < close_at):
                open_stack.append(span_start + open_at)
                cursor = open_at + len(open_marker)
            else:
                close_pos = span_start + close_at
                if not open_stack:
                    line, column = line_col(text, close_pos)
                    raise SQLParseError(
                        f'Unmatched verbatim close marker '
                        f'{close_marker!r} at line {line}, column '
                        f'{column}.')
                open_pos = open_stack.pop()
                directives.append(_build_directive(
                    text, open_pos, close_pos, open_marker, close_marker))
                cursor = close_at + len(close_marker)

    if open_stack:
        line, column = line_col(text, open_stack[-1])
        raise SQLParseError(
            f'Unclosed verbatim open marker {open_marker!r} at line '
            f'{line}, column {column}.')
    directives.sort(key=lambda d: d.open_start)
    return directives


def _build_directive(text, open_pos, close_pos, open_marker, close_marker):
    open_end = open_pos + len(open_marker)
    close_end = close_pos + len(close_marker)
    if open_end >= close_pos:
        line, column = line_col(text, open_pos)
        raise SQLParseError(
            f'Empty verbatim region at line {line}, column {column}.')
    directive = Directive(open_pos, open_end, open_end, close_pos,
                          close_pos, close_end)
    _check_balance(text, directive)
    return directive


def _check_balance(text, directive):
    """A region may cross subquery boundaries, but every bracket opened
    inside it must close inside it as well; otherwise statement splitting
    and grouping (which treat the region as opaque) would lose track of
    the nesting level."""
    content = text[directive.content_start:directive.content_end]
    opening = {'(': ')', '[': ']'}
    closing = {')': '(', ']': '['}
    stack = []
    for span_start, span_end in iter_code_spans(content):
        for offset in range(span_start, span_end):
            char = content[offset]
            if char in opening:
                stack.append((char, offset))
            elif char in closing:
                if not stack or stack[-1][0] != closing[char]:
                    source_pos = directive.content_start + offset
                    line, column = line_col(text, source_pos)
                    raise SQLParseError(
                        f'Unbalanced {char!r} inside verbatim region at '
                        f'line {line}, column {column}: a region must '
                        'not leave a bracket open across its boundary.')
                stack.pop()
    if stack:
        char, offset = stack[-1]
        source_pos = directive.content_start + offset
        line, column = line_col(text, source_pos)
        raise SQLParseError(
            f'Unclosed {char!r} inside verbatim region at line {line}, '
            f'column {column}: a region must not leave a bracket open '
            'across its boundary.')


def resolve_directives(text, markers=None, ranges=None):
    """Combine inline markers with explicit ranges into one validated,
    non-overlapping directive list in source order."""
    open_marker, close_marker = markers or (
        DEFAULT_OPEN_MARKER, DEFAULT_CLOSE_MARKER)
    directives = list(parse_markers(text, open_marker, close_marker))

    for index, item in enumerate(ranges or []):
        try:
            start, end = item
        except (TypeError, ValueError) as err:
            raise SQLParseError(
                f'Verbatim range #{index} must be a (start, end) pair, '
                f'got {item!r}.') from err
        if not isinstance(start, int) or not isinstance(end, int):
            raise SQLParseError(
                f'Verbatim range #{index} offsets must be integers, '
                f'got {item!r}.')
        if start < 0 or end > len(text) or start >= end:
            raise SQLParseError(
                f'Verbatim range #{index} {item!r} is out of bounds for '
                f'an input of length {len(text)}.')
        directive = Directive(start, start, start, end, end, end)
        _check_balance(text, directive)
        directives.append(directive)

    directives.sort(key=lambda d: d.open_start)
    for earlier, later in zip(directives, directives[1:]):
        if later.open_start < earlier.close_end:
            line, column = line_col(text, later.open_start)
            raise SQLParseError(
                f'Overlapping verbatim region at line {line}, column '
                f'{column}: regions must be disjoint.')
    return directives


# ---------------------------------------------------------------------------
# Output-to-source map
# ---------------------------------------------------------------------------

class SourceMap:
    """Projects formatted-output ranges onto the original input.

    All offsets are Python character offsets (code points), half-open.
    Use :meth:`source_for` to resolve a diagnostic position and
    :attr:`regions` for the exact spans of the verbatim regions.
    """

    def __init__(self, original, segments=None, regions=None):
        self._original = original
        self._segments = list(segments or [])
        #: ``(out_start, out_end, src_start, src_end)`` per region.
        self.regions = list(regions or [])
        self._rendered = original
        # Segments are appended in output order, so their starts are
        # sorted; the index array is materialized once and reused by every
        # projection (otherwise one lookup per output column turned into
        # a quadratic scan).
        self._starts = None

    @property
    def original(self):
        """The unmodified input SQL the map refers to."""
        return self._original

    def _index(self, out_pos):
        if self._starts is None:
            self._starts = [seg.out_start for seg in self._segments]
        idx = bisect_right(self._starts, out_pos) - 1
        if idx < 0:
            raise self._unmappable(out_pos)
        seg = self._segments[idx]
        if out_pos >= seg.out_end:
            # Position sits in a gap between sourced chunks, i.e. on
            # whitespace the formatter synthesized -- it has no source of
            # its own and a single origin cannot be assigned.
            raise self._unmappable(out_pos)
        return idx, seg

    def _unmappable(self, out_pos):
        line, column = line_col(self._rendered, out_pos)
        return SQLParseError(
            f'Output position {out_pos} (line {line}, column {column}) '
            'is not derived from the source: it lies on inserted '
            'whitespace or a line break.')

    def source_for(self, out_start, out_end=None):
        """Return the source ``(start, end)`` for an output position.

        With one argument the segment containing that position is
        returned.  With two arguments the output range may cover several
        adjacent tokens (for example a ``schema.table`` column or an
        ``a + b`` expression): they are combined into one contiguous
        source range as long as the covered segments are adjacent both in
        the output and in the source and share the same verbatim flag.
        A range crossing inserted whitespace, a verbatim boundary or any
        other source discontinuity raises an ambiguous-mapping error
        carrying the offending position.
        """
        if out_end is not None and out_end < out_start:
            raise SQLParseError(
                f'Invalid query range: end {out_end} precedes start '
                f'{out_start}.')
        first_idx, first = self._index(out_start)
        if out_end is None or out_end == out_start:
            return first.src_start, first.src_end

        last_idx, last = self._index(out_end - 1)
        previous = first
        for index in range(first_idx + 1, last_idx + 1):
            current = self._segments[index]
            contiguous_output = current.out_start == previous.out_end
            contiguous_source = current.src_start == previous.src_end
            same_kind = current.verbatim is first.verbatim
            if not (contiguous_output and contiguous_source and same_kind):
                line, column = line_col(self._rendered, out_start)
                raise SQLParseError(
                    f'Ambiguous range at output position {out_start} '
                    f'(line {line}, column {column}): the span crosses '
                    'a source discontinuity (inserted whitespace, a '
                    'verbatim boundary or independently sourced tokens).')
            previous = current
        return first.src_start, last.src_end

    def fragment_for(self, out_start, out_end=None):
        """Like :meth:`source_for` but returns the original text slice."""
        start, end = self.source_for(out_start, out_end)
        return self._original[start:end]

    def shift(self, out_delta):
        """Return a copy with every output offset moved by *out_delta*."""
        moved_segments = [
            Segment(seg.out_start + out_delta, seg.out_end + out_delta,
                    seg.src_start, seg.src_end, seg.verbatim)
            for seg in self._segments]
        moved_regions = [
            (os_ + out_delta, oe + out_delta, ss, se)
            for os_, oe, ss, se in self.regions]
        cloned = SourceMap(self._original, moved_segments, moved_regions)
        cloned._rendered = self._rendered
        return cloned

    def merge(self, other):
        """Append the segments of an already shifted map."""
        self._segments.extend(other._segments)
        self.regions.extend(other.regions)
        self._starts = None

    def bind_output(self, rendered):
        self._rendered = rendered

    def __repr__(self):
        return (f'<SourceMap regions={len(self.regions)} '
                f'segments={len(self._segments)}>')


# ---------------------------------------------------------------------------
# Serialization with a live output-to-source map
# ---------------------------------------------------------------------------

def _normalized_pieces(value, source_base):
    """Split a non-whitespace token into ``(text, src_start, src_end)``
    pieces, normalizing unquoted CR/CRLF to LF.

    The quote handling mirrors :func:`sqlparse.utils.split_unquoted_newlines`
    exactly: only single and double quotes delimit literal text, a
    backslash escapes the next character inside one, and line breaks inside
    a literal are kept untouched.  Every yielded piece therefore knows the
    source range it came from even when a two-byte CRLF becomes one LF.
    """
    length = len(value)
    pos = 0
    run_start = 0
    quote = None
    while pos < length:
        char = value[pos]
        if quote is not None:
            if char == '\\':
                pos += 2
                continue
            if char == quote:
                quote = None
            pos += 1
            continue
        if char in ("'", '"'):
            quote = char
            pos += 1
            continue
        if char in '\r\n':
            if pos > run_start:
                yield (value[run_start:pos],
                       source_base + run_start, source_base + pos)
            if char == '\r' and pos + 1 < length and value[pos + 1] == '\n':
                yield '\n', source_base + pos, source_base + pos + 2
                pos += 2
            else:
                yield '\n', source_base + pos, source_base + pos + 1
                pos += 1
            run_start = pos
            continue
        pos += 1
    if run_start < length:
        yield (value[run_start:],
               source_base + run_start, source_base + length)


class LineageSerializer:
    """Postprocess filter emitting the statement while recording where
    every sourced chunk came from.

    Verbatim tokens are emitted untouched (including CRLF and trailing
    spaces).  Ordinary whitespace has CRLF converted to LF and trailing
    spaces are dropped at each line end, which mirrors
    :class:`~sqlparse.filters.SerializerUnicode` outside the protected
    regions.  Chunks synthesized by formatting filters (indentation,
    inserted line breaks) carry no source range and become unmappable
    gaps rather than guessed origins.
    """

    def __init__(self, original=None):
        self._original = original
        self.map = None
        self._rendered = []
        self._local_maps = []

    def bind(self, original):
        self._original = original

    def process(self, stmt):
        parts = []
        segments = []
        regions = []
        out_pos = 0
        # Ordinary spaces waiting to be confirmed: a line break or the end
        # of the statement discards them (line rstrip), any real token or
        # a verbatim chunk flushes them.
        pending = []

        def flush_pending():
            nonlocal out_pos
            for chunk, src in pending:
                parts.append(chunk)
                if src is not None:
                    segments.append(Segment(
                        out_pos, out_pos + len(chunk),
                        src[0], src[1], False))
                out_pos += len(chunk)
            pending.clear()

        for token in stmt.flatten():
            value = token.value
            if not value:
                continue
            src = (token.source_start, token.source_end)

            if token.ttype is T.Verbatim:
                flush_pending()
                parts.append(value)
                segments.append(Segment(
                    out_pos, out_pos + len(value), src[0], src[1], True))
                regions.append(
                    (out_pos, out_pos + len(value), src[0], src[1]))
                out_pos += len(value)
                continue

            if token.is_whitespace:
                if src[0] is None:
                    # Whitespace synthesized by a formatting filter: it has
                    # no source of its own and must stay unmappable.
                    normalized = value.replace(
                        '\r\n', '\n').replace('\r', '\n')
                    head = 0
                    while head < len(normalized):
                        nl_at = normalized.find('\n', head)
                        if nl_at == -1:
                            if normalized[head:]:
                                pending.append(
                                    (normalized[head:], None))
                            break
                        if normalized[head:nl_at]:
                            pending.append(
                                (normalized[head:nl_at], None))
                        pending.clear()
                        parts.append('\n')
                        out_pos += 1
                        head = nl_at + 1
                    continue

                # Sourced whitespace: walk both the original value and its
                # LF-normalized form so each emitted piece, including a
                # CRLF/CR turned into a single LF, keeps its exact source
                # sub-range even though the lengths no longer match.
                value_start = src[0]
                source_cursor = 0
                head = 0
                normalized = value.replace('\r\n', '\n').replace('\r', '\n')
                while head < len(normalized):
                    nl_at = normalized.find('\n', head)
                    piece_end = len(normalized) if nl_at == -1 else nl_at
                    spaces = normalized[head:piece_end]
                    if spaces:
                        pending.append(
                            (spaces,
                             (value_start + source_cursor,
                              value_start + source_cursor + len(spaces))))
                        source_cursor += len(spaces)
                    if nl_at != -1:
                        # Drop buffered trailing spaces (line rstrip) and
                        # emit the break, sourced at the original line
                        # ending (LF, CR or CRLF).
                        pending.clear()
                        break_start = value_start + source_cursor
                        if value[source_cursor:source_cursor + 2] == '\r\n':
                            source_cursor += 2
                        else:
                            source_cursor += 1
                        parts.append('\n')
                        segments.append(Segment(
                            out_pos, out_pos + 1,
                            break_start, value_start + source_cursor,
                            False))
                        out_pos += 1
                        head = nl_at + 1
                    else:
                        head = piece_end
                continue

            flush_pending()
            # Non-whitespace tokens are normalized exactly like
            # SerializerUnicode: line breaks outside single/double quoted
            # literals become LF (CRLF/CR included), while quoted text and
            # every other byte are kept.  Each emitted piece carries the
            # matching source sub-range, so a converted break still points
            # back at its original CR/CRLF.
            if src[0] is None:
                parts.append(value)
                out_pos += len(value)
            else:
                for piece, piece_start, piece_end in _normalized_pieces(
                        value, src[0]):
                    parts.append(piece)
                    segments.append(Segment(
                        out_pos, out_pos + len(piece),
                        piece_start, piece_end, False))
                    out_pos += len(piece)

        # Statement-level rstrip: spaces with no following token are
        # discarded, matching SerializerUnicode.
        pending.clear()
        rendered = ''.join(parts)
        local_map = SourceMap(self._original, segments, regions)
        local_map.bind_output(rendered)
        self._rendered.append(rendered)
        self._local_maps.append(local_map)
        return rendered

    def build_map(self):
        """Merge the per-statement maps, shifting output offsets past the
        statements emitted earlier."""
        merged = SourceMap(self._original)
        output_offset = 0
        full_output = []
        for rendered, local_map in zip(self._rendered, self._local_maps,
                                       strict=True):
            shifted = local_map.shift(output_offset)
            merged.merge(shifted)
            full_output.append(rendered)
            output_offset += len(rendered)
        merged.bind_output(''.join(full_output))
        self.map = merged
        return merged
