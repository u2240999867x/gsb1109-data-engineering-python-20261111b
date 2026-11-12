#
# Copyright (C) 2009-2020 the sqlparse authors and contributors
# <see AUTHORS file>
#
# This module is part of python-sqlparse and is released under
# the BSD License: https://opensource.org/licenses/BSD-3-Clause

"""Explicit per-version views of MySQL-flavored SQL scripts.

Archived SQL (dump headers, report queries, migration scripts) mixes
three kinds of comments with different semantics:

* ordinary comments (``-- ...``, ``# ...``, ``/* ... */) never reach the
  server and are stripped from the version view,
* optimizer hints (``/*+ ... */``) are directives for the server and keep
  their syntactic position in the view,
* MySQL conditional execution comments (``/*!40101 ... */``) are comments
  for servers older than the encoded version and real SQL otherwise.

:class:`VersionView` materializes the effective SQL for one explicit
target version and maps every extracted projection field back to the
byte range it occupies in the original archived text.  Conditional
comments are spliced into the token stream *before* parsing, because
enabled content may contribute a comma, an alias or half of an
expression -- it cannot be treated as an independent statement.

The declared syntax is deliberately narrow: a script is analyzable when
it consists of exactly one ``SELECT`` statement (optionally preceded by
``WITH``) plus any number of ancillary ``SET``/``USE`` statements.
Anything beyond that (DDL, other DML, multiple SELECT statements, ...)
is reported as :data:`UNDETERMINED` instead of being guessed.
"""

import re
from bisect import bisect_right
from dataclasses import dataclass

from sqlparse import tokens as T
from sqlparse.engine import FilterStack
from sqlparse.sql import Comment, Identifier, IdentifierList

__all__ = [
    "OK",
    "UNDETERMINED",
    "VersionView",
    "VersionedSQL",
    "ProjectionField",
    "ProjectionResult",
    "VersionSegment",
    "ProjectionChange",
    "SegmentReport",
    "normalize_version",
    "format_version",
]

#: Status of a fully analyzed projection.
OK = "ok"
#: Status when the script uses syntax beyond the declared subset.
UNDETERMINED = "undetermined"

_CONDITIONAL_OPEN = re.compile(r"/\*!(\d*)")

# Statements that may surround the report query in an archived script.
_ANCILLARY_STATEMENTS = ("SET", "USE")

# Top-level keywords that terminate the projection list.
_PROJECTION_STOPS = ("FROM", "INTO", "UNION", "EXCEPT", "INTERSECT")


def normalize_version(version):
    """Normalize a version given as int (``50744``), string (``"5.7.44"``)
    or tuple (``(5, 7, 44)``) to the MySQL integer form ``Mmmpp``.
    """
    if isinstance(version, int):
        return version
    if isinstance(version, str):
        parts = version.split(".")
    else:
        parts = list(version)
    parts = [int(p) for p in parts]
    major, minor, patch = (parts + [0, 0, 0])[:3]
    return major * 10000 + minor * 100 + patch


def format_version(version):
    """Render a normalized integer version as ``"M.mm.pp"`` string."""
    return f"{version // 10000}.{(version // 100) % 100}.{version % 100}"


class _Piece:
    """A lexical slice of the original script."""

    __slots__ = ("kind", "start", "end", "version",
                 "content_start", "content_end")

    def __init__(self, kind, start, end, version=None,
                 content_start=None, content_end=None):
        self.kind = kind  # 'code' | 'comment' | 'hint' | 'conditional'
        self.start = start
        self.end = end
        self.version = version
        self.content_start = content_start
        self.content_end = content_end


def _split_pieces(sql):
    """Split *sql* into code and comment pieces.

    String literals and quoted names are honored so that comment
    openers inside them are not treated as comments.  Follows MySQL
    rules: ``--`` starts a comment only when followed by whitespace or
    end of text, ``#`` always does, block comments do not nest.
    """
    pieces = []
    n = len(sql)
    i = 0
    code_start = 0
    while i < n:
        ch = sql[i]
        if ch in "'\"`":
            quote = ch
            i += 1
            while i < n:
                c = sql[i]
                if c == "\\" and quote != "`":
                    i += 2
                    continue
                if c == quote:
                    if i + 1 < n and sql[i + 1] == quote:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
        elif ch == "-" and sql.startswith("--", i) and (
                i + 2 >= n or sql[i + 2] in " \t\r\n\f\v"):
            end = sql.find("\n", i + 2)
            end = n if end == -1 else end
            if code_start < i:
                pieces.append(_Piece("code", code_start, i))
            pieces.append(_Piece("comment", i, end))
            i = code_start = end
        elif ch == "#":
            end = sql.find("\n", i + 1)
            end = n if end == -1 else end
            if code_start < i:
                pieces.append(_Piece("code", code_start, i))
            pieces.append(_Piece("comment", i, end))
            i = code_start = end
        elif ch == "/" and sql.startswith("/*", i):
            close = sql.find("*/", i + 2)
            end = n if close == -1 else close + 2
            if code_start < i:
                pieces.append(_Piece("code", code_start, i))
            match = _CONDITIONAL_OPEN.match(sql, i)
            if match is not None:
                digits = match.group(1)
                content_start = match.end()
                content_end = end if close == -1 else end - 2
                pieces.append(_Piece(
                    "conditional", i, end,
                    version=int(digits) if digits else None,
                    content_start=content_start, content_end=content_end))
            elif sql.startswith("/*+", i):
                pieces.append(_Piece("hint", i, end))
            else:
                pieces.append(_Piece("comment", i, end))
            i = code_start = end
        else:
            i += 1
    if code_start < n:
        pieces.append(_Piece("code", code_start, n))
    return pieces


def _parse(sql_text):
    stack = FilterStack()
    stack.enable_grouping()
    return tuple(stack.run(sql_text, None))


def _is_comment(token):
    # Grouped comments surface as Comment instances with ttype None.
    return token.ttype in T.Comment or isinstance(token, Comment)


def _assign_spans(token, pos, spans):
    """Record the ``(start, end)`` offsets of every token of a parsed
    statement, given that leaf tokens concatenate exactly to the source.
    """
    start = pos
    if token.is_group:
        for child in token.tokens:
            pos = _assign_spans(child, pos, spans)
    else:
        pos += len(token.value)
    spans[id(token)] = (start, pos)
    return pos


@dataclass(frozen=True)
class ProjectionField:
    """One field of a SELECT projection, mapped to the original script.

    ``orig_start``/``orig_end`` are character offsets into the original
    archived SQL, ``byte_start``/``byte_end`` the corresponding UTF-8
    byte offsets.  When the field was (partly) supplied by an enabled
    conditional comment, the range covers the comment in the original
    text.
    """

    name: str
    text: str
    alias: str | None
    orig_start: int
    orig_end: int
    byte_start: int
    byte_end: int


@dataclass(frozen=True)
class ProjectionResult:
    """Projection of a script at one version.

    ``status`` is :data:`OK` when the script stays within the declared
    syntax and :data:`UNDETERMINED` otherwise (see ``reason``).
    """

    status: str
    reason: str | None
    fields: tuple[ProjectionField, ...]
    versioned: "VersionedSQL"

    @property
    def names(self):
        return tuple(field.name for field in self.fields)


@dataclass(frozen=True)
class VersionSegment:
    """A maximal version interval with a constant projection.

    ``start``/``end`` are inclusive normalized integer versions.
    """

    start: int
    end: int
    result: ProjectionResult

    @property
    def names(self):
        return self.result.names if self.result.status == OK else None


@dataclass(frozen=True)
class ProjectionChange:
    """Projection difference between two adjacent version segments.

    ``version`` is the first version of the later segment.  ``added``
    and ``removed`` hold the column names that actually differ;
    ``reordered`` flags a change in column order only.  ``undetermined``
    is set when either side of the boundary could not be analyzed.
    """

    version: int
    added: tuple[str, ...]
    removed: tuple[str, ...]
    reordered: bool
    undetermined: bool


@dataclass(frozen=True)
class SegmentReport:
    """Segments and changes of the projection over a version range."""

    segments: tuple[VersionSegment, ...]
    changes: tuple[ProjectionChange, ...]


class VersionedSQL:
    """The effective SQL of a script for one target version.

    ``sql`` is submittable to a server of that version: ordinary
    comments are removed, enabled conditional comments are inlined,
    optimizer hints keep their syntactic position.  Offsets into ``sql``
    can be mapped back to the original archived text.
    """

    __slots__ = ("sql", "version", "_runs", "_starts", "_view")

    def __init__(self, sql, version, runs, view):
        self.sql = sql
        self.version = version
        # (out_start, out_end, orig_start) runs of text copied 1:1.
        self._runs = runs
        self._starts = [run[0] for run in runs]
        self._view = view

    def _locate(self, pos):
        idx = bisect_right(self._starts, pos) - 1
        if idx < 0:
            return self._runs[0][2] if self._runs else 0
        out_start, out_end, orig_start = self._runs[idx]
        return orig_start + min(pos, out_end) - out_start

    def orig_range(self, start, end):
        """Map an offset range of :attr:`sql` to character offsets of
        the original script.  Gaps (removed comments) clamp to the
        nearest surviving original position.
        """
        return self._locate(start), self._locate(end)

    def orig_byte_range(self, start, end):
        """Like :meth:`orig_range` but returns UTF-8 byte offsets."""
        orig_start, orig_end = self.orig_range(start, end)
        return (self._view.byte_offset(orig_start),
                self._view.byte_offset(orig_end))


class VersionView:
    """An explicit version-parametrized view of an archived SQL script."""

    def __init__(self, sql):
        self.source = sql
        self._pieces = _split_pieces(sql)
        self._byte_prefix = None

    @property
    def conditional_versions(self):
        """Sorted versions encoded in the script's conditional comments."""
        return sorted({piece.version for piece in self._pieces
                       if piece.kind == "conditional"
                       and piece.version is not None})

    def byte_offset(self, char_pos):
        """UTF-8 byte offset of a character offset into the source."""
        if self._byte_prefix is None:
            prefix = [0] * (len(self.source) + 1)
            total = 0
            for idx, char in enumerate(self.source):
                total += len(char.encode("utf-8"))
                prefix[idx + 1] = total
            self._byte_prefix = prefix
        return self._byte_prefix[char_pos]

    def render(self, version=None):
        """Materialize the submittable SQL for *version*.

        ``version=None`` disables all versioned conditional comments
        (plain comment behavior); unversioned ``/*! ... */`` comments
        are always inlined.
        """
        target = None if version is None else normalize_version(version)
        parts = []
        runs = []
        pos = 0
        for piece in self._pieces:
            if piece.kind in ("code", "hint"):
                text = self.source[piece.start:piece.end]
                orig_start = piece.start
            elif piece.kind == "conditional" and (
                    piece.version is None
                    or (target is not None and target >= piece.version)):
                text = self.source[piece.content_start:piece.content_end]
                orig_start = piece.content_start
            else:
                # Dropped comments become a single space so that
                # surrounding tokens cannot fuse together.
                text = " "
                orig_start = None
            if orig_start is not None:
                runs.append((pos, pos + len(text), orig_start))
            parts.append(text)
            pos += len(text)
        return VersionedSQL("".join(parts), target, tuple(runs), self)

    def projection(self, version=None):
        """Extract the projection fields for *version*.

        Returns a :class:`ProjectionResult`; scripts beyond the declared
        syntax yield status :data:`UNDETERMINED`.
        """
        versioned = self.render(version)
        try:
            statements = _parse(versioned.sql)
        except Exception as exc:  # pragma: no cover - defensive
            return ProjectionResult(
                UNDETERMINED, f"parse failed: {exc}", (), versioned)
        selects = []
        cursor = 0
        for stmt in statements:
            text = str(stmt)
            base = versioned.sql.find(text, cursor)
            cursor = base + len(text)
            first = stmt.token_first(skip_ws=True, skip_cm=True)
            if first is None or first.match(T.Punctuation, ";"):
                # empty statement, possibly a lone semicolon left over
                # from a disabled conditional comment
                continue
            if first.ttype == T.DML and first.normalized == "SELECT" \
                    or first.ttype == T.CTE and first.normalized == "WITH":
                selects.append((stmt, base))
            elif first.normalized.upper() not in _ANCILLARY_STATEMENTS:
                return ProjectionResult(
                    UNDETERMINED,
                    f"unsupported statement starting with {first.value!r}",
                    (), versioned)
        if not selects:
            return ProjectionResult(
                UNDETERMINED, "no SELECT statement found", (), versioned)
        if len(selects) > 1:
            return ProjectionResult(
                UNDETERMINED, "multiple SELECT statements", (), versioned)
        stmt, base = selects[0]
        try:
            fields = self._extract_projection(stmt, base, versioned)
        except ValueError as exc:
            return ProjectionResult(UNDETERMINED, str(exc), (), versioned)
        if fields is None:
            return ProjectionResult(
                UNDETERMINED, "no top-level SELECT found", (), versioned)
        if not fields:
            return ProjectionResult(
                UNDETERMINED, "empty projection", (), versioned)
        return ProjectionResult(OK, None, tuple(fields), versioned)

    def projection_segments(self, low, high):
        """Segment the inclusive version range [low, high] by projection.

        Returns a :class:`SegmentReport` listing the maximal version
        segments with constant projection and, for every boundary where
        the projection actually changes, the columns that differ.
        """
        low = normalize_version(low)
        high = normalize_version(high)
        if high < low:
            raise ValueError("empty version range")
        bounds = [low]
        bounds += [v for v in self.conditional_versions if low < v <= high]
        segments = []
        for idx, start in enumerate(bounds):
            end = bounds[idx + 1] - 1 if idx + 1 < len(bounds) else high
            segments.append(VersionSegment(start, end, self.projection(start)))
        changes = []
        for prev, cur in zip(segments, segments[1:]):
            if prev.result.status != OK or cur.result.status != OK:
                changes.append(ProjectionChange(
                    cur.start, (), (), False, True))
                continue
            old, new = prev.names, cur.names
            if old == new:
                continue
            added = tuple(name for name in new if name not in old)
            removed = tuple(name for name in old if name not in new)
            reordered = not added and not removed
            changes.append(ProjectionChange(
                cur.start, added, removed, reordered, False))
        return SegmentReport(tuple(segments), tuple(changes))

    def _extract_projection(self, stmt, base, versioned):
        spans = {}
        end = _assign_spans(stmt, base, spans)
        if end > len(versioned.sql) or str(stmt) != versioned.sql[base:end]:
            raise ValueError("could not map tokens to source offsets")

        tokens = [tok for tok in stmt.tokens
                  if not (tok.is_whitespace or _is_comment(tok))]
        select_idx = None
        for idx, tok in enumerate(tokens):
            if tok.ttype == T.DML:
                if tok.normalized == "SELECT":
                    select_idx = idx
                break
            if tok.ttype == T.DDL:
                break
        if select_idx is None:
            return None

        items = []
        idx = select_idx + 1
        # Skip SELECT modifiers (DISTINCT, SQL_CALC_FOUND_ROWS, ...).
        while idx < len(tokens) and tokens[idx].ttype in T.Keyword:
            idx += 1
        while idx < len(tokens):
            tok = tokens[idx]
            if tok.ttype in T.Keyword \
                    and tok.normalized.split(" ")[0] in _PROJECTION_STOPS:
                break
            items.append(tok)
            idx += 1
        if not items:
            return []

        if len(items) == 1 and isinstance(items[0], IdentifierList):
            groups = [[ident] for ident in items[0].get_identifiers()]
        elif len(items) == 1:
            groups = [items]
        else:
            # Grouping did not merge the list (e.g. an enabled
            # conditional comment supplied only part of an item);
            # split the remaining tokens on top-level commas instead.
            groups = [[]]
            for tok in items:
                if tok.match(T.Punctuation, ","):
                    groups.append([])
                else:
                    groups[-1].append(tok)
            groups = [group for group in groups if group]

        fields = []
        for group in groups:
            first, last = group[0], group[-1]
            start = spans[id(first)][0]
            stop = spans[id(last)][1]
            text = versioned.sql[start:stop].strip()
            alias = None
            name = None
            if len(group) == 1 and isinstance(group[0], Identifier):
                alias = group[0].get_alias()
                name = alias or group[0].get_real_name()
            elif len(group) >= 2:
                # "<expr> AS alias" surviving as separate tokens.
                prev = group[-2]
                if prev.ttype in T.Keyword and prev.normalized == "AS":
                    alias = group[-1].value
                    name = alias
            if name is None:
                name = " ".join(text.split())
            orig_start, orig_end = versioned.orig_range(start, stop)
            fields.append(ProjectionField(
                name=name,
                text=text,
                alias=alias,
                orig_start=orig_start,
                orig_end=orig_end,
                byte_start=self.byte_offset(orig_start),
                byte_end=self.byte_offset(orig_end),
            ))
        return fields
