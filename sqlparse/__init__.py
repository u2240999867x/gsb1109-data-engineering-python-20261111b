#
# Copyright (C) 2009-2020 the sqlparse authors and contributors
# <see AUTHORS file>
#
# This module is part of python-sqlparse and is released under
# the BSD License: https://opensource.org/licenses/BSD-3-Clause

"""Parse SQL statements."""

# Setup namespace
from collections.abc import Generator
from typing import IO, Any

from sqlparse import cli, engine, filters, formatter, lineage, sql, tokens
from sqlparse.exceptions import SQLParseError
from sqlparse.lineage import (
    DEFAULT_CLOSE_MARKER,
    DEFAULT_OPEN_MARKER,
    SourceMap,
    resolve_directives,
)

__version__ = "0.6.1"
__all__ = [
    "DEFAULT_CLOSE_MARKER",
    "DEFAULT_OPEN_MARKER",
    "SourceMap",
    "cli",
    "engine",
    "filters",
    "format_lineage",
    "formatter",
    "lineage",
    "resolve_directives",
    "sql",
    "tokens",
]


def parse(
    sql: str, encoding: str | None = None
) -> tuple[sql.Statement, ...]:
    """Parse sql and return a list of statements.

    :param sql: A string containing one or more SQL statements.
    :param encoding: The encoding of the statement (optional).
    :returns: A tuple of :class:`~sqlparse.sql.Statement` instances.
    """
    return tuple(parsestream(sql, encoding))


def parsestream(
    stream: str | IO[str], encoding: str | None = None
) -> Generator[sql.Statement, None, None]:
    """Parses sql statements from file-like object.

    :param stream: A file-like object.
    :param encoding: The encoding of the stream contents (optional).
    :returns: A generator of :class:`~sqlparse.sql.Statement` instances.
    """
    stack = engine.FilterStack()
    stack.enable_grouping()
    return stack.run(stream, encoding)


def format(sql: str, encoding: str | None = None, **options: Any) -> str:
    """Format *sql* according to *options*.

    Available options are documented in :ref:`formatting`.

    In addition to the formatting options this function accepts the
    keyword "encoding" which determines the encoding of the statement.

    :returns: The formatted SQL statement as string.
    """
    stack = engine.FilterStack()
    options = formatter.validate_options(options)
    stack = formatter.build_filter_stack(stack, options)
    stack.postprocess.append(filters.SerializerUnicode())
    return "".join(stack.run(sql, encoding))


def format_lineage(
    sql: str,
    encoding: str | None = None,
    *,
    verbatim_ranges: list[tuple[int, int]] | None = None,
    verbatim_markers: tuple[str, str] | None = None,
    **options: Any,
) -> tuple[str, SourceMap]:
    """Format *sql* while keeping verbatim regions byte for byte.

    :param sql: The SQL statement(s) to format.
    :param verbatim_ranges: Explicit half-open ``(start, end)`` character
        ranges of *sql* that must be preserved exactly.
    :param verbatim_markers: Overrides the default inline marker pair
        ``(open, close)``; marker text is consumed and never emitted.
    :param options: Regular :func:`format` options (reindent,
        keyword_case, ...).  ``output_format`` is not supported.
    :returns: The formatted SQL and a :class:`SourceMap` projecting
        output ranges back onto the original input.
    """
    options = formatter.validate_options(options)
    if options.get('output_format'):
        raise SQLParseError(
            'output_format is not supported by format_lineage: python/php '
            'serialization has no source positions to map back to.')

    if isinstance(sql, bytes):
        if encoding:
            sql = sql.decode(encoding)
        else:
            try:
                sql = sql.decode('utf-8')
            except UnicodeDecodeError:
                sql = sql.decode('unicode-escape')

    directives = resolve_directives(
        sql, markers=verbatim_markers, ranges=verbatim_ranges)

    stack = engine.FilterStack()
    stack.enable_lineage(directives)
    stack = formatter.build_filter_stack(stack, options)
    serializer = lineage.LineageSerializer(sql)
    stack.postprocess.append(serializer)

    rendered = ''.join(stack.run(sql, encoding))
    source_map = serializer.build_map()
    return rendered, source_map


def split(
    sql: str, encoding: str | None = None, strip_semicolon: bool = False
) -> list[str]:
    """Split *sql* into single statements.

    :param sql: A string containing one or more SQL statements.
    :param encoding: The encoding of the statement (optional).
    :param strip_semicolon: If True, remove trailing semicolons
        (default: False).
    :returns: A list of strings.
    """
    stack = engine.FilterStack(strip_semicolon=strip_semicolon)
    return [str(stmt).strip() for stmt in stack.run(sql, encoding)]
