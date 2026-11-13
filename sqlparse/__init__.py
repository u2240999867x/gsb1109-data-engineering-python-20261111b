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

from sqlparse import (
    cli,
    engine,
    filters,
    formatter,
    ingestion,
    profiles,
    quotes,
    sql,
    tokens,
)

__version__ = "0.6.1"
__all__ = [
    "cli",
    "compare_profiles",
    "engine",
    "filters",
    "format",
    "formatter",
    "ingest",
    "ingest_stream",
    "ingestion",
    "parse",
    "parsestream",
    "profiles",
    "quotes",
    "split",
    "sql",
    "statement_boundaries",
    "tokens",
]


def ingest(
    sql: str,
    profile: Any = None,
    source: str | None = None,
    encoding: str | None = None,
):
    """Parse *sql* into a row manifest under one quoting profile.

    See :func:`sqlparse.ingestion.ingest`.
    """
    return ingestion.ingest(
        sql, profile=profile, source=source, encoding=encoding)


def ingest_stream(profile: Any = None, source: str | None = None):
    """Create an incremental ingestion stream for *profile*.

    See :class:`sqlparse.ingestion.IngestStream`.
    """
    return ingestion.ingest_stream(profile=profile, source=source)


def compare_profiles(
    sql: str,
    profile_a: Any,
    profile_b: Any,
    encoding: str | None = None,
):
    """Compare statement boundaries under two quoting profiles.

    See :func:`sqlparse.ingestion.compare_profiles`.
    """
    return ingestion.compare_profiles(
        sql, profile_a, profile_b, encoding=encoding)


def statement_boundaries(
    sql: str,
    profile: Any = None,
    encoding: str | None = None,
):
    """Return the statement boundaries of *sql* under *profile*.

    See :func:`sqlparse.ingestion.statement_boundaries`.
    """
    return ingestion.statement_boundaries(
        sql, profile=profile, encoding=encoding)


def parse(
    sql: str,
    encoding: str | None = None,
    *,
    profile: Any = None,
) -> tuple[sql.Statement, ...]:
    """Parse sql and return a list of statements.

    :param sql: A string containing one or more SQL statements.
    :param encoding: The encoding of the statement (optional).
    :param profile: Quoting profile used for string literals.  ``None``
        keeps the historical parser behaviour, which accepts both the SQL
        standard and backslash escapes.  See :mod:`sqlparse.profiles`.
    :returns: A tuple of :class:`~sqlparse.sql.Statement` instances.
    """
    return tuple(parsestream(sql, encoding, profile=profile))


def parsestream(
    stream: str | IO[str],
    encoding: str | None = None,
    *,
    profile: Any = None,
) -> Generator[sql.Statement, None, None]:
    """Parses sql statements from file-like object.

    :param stream: A file-like object.
    :param encoding: The encoding of the stream contents (optional).
    :param profile: Quoting profile (optional).
    :returns: A generator of :class:`~sqlparse.sql.Statement` instances.
    """
    stack = engine.FilterStack()
    stack.enable_grouping()
    return stack.run(stream, encoding, profile=profile)


def format(
    sql: str,
    encoding: str | None = None,
    **options: Any,
) -> str:
    """Format *sql* according to *options*.

    Available options are documented in :ref:`formatting`.

    In addition to the formatting options this function accepts the
    keyword "encoding" which determines the encoding of the statement
    and "profile", which selects the quoting rules used to read the
    statement and to split its lines.

    :returns: The formatted SQL statement as string.
    """
    profile = options.pop("profile", None)
    stack = engine.FilterStack()
    options = formatter.validate_options(options)
    stack = formatter.build_filter_stack(stack, options)
    stack.postprocess.append(filters.SerializerUnicode(profile))
    return "".join(stack.run(sql, encoding, profile=profile))


def split(
    sql: str,
    encoding: str | None = None,
    strip_semicolon: bool = False,
    profile: Any = None,
) -> list[str]:
    """Split *sql* into single statements.

    :param sql: A string containing one or more SQL statements.
    :param encoding: The encoding of the statement (optional).
    :param strip_semicolon: If True, remove trailing semicolons
        (default: False).
    :param profile: Quoting profile (optional).
    :returns: A list of strings.
    """
    stack = engine.FilterStack(strip_semicolon=strip_semicolon)
    return [
        str(stmt).strip()
        for stmt in stack.run(sql, encoding, profile=profile)
    ]
