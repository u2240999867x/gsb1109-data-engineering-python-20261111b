#
# Copyright (C) 2009-2020 the sqlparse authors and contributors
# <see AUTHORS file>
#
# This module is part of python-sqlparse and is released under
# the BSD License: https://opensource.org/licenses/BSD-3-Clause

"""Explicit script mode for splitting migration scripts into batches.

Unlike :func:`sqlparse.split`, which only knows semicolons, script mode
understands the client-side directives used by migration scripts:

- ``GO`` on a line of its own (T-SQL batch separator), optionally with a
  repeat count (``GO 3``),
- ``DELIMITER <token>`` (MySQL client) switching the statement terminator,

and yields :class:`ScriptBatch` objects that carry the source range, the
delimiter policy that was actually in effect, and the number of times the
batch must be executed.  Unknown client commands (e.g. ``:setvar`` or
``\\connect``) are collected separately and never leak into batches.
"""

import bisect
import dataclasses
import re
from collections.abc import Iterator
from io import TextIOBase
from typing import IO

from sqlparse import tokens as T
from sqlparse.lexer import Lexer

__all__ = [
    'ClientCommand',
    'ScriptBatch',
    'ScriptCheckpoint',
    'ScriptResult',
    'split_script',
]

#: Default statement terminator (MySQL client default).
DEFAULT_DELIMITER = ';'

#: Delimiter policy reported for batches terminated by a GO line.
GO_DELIMITER = 'GO'

_GO_RE = re.compile(r'go(?:[ \t]+(\d+))?', re.IGNORECASE)
_DELIMITER_RE = re.compile(r'delimiter[ \t]+(\S+)', re.IGNORECASE)
_CLIENT_COMMAND_RE = re.compile(r'[\\:]\S')


@dataclasses.dataclass(frozen=True)
class ScriptBatch:
    """One executable batch of a script.

    ``start``/``end`` are character offsets into the source text such that
    ``source[start:end] == sql``.  Two batches with identical ``sql`` but
    different offsets are distinct inputs and are never merged.
    """

    sql: str
    start: int
    end: int
    delimiter: str
    count: int = 1

    def __str__(self) -> str:
        return self.sql


@dataclasses.dataclass(frozen=True)
class ClientCommand:
    """An unknown client command line, kept out of the batches."""

    text: str
    start: int
    end: int


@dataclasses.dataclass(frozen=True)
class ScriptCheckpoint:
    """Acknowledged progress: ``executed`` runs of ``batches[batch]``."""

    batch: int
    executed: int = 0


class ScriptResult:
    """Batches and client commands produced by :func:`split_script`."""

    def __init__(
        self,
        batches: list[ScriptBatch],
        client_commands: list[ClientCommand],
        delimiter: str = DEFAULT_DELIMITER,
    ):
        self.batches = tuple(batches)
        self.client_commands = tuple(client_commands)
        #: Terminator policy in effect at the end of the script.
        self.delimiter = delimiter

    def __iter__(self) -> Iterator[ScriptBatch]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)

    def checkpoint(self, batch: int, executed: int = 0) -> ScriptCheckpoint:
        """Create a checkpoint after *executed* runs of ``batches[batch]``."""
        cp = ScriptCheckpoint(batch=batch, executed=executed)
        self._validate(cp)
        return cp

    def resume(self, checkpoint: ScriptCheckpoint) -> Iterator[ScriptBatch]:
        """Yield the batches still owed after *checkpoint*.

        The batch the checkpoint points at is yielded with its remaining
        execution count (already acknowledged runs are not repeated), all
        later batches are yielded unchanged with their recorded delimiter
        policy, so a ``DELIMITER`` switch is never lost on resume.
        """
        self._validate(checkpoint)
        idx = checkpoint.batch
        if idx == len(self.batches):
            return iter(())
        current = self.batches[idx]
        remaining = current.count - checkpoint.executed
        following = self.batches[idx + 1:]
        if not remaining:
            return iter(following)
        return iter((dataclasses.replace(current, count=remaining),)
                    + following)

    def _validate(self, checkpoint: ScriptCheckpoint) -> None:
        idx = checkpoint.batch
        if not 0 <= idx <= len(self.batches):
            raise ValueError(f'checkpoint batch {idx} out of range')
        count = self.batches[idx].count if idx < len(self.batches) else 0
        if not 0 <= checkpoint.executed <= count:
            raise ValueError(
                f'checkpoint executed {checkpoint.executed} out of range '
                f'for batch {idx} with count {count}')


def _to_text(sql: str | bytes | IO[str], encoding: str | None) -> str:
    if isinstance(sql, TextIOBase):
        sql = sql.read()
    if isinstance(sql, str):
        return sql
    if isinstance(sql, bytes):
        if encoding:
            return sql.decode(encoding)
        try:
            return sql.decode('utf-8')
        except UnicodeDecodeError:
            return sql.decode('unicode-escape')
    raise TypeError(f'Expected text or file-like object, got {type(sql)!r}')


def _protected_spans(text: str) -> tuple[list[int], list[int]]:
    """Sorted (starts, ends) of string and comment token spans."""
    starts: list[int] = []
    ends: list[int] = []
    offset = 0
    for ttype, value in Lexer.get_default_instance().get_tokens(text):
        end = offset + len(value)
        if ttype in T.String or ttype in T.Comment:
            starts.append(offset)
            ends.append(end)
        offset = end
    return starts, ends


class _ScriptSplitter:
    def __init__(self, text: str):
        self.text = text
        self.span_starts, self.span_ends = _protected_spans(text)
        self.batches: list[ScriptBatch] = []
        self.client_commands: list[ClientCommand] = []
        self.delimiter = DEFAULT_DELIMITER

    def _is_protected(self, start: int, end: int) -> bool:
        """True if [start, end) intersects a string or comment span."""
        idx = bisect.bisect_right(self.span_starts, start) - 1
        return (idx >= 0 and self.span_ends[idx] > start) or (
            idx + 1 < len(self.span_starts)
            and self.span_starts[idx + 1] < end)

    def _is_covered(self, pos: int) -> bool:
        """True if *pos* lies inside a string or comment span."""
        idx = bisect.bisect_right(self.span_starts, pos) - 1
        return idx >= 0 and self.span_ends[idx] > pos

    def _add_batch(self, start: int, end: int, delimiter: str,
                   count: int) -> None:
        text = self.text
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            self.batches.append(ScriptBatch(
                sql=text[start:end], start=start, end=end,
                delimiter=delimiter, count=count))

    def _flush_terminated(self, start: int, end: int) -> None:
        """Emit batches for [start, end) split on the current delimiter."""
        text = self.text
        delim = self.delimiter
        pos = start
        found = text.find(delim, pos, end)
        while found != -1:
            if not self._is_protected(found, found + len(delim)):
                self._add_batch(pos, found, delim, 1)
                pos = found + len(delim)
                found = text.find(delim, pos, end)
            else:
                found = text.find(delim, found + 1, end)
        # Unterminated tail still belongs to the manifest.
        self._add_batch(pos, end, delim, 1)

    def _directive(self, line: str, content_start: int):
        """Classify a line, or return None if it is ordinary SQL."""
        # A directive keyword must start outside strings and comments;
        # quoted text later in the line (e.g. :setvar X 'v') is fine.
        if not line or self._is_covered(content_start):
            return None
        match = _GO_RE.fullmatch(line)
        if match:
            count = int(match.group(1)) if match.group(1) else 1
            return ('go', count)
        match = _DELIMITER_RE.fullmatch(line)
        if match:
            return ('delimiter', match.group(1))
        if _CLIENT_COMMAND_RE.match(line):
            return ('command', line)
        return None

    def split(self) -> ScriptResult:
        text = self.text
        pending_start = 0
        offset = 0
        for line in text.splitlines(keepends=True):
            line_end = offset + len(line)
            stripped = line.strip()
            leading = len(line) - len(line.lstrip())
            content_start = offset + leading
            content_end = content_start + len(stripped)
            directive = self._directive(stripped, content_start)
            if directive is None:
                offset = line_end
                continue
            kind, payload = directive
            if kind == 'go':
                # GO terminates the whole pending text as a single batch,
                # executed ``payload`` times (sqlcmd semantics).
                self._add_batch(pending_start, offset, GO_DELIMITER, payload)
            else:
                # DELIMITER and unknown client commands flush pending SQL
                # with the policy in effect before the line.
                self._flush_terminated(pending_start, offset)
                if kind == 'delimiter':
                    self.delimiter = payload
                else:
                    self.client_commands.append(ClientCommand(
                        text=payload, start=content_start,
                        end=content_end))
            pending_start = line_end
            offset = line_end
        self._flush_terminated(pending_start, len(text))
        return ScriptResult(self.batches, self.client_commands,
                            self.delimiter)


def split_script(
    sql: str | bytes | IO[str], encoding: str | None = None
) -> ScriptResult:
    """Split *sql* in explicit script mode and return a :class:`ScriptResult`.

    Handles ``GO`` lines (optionally with a repeat count), MySQL
    ``DELIMITER`` switches and collects unknown client commands separately.
    Same-named text inside strings, procedure bodies and comments stays
    part of the surrounding SQL.

    :param sql: A string (or bytes/file-like object) with the script.
    :param encoding: The encoding of the script (optional).
    :returns: A :class:`ScriptResult` with batches and client commands.
    """
    return _ScriptSplitter(_to_text(sql, encoding)).split()
