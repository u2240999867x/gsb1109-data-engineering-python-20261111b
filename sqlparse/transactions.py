#
# Copyright (C) 2009-2020 the sqlparse authors and contributors
# <see AUTHORS file>
#
# This module is part of python-sqlparse and is released under
# the BSD License: https://opensource.org/licenses/BSD-3-Clause

"""Incremental, transaction-aware batching of SQL import streams.

This module reads a SQL import stream chunk by chunk and yields one
:class:`TransactionBatch` per complete unit of work as soon as it has
been read, without waiting for the end of the stream.

The accepted input is limited to explicit transaction control statements
(``BEGIN``/``START TRANSACTION``, ``COMMIT``/``END``, ``ROLLBACK``,
``SAVEPOINT name`` and ``ROLLBACK TO [SAVEPOINT] name``) and plain DML
statements. A DML statement that appears outside of an explicit
transaction forms a batch of its own.

Statement splitting is aware of quoted strings, quoted identifiers,
comments and PostgreSQL dollar-quoted strings, even when such a
construct spans a chunk boundary. A ``COMMIT;`` that only appears
inside one of those constructs does not close a transaction.

Each batch reports the statements that are actually retained together
with their offsets in the original stream. Statements that were rolled
back to a savepoint are not part of the batch. If the stream ends with
an unclosed transaction, no batch is emitted for it; the offset where
the incomplete batch started is kept in
:attr:`TransactionStreamReader.pending_offset` so the caller can resume
reading from there. Batches that were already delivered remain
replayable as they are.
"""

from __future__ import annotations

import codecs
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import IO, Any

__all__ = [
    "TransactionBatch",
    "TransactionStreamReader",
    "iter_transaction_batches",
]


@dataclass
class TransactionBatch:
    """A complete, replayable unit of work read from an import stream.

    ``statements`` holds the retained statement texts and ``spans`` the
    matching ``(start, end)`` offsets into the original stream, so that
    ``stream[spans[i][0]:spans[i][1]] == statements[i]``. ``start`` and
    ``end`` delimit the whole batch in the stream, from its ``BEGIN``
    (or its single statement) through its ``COMMIT`` (or that same
    statement). Offsets are characters for text input and bytes for
    binary input. ``transactional`` is False for batches made of a
    single DML statement that appeared outside of any transaction.
    """

    statements: list[str]
    spans: list[tuple[int, int]]
    start: int
    end: int
    transactional: bool = True


_DOLLAR_TAG_RE = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$")
_INCOMPLETE_TAG_RE = re.compile(r"\$[A-Za-z0-9_]*\Z")


class _StatementScanner:
    """Incrementally splits fed text into semicolon-terminated statements.

    Quoted strings, quoted identifiers, line and block comments and
    dollar-quoted strings may span feed boundaries; semicolons inside
    them do not terminate a statement. Reported offsets are absolute
    character offsets, or byte offsets when *encoding* is given (the
    fed text being decoded from a byte stream using that encoding).
    """

    def __init__(self, encoding: str | None = None) -> None:
        self._encoding = encoding
        self._buffer = ""
        self._base = 0
        self._pos = 0
        self._stmt_start = 0
        self._state = "normal"
        self._block_depth = 0
        self._dollar_tag: str | None = None

    def feed(self, text: str) -> list[tuple[str, int, int]]:
        self._buffer += text
        return self._scan(final=False)

    def finish(self) -> list[tuple[str, int, int]]:
        events = self._scan(final=True)
        start = self._stmt_start
        end = len(self._buffer)
        segment = self._buffer[start:end]
        if segment.strip():
            first = end - len(segment.lstrip())
            events.append(
                (self._buffer[first:end], self._offset(first),
                 self._offset(end)))
        self._buffer = ""
        self._pos = 0
        self._stmt_start = 0
        return events

    def _offset(self, index: int) -> int:
        if self._encoding is None:
            return self._base + index
        return self._base + len(self._buffer[:index].encode(self._encoding))

    def _make_event(self, end: int) -> tuple[str, int, int] | None:
        segment = self._buffer[self._stmt_start:end]
        stripped = segment.lstrip()
        self._stmt_start = end
        if not stripped:
            return None
        first = end - len(stripped)
        return stripped, self._offset(first), self._offset(end)

    def _compact(self) -> None:
        start = self._stmt_start
        if start == 0:
            return
        if self._encoding is None:
            self._base += start
        else:
            self._base += len(self._buffer[:start].encode(self._encoding))
        self._buffer = self._buffer[start:]
        self._pos -= start
        self._stmt_start = 0

    def _scan(self, final: bool) -> list[tuple[str, int, int]]:
        events: list[tuple[str, int, int]] = []
        buf = self._buffer
        i = self._pos
        n = len(buf)
        state = self._state
        while i < n:
            char = buf[i]
            if state == "normal":
                if char == "'":
                    state = "quote"
                    self._dollar_tag = "'"
                    i += 1
                elif char == '"' or char == "`":
                    state = "quote"
                    self._dollar_tag = char
                    i += 1
                elif char == "-" or char == "/":
                    if i + 1 < n:
                        nxt = buf[i + 1]
                        if char == "-" and nxt == "-":
                            state = "line_comment"
                            i += 2
                        elif char == "/" and nxt == "*":
                            state = "block_comment"
                            self._block_depth = 1
                            i += 2
                        else:
                            i += 1
                    elif final:
                        i += 1
                    else:
                        break
                elif char == "$":
                    match = _DOLLAR_TAG_RE.match(buf, i)
                    if match is not None:
                        self._dollar_tag = match.group(0)
                        state = "dollar"
                        i = match.end()
                    elif not final and _INCOMPLETE_TAG_RE.match(buf, i):
                        break
                    else:
                        i += 1
                elif char == ";":
                    event = self._make_event(i + 1)
                    if event is not None:
                        events.append(event)
                    i += 1
                else:
                    i += 1
            elif state == "quote":
                quote = self._dollar_tag
                if char == quote:
                    if i + 1 < n:
                        if buf[i + 1] == quote:
                            i += 2
                        else:
                            state = "normal"
                            self._dollar_tag = None
                            i += 1
                    elif final:
                        state = "normal"
                        self._dollar_tag = None
                        i += 1
                    else:
                        break
                else:
                    i += 1
            elif state == "line_comment":
                if char == "\n":
                    state = "normal"
                i += 1
            elif state == "block_comment":
                if char == "/" or char == "*":
                    if i + 1 < n:
                        pair = buf[i:i + 2]
                        if pair == "/*":
                            self._block_depth += 1
                            i += 2
                        elif pair == "*/":
                            self._block_depth -= 1
                            if self._block_depth == 0:
                                state = "normal"
                            i += 2
                        else:
                            i += 1
                    elif final:
                        i += 1
                    else:
                        break
                else:
                    i += 1
            else:
                tag = self._dollar_tag
                idx = buf.find(tag, i)
                if idx == -1:
                    if final:
                        i = n
                    else:
                        i = max(i, n - len(tag) + 1)
                        break
                else:
                    i = idx + len(tag)
                    state = "normal"
                    self._dollar_tag = None
        self._state = state
        self._pos = i
        self._compact()
        return events


_BEGIN_RE = re.compile(r"(?:BEGIN|START)\b", re.IGNORECASE)
_COMMIT_RE = re.compile(r"(?:COMMIT|END)\b", re.IGNORECASE)
_SAVEPOINT_RE = re.compile(r"SAVEPOINT\s+", re.IGNORECASE)
_ROLLBACK_TO_RE = re.compile(
    r"ROLLBACK(?:\s+WORK)?\s+TO(?:\s+SAVEPOINT)?\s+", re.IGNORECASE)
_ROLLBACK_RE = re.compile(r"ROLLBACK\b", re.IGNORECASE)
_NAME_RE = re.compile(r'"((?:[^"]|"")*)"|([^\s;]+)')


def _code_prefix(text: str) -> str:
    pos = 0
    end = len(text)
    while pos < end:
        if text[pos].isspace():
            pos += 1
        elif text.startswith("--", pos):
            newline = text.find("\n", pos)
            pos = end if newline == -1 else newline + 1
        elif text.startswith("/*", pos):
            close = text.find("*/", pos + 2)
            pos = end if close == -1 else close + 2
        else:
            break
    return text[pos:]


def _parse_name(code: str, pos: int) -> str:
    match = _NAME_RE.match(code, pos)
    if match is None:
        raise ValueError(f"missing savepoint name in {code!r}")
    quoted, bare = match.groups()
    if quoted is not None:
        return quoted.replace('""', '"')
    return bare.lower()


def _classify(code: str) -> tuple[str, str | None]:
    if _BEGIN_RE.match(code):
        return "begin", None
    if _COMMIT_RE.match(code):
        return "commit", None
    match = _ROLLBACK_TO_RE.match(code)
    if match is not None:
        return "rollback_to", _parse_name(code, match.end())
    if _ROLLBACK_RE.match(code):
        return "rollback", None
    match = _SAVEPOINT_RE.match(code)
    if match is not None:
        return "savepoint", _parse_name(code, match.end())
    return "dml", None


class TransactionStreamReader:
    """Single-pass iterable of :class:`TransactionBatch` objects.

    Reads *stream* (a string, bytes, a file-like object or an iterable
    of chunks) incrementally and yields each completed batch as soon as
    its closing statement has been read. Iterating lazily: stopping
    early does not pull any further data from the source.

    After exhaustion, :attr:`pending_offset` holds the offset where an
    unclosed transaction started, or None if the stream ended cleanly.
    """

    def __init__(
        self,
        stream: str | bytes | IO[Any] | Iterable[Any],
        encoding: str | None = None,
        chunk_size: int = 8192,
    ) -> None:
        self._stream = stream
        self._chunk_size = chunk_size
        self._encoding = encoding
        self.pending_offset: int | None = None
        self._tx_start: int | None = None
        self._retained: list[tuple[str, int, int]] = []
        self._savepoints: list[tuple[str, int]] = []

    def __iter__(self) -> Iterator[TransactionBatch]:
        for text, start, end in self._iter_statements():
            batch = self._handle(text, start, end)
            if batch is not None:
                yield batch
        if self._tx_start is not None:
            self.pending_offset = self._tx_start

    def _chunks(self) -> Iterator[Any]:
        stream = self._stream
        if isinstance(stream, (str, bytes, bytearray)):
            for pos in range(0, len(stream), self._chunk_size):
                yield stream[pos:pos + self._chunk_size]
        elif hasattr(stream, "read"):
            while True:
                chunk = stream.read(self._chunk_size)
                if not chunk:
                    return
                yield chunk
        else:
            yield from stream

    def _iter_statements(self) -> Iterator[tuple[str, int, int]]:
        scanner: _StatementScanner | None = None
        decoder: codecs.IncrementalDecoder | None = None
        for chunk in self._chunks():
            if scanner is None:
                binary = isinstance(chunk, (bytes, bytearray))
                encoding = self._encoding or ("utf-8" if binary else None)
                if binary:
                    decoder = codecs.getincrementaldecoder(encoding)()
                scanner = _StatementScanner(encoding if binary else None)
            if decoder is not None:
                chunk = decoder.decode(bytes(chunk))
            if chunk:
                yield from scanner.feed(chunk)
        if scanner is None:
            return
        if decoder is not None:
            tail = decoder.decode(b"", final=True)
            if tail:
                yield from scanner.feed(tail)
        yield from scanner.finish()

    def _reset_transaction(self) -> None:
        self._tx_start = None
        self._retained = []
        self._savepoints = []

    def _rollback_to(self, name: str) -> None:
        for idx in range(len(self._savepoints) - 1, -1, -1):
            if self._savepoints[idx][0] == name:
                keep = self._savepoints[idx][1]
                del self._retained[keep:]
                del self._savepoints[idx + 1:]
                return
        raise ValueError(f"ROLLBACK TO unknown savepoint: {name!r}")

    def _handle(
        self, text: str, start: int, end: int
    ) -> TransactionBatch | None:
        code = _code_prefix(text)
        if not code.strip().rstrip(";").strip():
            return None
        kind, name = _classify(code)
        if kind == "begin":
            if self._tx_start is None:
                self._tx_start = start
                self._retained = []
                self._savepoints = []
            return None
        if kind == "commit":
            if self._tx_start is None:
                return None
            batch = TransactionBatch(
                statements=[item[0] for item in self._retained],
                spans=[(item[1], item[2]) for item in self._retained],
                start=self._tx_start,
                end=end,
                transactional=True,
            )
            self._reset_transaction()
            return batch
        if kind == "rollback":
            self._reset_transaction()
            return None
        if kind == "savepoint":
            if self._tx_start is not None:
                self._savepoints.append((name, len(self._retained)))
            return None
        if kind == "rollback_to":
            if self._tx_start is not None:
                self._rollback_to(name)
            return None
        if self._tx_start is not None:
            self._retained.append((text, start, end))
            return None
        return TransactionBatch(
            statements=[text],
            spans=[(start, end)],
            start=start,
            end=end,
            transactional=False,
        )


def iter_transaction_batches(
    stream: str | bytes | IO[Any] | Iterable[Any],
    encoding: str | None = None,
    chunk_size: int = 8192,
) -> TransactionStreamReader:
    """Iterate over complete transaction batches read from *stream*.

    *stream* may be a string, bytes, a file-like object or an iterable
    of chunks. *encoding* decodes binary input (default: UTF-8) and
    *chunk_size* controls how much is read from the source at a time.
    Returns a :class:`TransactionStreamReader`.
    """
    return TransactionStreamReader(
        stream, encoding=encoding, chunk_size=chunk_size)
