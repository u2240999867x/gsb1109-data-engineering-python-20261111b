"""Per-batch parsing sessions for the shared column-mapping grammar.

Two import streams share one pyparsing grammar.  The grammar's code-table
lookups interpret the same short code differently depending on the batch's
semantic version.  A :class:`ParseSession` binds one parse run to:

* a source identifier,
* a semantic version,
* the batch's code tables,

and owns a private packrat cache.  A packrat entry may therefore only be
reused when the source, semantic version and input position all match, which
the session-scoped cache guarantees.

The grammar itself is never deep-copied: sessions temporarily swap the
class-level ``ParserElement.packrat_cache`` for their own cache.  Closing a
session releases that batch's cache.
"""

from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import pyparsing as pp
from pyparsing.core import ParserElement

__all__ = [
    "CodeLookupError",
    "ParseSession",
    "current_session",
    "require_session",
    "code_lookup",
    "nested_parse",
]


class CodeLookupError(LookupError):
    """Raised when a short code is missing from the active code table."""


# The currently active session.  A ContextVar gives asyncio Tasks / explicit
# context copies their own binding; a re-entrant lock inside the session keeps
# the shared grammar safe when several threads parse inside one session.
_current_session: "contextvars.ContextVar[Optional[ParseSession]]" = (
    contextvars.ContextVar("column_import_session", default=None)
)

# pyparsing keeps packrat state on class attributes, so swapping in a private
# session cache is process-global.  This lock serializes whole session spans,
# letting sessions be used from multiple threads without cross-session cache
# leaks.  It is re-entrant so nested sessions on the same thread work.
_global_parse_lock = threading.RLock()


def current_session() -> Optional["ParseSession"]:
    """Return the session active on this context, if any."""
    return _current_session.get()


def require_session() -> "ParseSession":
    """Return the active session or raise if parsing happens outside one."""
    session = _current_session.get()
    if session is None:
        raise RuntimeError(
            "column mapping grammar must run inside a ParseSession; "
            "code-table lookups have no batch context without one"
        )
    return session


def code_lookup(table_name: str, code: Any) -> Any:
    """Parse action helper: resolve ``code`` against the active code table."""
    return require_session().lookup(table_name, code)


@dataclass
class _SessionCache:
    """Packrat cache visible to one session only.

    Implements the small protocol pyparsing expects (``get`` / ``set`` /
    ``clear`` / ``not_in_cache`` / ``size``).  Every stored key keeps the
    pyparsing tuple ``(element, instring, loc, call_pre_parse, do_actions)``,
    so entries are already scoped to a source string and input position; the
    owning :class:`ParseSession` adds the semantic-version scope.
    """

    size_limit: Optional[int] = None
    not_in_cache: object = field(default_factory=object, init=False)
    hits: int = 0
    misses: int = 0
    # While a session owns the cache, pyparsing's per-parse_string
    # reset_cache() calls must not drop entries accumulated earlier in the
    # same session.  The session flips this off on close to hard-release.
    retain: bool = False
    _store: Dict[tuple, Any] = field(default_factory=dict, init=False)

    @property
    def size(self) -> Optional[int]:
        # Mirrors the ``size`` attribute of pyparsing's own cache classes.
        return self.size_limit

    def get(self, key: tuple) -> Any:
        value = self._store.get(key, self.not_in_cache)
        if value is self.not_in_cache:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def set(self, key: tuple, value: Any) -> None:
        self._store[key] = value
        if self.size_limit is not None:
            store = self._store
            while len(store) > self.size_limit:
                # Evict the oldest entry, matching pyparsing's _FifoCache.
                store.pop(next(iter(store)))

    def clear(self) -> None:
        if self.retain:
            # Ignore pyparsing's routine reset_cache(); this cache belongs to
            # an active session whose entries must survive between parses.
            return
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def hard_clear(self) -> None:
        """Unconditionally release every entry (used when the session ends)."""
        self.retain = False
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)


class ParseSession:
    """An explicit, isolated context for one batch parse run.

    Parameters
    ----------
    source:
        Identifier of the data source (for example an import-stream name or a
        feed file).  Cache entries never cross sources.
    semantic_version:
        Batch version that selects how short codes are interpreted.  Sessions
        with different versions keep fully independent caches.
    code_tables:
        Mapping ``table name -> {short code: resolved value}``.
    cache_size_limit:
        Forwarded to :meth:`ParserElement.enable_packrat`; ``None`` selects an
        unbounded cache, an integer a FIFO cache of that size.
    """

    def __init__(
        self,
        source: str,
        semantic_version: Any,
        code_tables: Optional[Mapping[str, Mapping[Any, Any]]] = None,
        cache_size_limit: Optional[int] = None,
    ) -> None:
        self.source = source
        self.semantic_version = semantic_version
        self._code_tables: Dict[str, Mapping[Any, Any]] = {
            name: dict(table) for name, table in (code_tables or {}).items()
        }
        self._cache = _SessionCache(size_limit=cache_size_limit)
        self._lock = threading.RLock()
        self._closed = False
        self._token: Optional[contextvars.Token] = None
        self._depth = 0
        self._saved_packrat: Dict[str, Any] = {}

    # -- code tables -------------------------------------------------------

    @property
    def code_tables(self) -> Mapping[str, Mapping[Any, Any]]:
        return self._code_tables

    def add_code_table(self, name: str, table: Mapping[Any, Any]) -> None:
        self._code_tables[name] = dict(table)

    def lookup(self, table_name: str, code: Any) -> Any:
        try:
            table = self._code_tables[table_name]
        except KeyError:
            raise CodeLookupError(
                f"no code table {table_name!r} in session "
                f"(source={self.source!r}, version={self.semantic_version!r})"
            ) from None
        try:
            return table[code]
        except KeyError:
            raise CodeLookupError(
                f"code {code!r} is not defined in table {table_name!r} "
                f"for version {self.semantic_version!r} (source={self.source!r})"
            ) from None

    # -- cache introspection ----------------------------------------------

    @property
    def cache(self) -> _SessionCache:
        return self._cache

    def cache_stats(self) -> Tuple[int, int]:
        """Return ``(hits, misses)`` counted on this session's cache."""
        with self._lock:
            return self._cache.hits, self._cache.misses

    def cache_entries(self) -> Tuple[tuple, ...]:
        """Snapshot of the raw packrat keys currently held by this session."""
        with self._lock:
            return tuple(self._cache._store.keys())

    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.hard_clear()

    # -- context management ------------------------------------------------

    def __enter__(self) -> "ParseSession":
        # Hold the process-global lock for the whole session span: pyparsing's
        # packrat cache lives on a class attribute and cannot be swapped per
        # thread.  Re-entrant, so nested sessions on the same thread are fine.
        _global_parse_lock.acquire()
        with self._lock:
            if self._closed:
                _global_parse_lock.release()
                raise RuntimeError("cannot reopen a closed ParseSession")
            self._depth += 1
            if self._depth == 1:
                self._activate()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        with self._lock:
            if self._closed:
                # close() already deactivated the session and released the
                # matching lock acquisition inside the block.
                return
            self._depth -= 1
            if self._depth == 0:
                self._deactivate()
                # Session over: release this batch's cache.
                self._cache.hard_clear()
        _global_parse_lock.release()

    def close(self) -> None:
        """Release the session and its batch cache outside a ``with`` block."""
        with self._lock:
            if self._closed:
                return
            # Each nesting level of ``with`` acquired the global lock once.
            releases = self._depth
            if releases:
                self._depth = 0
                self._deactivate()
            self._cache.hard_clear()
            self._closed = True
            self._code_tables.clear()
        for _ in range(releases):
            _global_parse_lock.release()

    def _activate(self) -> None:
        pe = ParserElement
        self._saved_packrat = {
            "enabled": pe._packratEnabled,
            "cache": pe.packrat_cache,
            "parse": pe._parse,
            "left_recursion": pe._left_recursion_enabled,
            "recursion_memos": pe.recursion_memos,
        }
        # Packrat and bounded left-recursion are mutually exclusive.
        pe.enable_packrat(self._cache.size_limit, force=True)
        # Swap the freshly enabled global cache for this session's private one.
        self._cache.retain = True
        pe.packrat_cache = self._cache
        self._token = _current_session.set(self)

    def _deactivate(self) -> None:
        pe = ParserElement
        if self._token is not None:
            _current_session.reset(self._token)
            self._token = None
        saved = self._saved_packrat
        if saved:
            with pe.packrat_cache_lock:
                pe._left_recursion_enabled = saved["left_recursion"]
                pe._packratEnabled = saved["enabled"]
                pe.packrat_cache = saved["cache"]
                pe._parse = saved["parse"]
                pe.recursion_memos = saved["recursion_memos"]
        self._saved_packrat = {}


def nested_parse(
    element: pp.ParserElement,
    instring: str,
    *,
    parse_all: bool = False,
) -> pp.ParseResults:
    """Parse an attached/child table from inside a parent-row parse action.

    The parse runs on the active session, so it reuses (and fills) the same
    session-scoped packrat cache.  Any failure is raised to the caller, which
    can catch it and let the outer parse continue: already computed outer
    fields are untouched because pyparsing only appends child tokens to the
    parent result *after* this call returns successfully, and the session
    cache only stores results or exceptions belonging to this source and
    version.
    """
    require_session()
    return element.parse_string(instring, parse_all=parse_all)
