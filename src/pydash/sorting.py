"""
Partition-merge sorting policies.

This module implements a sort policy that is suitable for merging partitioned
ingestion batches (e.g. a supplier streaming a ``quantity`` column in batches
where values may be integers, text, ``NaN``, explicit nulls, or missing
altogether). Each column of a :class:`SortPolicy` declares its own type zones,
direction, and null placement. Every partition can independently generate
persistable sort keys, and another process can merge and resume pagination
using only those keys plus the global ingestion sequence number.

.. versionadded:: 8.2.0
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import hashlib
import json
import typing as t

from .exceptions import StaleCursorError
from .helpers import NUMBER_TYPES, UNSET, base_get


__all__ = (
    "ColumnSortSpec",
    "KeyedRecord",
    "MergeStore",
    "Page",
    "SortPolicy",
)


ZONE_NULL = "null"
ZONE_NUMBER = "number"
ZONE_TEXT = "text"

ASC = "asc"
DESC = "desc"
FIRST = "first"
LAST = "last"

_CURSOR_FORMAT_VERSION = 1
_CODEPOINT_CEILING = 0x10FFFF


def _invert_text(text: str) -> str:
    """Return an order-reversing encoding of `text` for descending sorts."""
    return "".join(chr(_CODEPOINT_CEILING - ord(char)) for char in text) + chr(_CODEPOINT_CEILING)


def _cmp(value_a, value_b) -> int:
    return (value_a > value_b) - (value_a < value_b)


def _encode_key(key: tuple) -> list:
    """Convert a sort key into a JSON-serializable structure."""
    return [[*component] for component in key]


def _decode_key(data: t.Iterable) -> tuple:
    """Restore a sort key from its JSON-serializable structure."""
    return tuple(tuple(component) for component in data)


@dataclass(frozen=True)
class ColumnSortSpec:
    """
    Sort specification for a single column of a :class:`SortPolicy`.

    Args:
        name: Column name used to extract values from each row.
        direction: ``"asc"`` or ``"desc"``.
        null_position: ``"first"`` or ``"last"``; controls where null-ish
            values (missing column, ``None``, and ``NaN``) sort relative to
            the type zones.
        type_zones: Ordered tuple of type zone names (subset of ``"number"``
            and ``"text"``) defining how non-null values are grouped and
            ordered. Zones listed earlier sort first.

    .. versionadded:: 8.2.0
    """

    name: str
    direction: str = ASC
    null_position: str = LAST
    type_zones: tuple = (ZONE_NUMBER, ZONE_TEXT)

    def __post_init__(self):
        if self.direction not in (ASC, DESC):
            raise ValueError(f"direction must be {ASC!r} or {DESC!r}, got {self.direction!r}")
        if self.null_position not in (FIRST, LAST):
            raise ValueError(
                f"null_position must be {FIRST!r} or {LAST!r}, got {self.null_position!r}"
            )
        type_zones = tuple(self.type_zones)
        if not type_zones:
            raise ValueError("type_zones must not be empty")
        if ZONE_NULL in type_zones:
            raise ValueError(f"{ZONE_NULL!r} is reserved and must not appear in type_zones")
        if len(set(type_zones)) != len(type_zones):
            raise ValueError(f"type_zones must not contain duplicates, got {type_zones!r}")
        object.__setattr__(self, "type_zones", type_zones)

    @staticmethod
    def classify(value) -> str:
        """Return the type zone name for `value`."""
        if value is UNSET or value is None:
            return ZONE_NULL
        if isinstance(value, NUMBER_TYPES):
            if value != value:  # noqa: PLR0124  # NaN is not equal to itself.
                return ZONE_NULL
            return ZONE_NUMBER
        return ZONE_TEXT

    def zone_rank(self, zone: str) -> int:
        """Return the sort rank of `zone` for this column."""
        if zone == ZONE_NULL:
            return -1 if self.null_position == FIRST else len(self.type_zones)
        try:
            return self.type_zones.index(zone)
        except ValueError:
            raise ValueError(
                f"zone {zone!r} is not covered by type_zones {self.type_zones!r}"
                f" of column {self.name!r}"
            ) from None

    def component(self, value) -> tuple:
        """
        Return the sort key component for `value`.

        Components are plain tuples of ints, numbers, and strings so that
        natural tuple comparison yields the configured ordering and keys stay
        persistable.
        """
        zone = self.classify(value)
        rank = self.zone_rank(zone)
        if zone == ZONE_NULL:
            return (rank,)
        if zone == ZONE_NUMBER:
            number = value if self.direction == ASC else -value
            return (rank, number)
        text = value if isinstance(value, str) else str(value)
        if self.direction == DESC:
            text = _invert_text(text)
        return (rank, text)


class SortPolicy:
    """
    A multi-column sort policy usable for partition merging.

    Each partition can call :meth:`sort_key` independently to produce
    persistable sort keys. A merger in another process only needs the keys and
    each record's global ingestion sequence number to order records: natural
    keys order first, and exact natural-key ties are always broken by the
    original ingestion sequence number (even for descending columns).

    Args:
        columns: Iterable of :class:`ColumnSortSpec` instances or
            specification dicts.
        version: Free-form policy version included in the policy fingerprint.
            Bump it when the policy changes so that cursors issued under the
            old policy are recognized as stale.

    .. versionadded:: 8.2.0
    """

    def __init__(self, columns: t.Iterable, version: t.Any = "1"):
        specs = []
        for column in columns:
            if isinstance(column, ColumnSortSpec):
                spec = column
            elif isinstance(column, dict):
                spec = ColumnSortSpec(**column)
            else:
                raise TypeError(f"unsupported column specification: {column!r}")
            specs.append(spec)
        if not specs:
            raise ValueError("a sort policy requires at least one column")
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"column names must be unique, got {names!r}")
        self.columns = tuple(specs)
        self.version = str(version)
        self._fingerprint: t.Optional[str] = None

    def __repr__(self):
        return f"{self.__class__.__name__}(columns={self.columns!r}, version={self.version!r})"

    @property
    def fingerprint(self) -> str:
        """Stable digest of the policy; embedded into every issued cursor."""
        if self._fingerprint is None:
            canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
            self._fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return self._fingerprint

    def to_dict(self) -> dict:
        """Return a canonical, persistable representation of the policy."""
        return {
            "version": self.version,
            "columns": [
                {
                    "name": spec.name,
                    "direction": spec.direction,
                    "null_position": spec.null_position,
                    "type_zones": list(spec.type_zones),
                }
                for spec in self.columns
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SortPolicy":
        """Rebuild a policy from :meth:`to_dict` output."""
        return cls(columns=data["columns"], version=data.get("version", "1"))

    def sort_key(self, row) -> tuple:
        """
        Return the persistable natural sort key for `row`.

        Missing columns, ``None``, and ``NaN`` are treated as nulls and placed
        according to each column's ``null_position``.
        """
        return tuple(
            spec.component(base_get(row, spec.name, default=None)) for spec in self.columns
        )

    @staticmethod
    def encode_key(key: tuple) -> list:
        """Convert a sort key into a JSON-serializable structure."""
        return _encode_key(key)

    @staticmethod
    def decode_key(data: t.Iterable) -> tuple:
        """Restore a sort key from its JSON-serializable structure."""
        return _decode_key(data)

    def compare(self, key_a: tuple, seq_a: int, key_b: tuple, seq_b: int) -> int:
        """
        Compare two keyed records.

        Natural keys order first; ties are always broken by the ascending
        global ingestion sequence number, regardless of column direction.
        """
        return self.compare_detail(key_a, seq_a, key_b, seq_b)[0]

    def compare_detail(
        self, key_a: tuple, seq_a: int, key_b: tuple, seq_b: int
    ) -> t.Tuple[int, str]:
        """
        Compare two keyed records and report what decided the outcome.

        Returns a ``(result, reason)`` tuple where ``reason`` is one of
        ``"zone"`` (different type zones), ``"value"`` (same zone, different
        values), ``"sequence"`` (identical natural keys, broken by ingestion
        sequence), or ``"equal"`` (same record).
        """
        if not isinstance(key_a, tuple):
            key_a = _decode_key(key_a)
        if not isinstance(key_b, tuple):
            key_b = _decode_key(key_b)
        for component_a, component_b in zip(key_a, key_b):
            if component_a[0] != component_b[0]:
                return _cmp(component_a[0], component_b[0]), "zone"
            if len(component_a) > 1 and component_a[1] != component_b[1]:
                return _cmp(component_a[1], component_b[1]), "value"
        if seq_a != seq_b:
            return _cmp(seq_a, seq_b), "sequence"
        return 0, "equal"

    def encode_cursor(self, key: t.Optional[tuple] = None, seq: t.Optional[int] = None) -> str:
        """
        Return a persistable cursor token for the position after `(key, seq)`.

        A cursor with ``key=None`` addresses the very beginning of the stream.
        """
        payload = {
            "v": _CURSOR_FORMAT_VERSION,
            "p": self.fingerprint,
            "k": _encode_key(key) if key is not None else None,
            "s": seq,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def decode_cursor(self, token) -> t.Tuple[t.Optional[tuple], t.Optional[int]]:
        """
        Decode a cursor token into its `(key, seq)` position.

        Raises:
            StaleCursorError: If the cursor was issued under a different
                (changed) policy.
            ValueError: If the token is malformed.
        """
        if isinstance(token, str):
            try:
                data = json.loads(token)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed cursor token: {exc}") from exc
        elif isinstance(token, dict):
            data = token
        else:
            raise TypeError(f"cursor token must be a JSON string or dict, got {type(token)!r}")
        if not isinstance(data, dict) or data.get("v") != _CURSOR_FORMAT_VERSION or "p" not in data:
            raise ValueError(f"malformed cursor token: {token!r}")
        if data["p"] != self.fingerprint:
            raise StaleCursorError(
                "cursor was issued under a different sort policy"
                f" (cursor policy {data['p']!r}, current policy {self.fingerprint!r})"
            )
        key = data.get("k")
        return (_decode_key(key) if key is not None else None, data.get("s"))

    def check_cursor(self, token) -> bool:
        """Return whether `token` was issued under this exact policy."""
        try:
            self.decode_cursor(token)
        except StaleCursorError:
            return False
        return True


@dataclass(frozen=True)
class KeyedRecord:
    """
    A record carrying its persistable sort key and global ingestion sequence.

    .. versionadded:: 8.2.0
    """

    key: tuple
    seq: int
    partition: str
    payload: t.Any = None

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of the record."""
        return {
            "partition": self.partition,
            "seq": self.seq,
            "key": _encode_key(self.key),
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KeyedRecord":
        """Rebuild a record from :meth:`to_dict` output."""
        return cls(
            key=_decode_key(data["key"]),
            seq=data["seq"],
            partition=data["partition"],
            payload=data.get("payload"),
        )


@dataclass
class Page:
    """
    One page of merged results.

    Attributes:
        items: The page's :class:`KeyedRecord` items in merge order.
        cursor: Cursor token addressing the position after the last item;
            pass it to :meth:`MergeStore.page` to continue reading.
        start: Cursor token addressing the position this page started from
            (a beginning token for the first page); suitable as a page
            boundary for :meth:`MergeStore.affected_pages`.
        has_more: Whether more items are available after this page.
        redo_from: If late-arriving records invalidated already-emitted
            results, a cursor token marking the earliest emitted position that
            must be re-read; ``None`` otherwise.

    .. versionadded:: 8.2.0
    """

    items: t.List[KeyedRecord]
    cursor: t.Optional[str]
    start: t.Optional[str]
    has_more: bool
    redo_from: t.Optional[str] = None

    @property
    def redo(self) -> bool:
        """Whether previously emitted results must be partially redone."""
        return self.redo_from is not None


class MergeStore:
    """
    Merges keyed records from multiple partitions and serves stable pages.

    Records are ordered by their natural sort key, with exact ties broken by
    the ascending global ingestion sequence number. The store tracks the
    high-water mark of emitted positions; when a late record arrives that
    sorts before already-emitted results, the affected segment is reported
    through :attr:`Page.redo_from` and :meth:`affected_pages` so that only the
    affected pages need to be re-emitted.

    .. versionadded:: 8.2.0
    """

    def __init__(self, policy: SortPolicy):
        if not isinstance(policy, SortPolicy):
            raise TypeError(f"policy must be a SortPolicy, got {type(policy)!r}")
        self.policy = policy
        self._records: t.List[KeyedRecord] = []
        self._sorted = True
        self._seq = 0
        self._high_water: t.Optional[tuple] = None
        self._redo_horizon: t.Optional[tuple] = None

    def __len__(self) -> int:
        return len(self._records)

    @property
    def next_seq(self) -> int:
        """The next global ingestion sequence number that will be assigned."""
        return self._seq

    @property
    def redo_from(self) -> t.Optional[str]:
        """
        Cursor token marking the earliest emitted position invalidated by
        late-arriving records, or ``None`` if emitted results are intact.
        """
        if self._redo_horizon is None:
            return None
        return self._redo_token()

    def records(self) -> t.List[KeyedRecord]:
        """Return all records in merge order."""
        self._ensure_sorted()
        return list(self._records)

    def dump(self) -> list:
        """Return all records as a JSON-serializable list."""
        return [record.to_dict() for record in self.records()]

    @classmethod
    def from_keyed(cls, policy: SortPolicy, records: t.Iterable) -> "MergeStore":
        """
        Rebuild a store from persisted records.

        `records` may contain :class:`KeyedRecord` instances or mappings with
        ``partition``, ``key``, ``seq``, and ``payload`` entries (e.g. the
        output of :meth:`dump` after a JSON round-trip from another process).
        Only the keys and sequence numbers are needed to merge and paginate.
        """
        store = cls(policy)
        for record in records:
            if isinstance(record, KeyedRecord):
                store.ingest_keyed(record.partition, record.key, record.seq, record.payload)
            else:
                store.ingest_keyed(
                    record["partition"], record["key"], record["seq"], record.get("payload")
                )
        return store

    def ingest(self, partition: str, rows: t.Iterable) -> t.List[KeyedRecord]:
        """
        Ingest a batch of rows from `partition`.

        Sort keys are computed with the store's policy and each row is
        assigned the next global ingestion sequence number.
        """
        records = [
            KeyedRecord(
                key=self.policy.sort_key(row),
                seq=self._alloc_seq(),
                partition=str(partition),
                payload=row,
            )
            for row in rows
        ]
        self._add(records)
        return records

    def ingest_keyed(
        self, partition: str, key: t.Union[tuple, list], seq: int, payload: t.Any = None
    ) -> KeyedRecord:
        """
        Ingest a single record whose sort key and sequence number were
        generated elsewhere (e.g. shipped from a partition process).
        """
        if not isinstance(key, tuple):
            key = _decode_key(key)
        record = KeyedRecord(key=key, seq=int(seq), partition=str(partition), payload=payload)
        self._seq = max(self._seq, record.seq + 1)
        self._add([record])
        return record

    def page(self, limit: int, cursor=None) -> Page:
        """
        Return the next page of up to `limit` records after `cursor`.

        Args:
            limit: Maximum number of records to return.
            cursor: ``None`` (or a beginning cursor) to read from the start,
                otherwise a cursor token returned by an earlier page.

        Raises:
            StaleCursorError: If `cursor` was issued under a different
                (changed) policy.
        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit!r}")
        if cursor is None:
            key, seq = None, None
        else:
            key, seq = self.policy.decode_cursor(cursor)
        position = None if key is None else (key, seq)
        start_token = self.policy.encode_cursor(key, seq)
        self._ensure_sorted()
        positions = [(record.key, record.seq) for record in self._records]
        index = 0 if position is None else bisect.bisect_right(positions, position)
        items = self._records[index : index + limit]
        has_more = index + limit < len(self._records)
        if items:
            last = items[-1]
            end_token = self.policy.encode_cursor(last.key, last.seq)
            self._emit((last.key, last.seq))
        else:
            end_token = start_token
        return Page(
            items=items,
            cursor=end_token,
            start=start_token,
            has_more=has_more,
            redo_from=self.redo_from,
        )

    def affected_pages(self, boundaries: t.Iterable) -> list:
        """
        Return the subset of previously emitted page `boundaries` whose pages
        must be re-emitted due to late-arriving records.

        `boundaries` are the start cursor tokens of previously emitted pages
        (see :attr:`Page.start`), given in any order. The returned tokens form
        a suffix of those boundaries: re-reading from the first returned
        boundary re-emits exactly the affected segment instead of restarting
        from the first page. Returns an empty list when no emitted results
        have been invalidated.

        Raises:
            StaleCursorError: If any boundary was issued under a different
                (changed) policy.
        """
        if self._redo_horizon is None:
            return []
        parsed = []
        for token in boundaries:
            key, seq = self.policy.decode_cursor(token)
            position = None if key is None else (key, seq)
            parsed.append((position, token))
        parsed.sort(key=lambda item: (item[0] is not None, item[0]))
        horizon = self._redo_horizon
        affected = []
        for index, (_position, token) in enumerate(parsed):
            end = parsed[index + 1][0] if index + 1 < len(parsed) else None
            if end is None or end >= horizon:
                affected.append(token)
        return affected

    def acknowledge_redo(self) -> None:
        """Clear the pending redo notification after re-emitting affected pages."""
        self._redo_horizon = None

    def _alloc_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    def _add(self, records: t.Iterable[KeyedRecord]) -> None:
        for record in records:
            position = (record.key, record.seq)
            if self._high_water is not None and position < self._high_water:
                if self._redo_horizon is None or position < self._redo_horizon:
                    self._redo_horizon = position
            self._records.append(record)
            self._sorted = False

    def _emit(self, position: tuple) -> None:
        if self._high_water is None or position > self._high_water:
            self._high_water = position

    def _ensure_sorted(self) -> None:
        if not self._sorted:
            self._records.sort(key=lambda record: (record.key, record.seq))
            self._sorted = True

    def _redo_token(self) -> str:
        self._ensure_sorted()
        positions = [(record.key, record.seq) for record in self._records]
        index = bisect.bisect_left(positions, self._redo_horizon)
        if index == 0:
            return self.policy.encode_cursor(None, None)
        previous = self._records[index - 1]
        return self.policy.encode_cursor(previous.key, previous.seq)
