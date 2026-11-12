"""Streaming validation adapters for row-ingestion functions.

Ingestion functions come in two calling conventions:

* Synchronous functions returning a single row.  Wrap them with
  :func:`row_validated`; the returned row is checked against the
  declared row contract before it is handed back to the caller, while
  the calling convention of the original function is preserved.

* Functions returning an asynchronous stream of detail rows.  Wrap them
  with :func:`details_validated`, or feed the stream to
  :class:`DetailStream` directly.  Each row is validated against the
  declared row contract as it is read, rows are grouped by their parent
  business key, and a :class:`GroupReceipt` is emitted once a group of
  details is known to be complete (the key changes or the source is
  exhausted).

The caller may stop reading a detail stream at any point by calling
:meth:`DetailStream.aclose` (or by leaving an ``async with`` block).  A
group that has been read but is not yet complete never receives a
receipt.  Cancelling closes the source iterator exactly once; if the
source raises again while it is being closed, the close error is
attached to the original validation or cancellation reason as
``close_error`` instead of masking it.
"""

# fmt: off
import collections
import functools
import inspect
import typing

from voluptuous import error as er
from voluptuous.schema_builder import Schema

# fmt: on

__all__ = [
    'Detail',
    'DetailStream',
    'GroupInvalid',
    'GroupReceipt',
    'details_validated',
    'row_validated',
]


class GroupInvalid(er.Invalid):
    """A detail row failed validation inside a stream.

    :attr line_no: 1-based number of the offending row in the original
        input.
    :attr position: 1-based position of the offending row within its
        group.
    :attr parent_key: parent business key of the failed group, when it
        could be determined.
    :attr close_error: error raised by the source iterator while it was
        being closed after this failure, if any.
    """

    def __init__(
        self,
        message: str,
        *,
        line_no: int,
        position: int,
        parent_key: typing.Optional[typing.Hashable] = None,
        path: typing.Optional[typing.List[typing.Hashable]] = None,
        error_message: typing.Optional[str] = None,
        error_type: typing.Optional[str] = None,
    ) -> None:
        super().__init__(message, path, error_message, error_type)
        self.line_no = line_no
        self.position = position
        self.parent_key = parent_key
        self.close_error: typing.Optional[Exception] = None


class Detail(typing.NamedTuple):
    """A single validated detail row.

    :attr value: the validated row.
    :attr parent_key: parent business key of the group the row belongs
        to.
    :attr line_no: 1-based number of the row in the original input.
    :attr position: 1-based position of the row within its group.
    """

    value: typing.Any
    parent_key: typing.Hashable
    line_no: int
    position: int


class GroupReceipt(typing.NamedTuple):
    """Receipt emitted once a group of details is known to be complete.

    :attr parent_key: parent business key identifying the group.
    :attr rows: number of detail rows in the group.
    :attr first_line: 1-based line number of the first row of the group
        in the original input.
    :attr last_line: 1-based line number of the last row of the group in
        the original input.
    """

    parent_key: typing.Hashable
    rows: int
    first_line: int
    last_line: int


_UNKNOWN = object()


def _as_validator(
    contract: typing.Any,
) -> typing.Callable[[typing.Any], typing.Any]:
    if isinstance(contract, Schema):
        return contract
    return Schema(contract)


def _as_key_extractor(
    key: typing.Union[str, typing.Callable[[typing.Any], typing.Hashable]],
) -> typing.Callable[[typing.Any], typing.Hashable]:
    if callable(key):
        return key

    def extract(row: typing.Any) -> typing.Hashable:
        return row[key]

    return extract


class DetailStream:
    """Async iterator validating a stream of detail rows row by row.

    :param source: async iterable or async iterator of raw rows.
    :param contract: row contract; anything accepted by :class:`Schema`.
    :param key: parent business key, either a callable applied to each
        (validated) row or the name of the row field holding the key.

    Yields :class:`Detail` for every validated row and a
    :class:`GroupReceipt` whenever a group of details becomes complete.
    """

    def __init__(
        self,
        source: typing.Any,
        contract: typing.Any,
        key: typing.Union[str, typing.Callable[[typing.Any], typing.Hashable]],
    ) -> None:
        self._validate = _as_validator(contract)
        self._key_of = _as_key_extractor(key)
        self._source = source.__aiter__()
        self._line_no = 0
        self._has_group = False
        self._group_key: typing.Hashable = None
        self._group_rows = 0
        self._group_first_line = 0
        self._group_last_line = 0
        self._outbox: typing.Deque[typing.Any] = collections.deque()
        self._pending_error: typing.Optional[GroupInvalid] = None
        self._done = False
        self._closed = False

    def __aiter__(self) -> 'DetailStream':
        return self

    async def __anext__(self) -> typing.Any:
        while True:
            if self._outbox:
                return self._outbox.popleft()
            if self._pending_error is not None:
                error = self._pending_error
                self._pending_error = None
                await self._shutdown(error)
                raise error
            if self._done:
                raise StopAsyncIteration
            try:
                await self._pump_once()
            except StopAsyncIteration:
                self._done = True
                await self._shutdown(None)
                if self._has_group:
                    self._outbox.append(self._receipt())
                    self._has_group = False
            except BaseException as primary:
                self._done = True
                await self._shutdown(primary)
                raise

    async def __aenter__(self) -> 'DetailStream':
        return self

    async def __aexit__(self, exc_type: typing.Any, exc: typing.Any, tb: typing.Any) -> bool:
        self._abandon()
        await self._shutdown(exc if isinstance(exc, BaseException) else None)
        return False

    async def aclose(self) -> None:
        """Stop reading early and close the source iterator exactly once.

        The pending, still incomplete group (if any) is discarded and
        never receives a receipt.  Errors raised by the source while it
        is closing are propagated to the caller of this method.
        """
        self._abandon()
        await self._shutdown(None)

    def _abandon(self) -> None:
        self._done = True
        self._outbox.clear()
        self._pending_error = None
        self._has_group = False

    async def _pump_once(self) -> None:
        raw = await self._source.__anext__()
        self._line_no += 1
        try:
            value = self._validate(raw)
        except er.Invalid as exc:
            self._done = True
            self._record_row_error(raw, exc)
            return
        key = self._key_of(value)
        if self._has_group and key != self._group_key:
            self._outbox.append(self._receipt())
            self._start_group(key)
        elif not self._has_group:
            self._start_group(key)
        else:
            self._group_rows += 1
            self._group_last_line = self._line_no
        self._outbox.append(
            Detail(
                value=value,
                parent_key=key,
                line_no=self._line_no,
                position=self._group_rows,
            )
        )

    def _record_row_error(self, raw: typing.Any, exc: er.Invalid) -> None:
        position = 1
        parent_key: typing.Optional[typing.Hashable] = None
        try:
            key = self._key_of(raw)
        except Exception:
            key = _UNKNOWN
        if key is _UNKNOWN:
            if self._has_group:
                position = self._group_rows + 1
                parent_key = self._group_key
        elif self._has_group and key == self._group_key:
            position = self._group_rows + 1
            parent_key = key
        else:
            # The failing row starts a new group, so its arrival completes
            # the previous one; that completed group is still delivered.
            parent_key = key
            if self._has_group:
                self._outbox.append(self._receipt())
                self._has_group = False
        error = GroupInvalid(
            'row %d failed validation: %s' % (self._line_no, exc.msg),
            line_no=self._line_no,
            position=position,
            parent_key=parent_key,
            path=list(exc.path),
            error_message=exc.error_message,
        )
        error.__cause__ = exc
        self._pending_error = error

    def _start_group(self, key: typing.Hashable) -> None:
        self._has_group = True
        self._group_key = key
        self._group_rows = 1
        self._group_first_line = self._line_no
        self._group_last_line = self._line_no

    def _receipt(self) -> GroupReceipt:
        return GroupReceipt(
            parent_key=self._group_key,
            rows=self._group_rows,
            first_line=self._group_first_line,
            last_line=self._group_last_line,
        )

    async def _shutdown(self, primary: typing.Optional[BaseException]) -> None:
        if self._closed:
            return
        self._closed = True
        aclose = getattr(self._source, 'aclose', None)
        if aclose is None:
            return
        try:
            result = aclose()
            if inspect.isawaitable(result):
                await result
        except Exception as close_error:
            if primary is None:
                raise
            # Keep the original validation or cancellation reason as the
            # propagated error; the close failure rides along on it.
            primary.close_error = close_error  # type: ignore[attr-defined]


def row_validated(
    contract: typing.Any,
) -> typing.Callable[[typing.Callable[..., typing.Any]], typing.Callable[..., typing.Any]]:
    """Decorate a synchronous ingestion function returning a single row.

    The wrapped function keeps the original calling convention; the row
    it returns is validated against ``contract`` before being handed
    back to the caller.
    """
    validate = _as_validator(contract)

    def decorator(func: typing.Callable[..., typing.Any]) -> typing.Callable[..., typing.Any]:
        @functools.wraps(func)
        def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            return validate(func(*args, **kwargs))

        return wrapper

    return decorator


def details_validated(
    contract: typing.Any,
    key: typing.Union[str, typing.Callable[[typing.Any], typing.Hashable]],
) -> typing.Callable[[typing.Callable[..., typing.Any]], typing.Callable[..., DetailStream]]:
    """Decorate a function returning an asynchronous stream of detail rows.

    The wrapped function keeps the original calling convention and
    returns a :class:`DetailStream` validating each row against
    ``contract`` and grouping rows by the parent business key ``key``.
    """

    def decorator(func: typing.Callable[..., typing.Any]) -> typing.Callable[..., DetailStream]:
        @functools.wraps(func)
        def wrapper(*args: typing.Any, **kwargs: typing.Any) -> DetailStream:
            return DetailStream(func(*args, **kwargs), contract, key)

        return wrapper

    return decorator
