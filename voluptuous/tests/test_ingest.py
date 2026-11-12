# fmt: off
import asyncio

import pytest

from voluptuous import (
    All,
    Coerce,
    Detail,
    DetailStream,
    GroupInvalid,
    GroupReceipt,
    Invalid,
    Range,
    Schema,
    details_validated,
    row_validated,
)

# fmt: on

ROW_CONTRACT = Schema({'order_id': str, 'qty': All(Coerce(int), Range(min=1))})


class FakeSource:
    """Async iterator over ``rows`` recording how often it was closed.

    Rows that are ``BaseException`` instances are raised instead of
    being returned, simulating a source that fails or gets cancelled
    mid-stream.
    """

    def __init__(self, rows, close_error=None):
        self.rows = list(rows)
        self.close_error = close_error
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.rows:
            raise StopAsyncIteration
        item = self.rows.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def run(coro):
    return asyncio.run(coro)


def good_rows():
    return [
        {'order_id': 'A', 'qty': '1'},
        {'order_id': 'A', 'qty': 2},
        {'order_id': 'B', 'qty': 3},
    ]


def collect(stream):
    async def gather():
        return [item async for item in stream]

    return run(gather())


def test_row_validated_returns_validated_row():
    @row_validated(ROW_CONTRACT)
    def ingest(row):
        return row

    assert ingest({'order_id': 'A', 'qty': '5'}) == {'order_id': 'A', 'qty': 5}


def test_row_validated_keeps_calling_convention():
    @row_validated(ROW_CONTRACT)
    def ingest(source, row, strict=False):
        return row

    assert ingest.__name__ == 'ingest'
    assert ingest('file.csv', {'order_id': 'A', 'qty': '2'}, strict=True) == {
        'order_id': 'A',
        'qty': 2,
    }
    assert ingest('file.csv', row={'order_id': 'B', 'qty': 1}) == {
        'order_id': 'B',
        'qty': 1,
    }


def test_row_validated_rejects_invalid_row():
    @row_validated(ROW_CONTRACT)
    def ingest(row):
        return row

    with pytest.raises(Invalid):
        ingest({'order_id': 'A', 'qty': 0})


def test_stream_validates_rows_and_completes_groups():
    items = collect(DetailStream(FakeSource(good_rows()), ROW_CONTRACT, 'order_id'))

    assert items == [
        Detail(value={'order_id': 'A', 'qty': 1}, parent_key='A', line_no=1, position=1),
        Detail(value={'order_id': 'A', 'qty': 2}, parent_key='A', line_no=2, position=2),
        GroupReceipt(parent_key='A', rows=2, first_line=1, last_line=2),
        Detail(value={'order_id': 'B', 'qty': 3}, parent_key='B', line_no=3, position=1),
        GroupReceipt(parent_key='B', rows=1, first_line=3, last_line=3),
    ]


def test_stream_supports_callable_key():
    rows = [{'order_id': 'A', 'qty': 1}, {'order_id': 'A', 'qty': 2}]
    items = collect(
        DetailStream(FakeSource(rows), ROW_CONTRACT, lambda row: row['order_id'])
    )
    receipts = [item for item in items if isinstance(item, GroupReceipt)]
    assert receipts == [GroupReceipt(parent_key='A', rows=2, first_line=1, last_line=2)]


def test_stream_closes_source_exactly_once_on_exhaustion():
    source = FakeSource(good_rows())
    collect(DetailStream(source, ROW_CONTRACT, 'order_id'))
    assert source.close_calls == 1


def test_early_stop_yields_no_receipt_for_incomplete_group():
    async def scenario():
        source = FakeSource(good_rows())
        stream = DetailStream(source, ROW_CONTRACT, 'order_id')
        first = await stream.__anext__()
        assert isinstance(first, Detail)
        await stream.aclose()
        await stream.aclose()  # closing twice is safe
        assert source.close_calls == 1
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()
        return first

    first = run(scenario())
    assert first.line_no == 1


def test_trailing_row_failure_keeps_completed_groups_deliverable():
    rows = good_rows() + [{'order_id': 'B', 'qty': 0}]
    stream = DetailStream(FakeSource(rows), ROW_CONTRACT, 'order_id')

    received = []
    with pytest.raises(GroupInvalid) as exc_info:
        while True:
            received.append(run_collect_one(stream))

    receipts = [item for item in received if isinstance(item, GroupReceipt)]
    assert receipts == [GroupReceipt(parent_key='A', rows=2, first_line=1, last_line=2)]

    error = exc_info.value
    assert error.line_no == 4
    assert error.position == 2
    assert error.parent_key == 'B'
    assert isinstance(error.__cause__, Invalid)


def run_collect_one(stream):
    async def one():
        return await stream.__anext__()

    return run(one())


def test_failing_row_with_new_key_completes_previous_group():
    rows = [
        {'order_id': 'A', 'qty': 1},
        {'order_id': 'B', 'qty': 0},
    ]
    stream = DetailStream(FakeSource(rows), ROW_CONTRACT, 'order_id')

    received = []
    with pytest.raises(GroupInvalid) as exc_info:
        while True:
            received.append(run_collect_one(stream))

    assert received[-1] == GroupReceipt(parent_key='A', rows=1, first_line=1, last_line=1)
    error = exc_info.value
    assert error.line_no == 2
    assert error.position == 1
    assert error.parent_key == 'B'


def test_mid_group_failure_reports_position_within_group():
    rows = [
        {'order_id': 'A', 'qty': 1},
        {'order_id': 'A', 'qty': 0},
    ]
    stream = DetailStream(FakeSource(rows), ROW_CONTRACT, 'order_id')

    received = []
    with pytest.raises(GroupInvalid) as exc_info:
        while True:
            received.append(run_collect_one(stream))

    assert [item for item in received if isinstance(item, GroupReceipt)] == []
    error = exc_info.value
    assert error.line_no == 2
    assert error.position == 2
    assert error.parent_key == 'A'


def test_cancellation_closes_source_once_and_propagates():
    async def scenario():
        source = FakeSource([{'order_id': 'A', 'qty': 1}, asyncio.CancelledError()])
        stream = DetailStream(source, ROW_CONTRACT, 'order_id')
        first = await stream.__anext__()
        assert isinstance(first, Detail)
        with pytest.raises(asyncio.CancelledError):
            await stream.__anext__()
        assert source.close_calls == 1

    run(scenario())


def test_close_error_does_not_mask_validation_reason():
    close_error = RuntimeError('close blew up')
    source = FakeSource([{'order_id': 'A', 'qty': 0}], close_error=close_error)
    stream = DetailStream(source, ROW_CONTRACT, 'order_id')

    with pytest.raises(GroupInvalid) as exc_info:
        run_collect_one(stream)

    error = exc_info.value
    assert error.line_no == 1
    assert error.close_error is close_error


def test_close_error_does_not_mask_cancellation_reason():
    async def scenario():
        close_error = RuntimeError('close blew up')
        source = FakeSource([asyncio.CancelledError()], close_error=close_error)
        stream = DetailStream(source, ROW_CONTRACT, 'order_id')
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await stream.__anext__()
        assert exc_info.value.close_error is close_error
        assert source.close_calls == 1

    run(scenario())


def test_close_error_on_clean_close_propagates_to_caller():
    async def scenario():
        close_error = RuntimeError('close blew up')
        source = FakeSource(good_rows(), close_error=close_error)
        stream = DetailStream(source, ROW_CONTRACT, 'order_id')
        await stream.__anext__()
        with pytest.raises(RuntimeError, match='close blew up'):
            await stream.aclose()
        assert source.close_calls == 1

    run(scenario())


def test_details_validated_decorator_keeps_calling_convention():
    @details_validated(ROW_CONTRACT, 'order_id')
    def ingest(prefix, count=2):
        return FakeSource(
            {'order_id': '%s-%d' % (prefix, i), 'qty': 1} for i in range(count)
        )

    assert ingest.__name__ == 'ingest'
    items = collect(ingest('A', count=2))
    assert [item for item in items if isinstance(item, GroupReceipt)] == [
        GroupReceipt(parent_key='A-0', rows=1, first_line=1, last_line=1),
        GroupReceipt(parent_key='A-1', rows=1, first_line=2, last_line=2),
    ]


def test_stream_works_as_async_context_manager():
    async def scenario():
        source = FakeSource(good_rows())
        async with DetailStream(source, ROW_CONTRACT, 'order_id') as stream:
            first = await stream.__anext__()
            assert isinstance(first, Detail)
        assert source.close_calls == 1

    run(scenario())


def test_context_manager_preserves_inflight_reason_over_close_error():
    async def scenario():
        close_error = RuntimeError('close blew up')
        source = FakeSource(good_rows(), close_error=close_error)
        with pytest.raises(ValueError, match='caller gave up') as exc_info:
            async with DetailStream(source, ROW_CONTRACT, 'order_id') as stream:
                await stream.__anext__()
                raise ValueError('caller gave up')
        assert exc_info.value.close_error is close_error
        assert source.close_calls == 1

    run(scenario())
