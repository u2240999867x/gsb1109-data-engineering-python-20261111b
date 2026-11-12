"""
Resumable, batchable document cleanup.

This module implements :class:`Cleanup`, a helper around
:meth:`tinydb.table.Table.remove` that allows deleting a (possibly large)
list of document IDs in small, individually committed batches.

The cleanup state -- which source positions have already produced a
verifiable result -- is persisted *inside* the database, in a dedicated
metadata table, in the very same storage write that performs the
deletions of the batch. A crash or an explicit interruption after a
batch has been committed can therefore be recovered simply by creating a
:class:`Cleanup` with the same ``cleanup_id`` again and calling
:meth:`Cleanup.run` once more: either the whole batch (deletions *and*
state) is visible after recovery, or none of it is.

Because every batch goes through the single :meth:`Storage.write`
boundary, the recovery state and the documents that were actually
deleted are always consistent within the commit boundary promised by the
storage -- an ``fsync``-ed JSON file for
:class:`~tinydb.storages.JSONStorage`, an in-memory reference swap for
:class:`~tinydb.storages.MemoryStorage`.
"""

from collections.abc import Iterable
from typing import Optional

from .storages import Storage
from .table import Table

__all__ = ('Cleanup', 'CleanupResult')

#: Name of the table that stores the recovery state of all cleanup runs.
#: It lives alongside the user tables inside the same storage dict, so a
#: single storage write covers both data and state.
CLEANUP_STATE_TABLE = '__tinydb_cleanup__'

#: Result status recorded for a single source position.
STATUS_REMOVED = 'removed'
STATUS_MISSING = 'missing'
STATUS_SKIPPED = 'skipped'

#: Overall lifecycle states of a cleanup run.
STATE_PENDING = 'pending'
STATE_DONE = 'done'


class CleanupResult:
    """
    The verifiable outcome of a single cleanup run.

    Every source position of the original request has a corresponding
    entry in :attr:`results`, so callers can reconcile what happened to
    each requested document.
    """

    def __init__(
        self,
        cleanup_id: str,
        doc_ids: list[int],
        results: list[dict],
        done: bool,
    ):
        self.cleanup_id = cleanup_id
        self.doc_ids = doc_ids
        self.results = results
        self.done = done

    @property
    def removed_ids(self) -> list[int]:
        """
        Unique document IDs that were deleted, in first-request order.

        Per-source-position detail (including duplicate positions) lives in
        :attr:`results`.
        """
        return self._unique_ids(STATUS_REMOVED)

    @property
    def missing_ids(self) -> list[int]:
        """
        Unique IDs whose first occurrence never referred to an existing
        document.

        A document that an earlier batch already deleted is *not* reported
        here: its original ``removed`` result is preserved.
        """
        return self._unique_ids(STATUS_MISSING)

    @property
    def skipped_ids(self) -> list[int]:
        """
        Unique IDs that an existing cleanup run refused to delete.

        Skipping happens when the same ID was explicitly re-created as a
        new document after the run had already evaluated an earlier
        occurrence of that ID. The old cleanup run must not touch the
        rebuilt version.
        """
        return self._unique_ids(STATUS_SKIPPED)

    def _unique_ids(self, status: str) -> list[int]:
        unique: list[int] = []
        seen: set[int] = set()
        for entry in self.results:
            if entry['status'] == status and entry['doc_id'] not in seen:
                seen.add(entry['doc_id'])
                unique.append(entry['doc_id'])
        return unique

    def __repr__(self):
        return (
            f'<CleanupResult cleanup_id={self.cleanup_id!r} '
            f'removed={len(self.removed_ids)} '
            f'missing={len(self.missing_ids)} '
            f'skipped={len(self.skipped_ids)} done={self.done}>'
        )

    def __eq__(self, other):
        if not isinstance(other, CleanupResult):
            return NotImplemented
        return (
            self.cleanup_id == other.cleanup_id
            and self.doc_ids == other.doc_ids
            and self.results == other.results
            and self.done == other.done
        )


class Cleanup:
    """
    A resumable batch deletion of documents identified by their IDs.

    :param table: the table whose documents shall be deleted
    :param doc_ids: the document IDs to delete, including duplicates;
                    each list occurrence is its own *source position*
    :param batch_size: maximum number of source positions processed per
                       storage commit
    :param cleanup_id: stable identifier of this cleanup run; retrying the
                       same cleanup means reusing the same ID
    """

    def __init__(
        self,
        table: Table,
        doc_ids: Iterable[int],
        batch_size: int = 100,
        cleanup_id: str = 'default',
    ):
        if batch_size < 1:
            raise ValueError('batch_size must be a positive integer')

        self._table = table
        self._storage: Storage = table.storage
        self._doc_ids = [int(doc_id) for doc_id in doc_ids]
        self._batch_size = batch_size
        self._cleanup_id = cleanup_id

    @property
    def cleanup_id(self) -> str:
        return self._cleanup_id

    @property
    def state_table_name(self) -> str:
        return CLEANUP_STATE_TABLE

    def run(self) -> CleanupResult:
        """
        Execute (or resume) the cleanup until every position is processed.

        Already processed source positions are skipped and their original
        results are preserved. New positions are processed in batches;
        every batch is committed before the next one starts, so an
        interruption only ever loses an uncommitted batch.

        :returns: the full, per-source-position result of this cleanup
        """
        while self.run_batch() is not None:
            pass

        return self.result()

    def run_batch(self) -> Optional[CleanupResult]:
        """
        Process exactly one batch and commit it atomically.

        This is the unit of interruption: one call performs at most one
        :meth:`Storage.write`, and that single write contains both the
        deletions of the batch and the updated recovery state.

        :returns: the result accumulated so far, or ``None`` if every
                  source position already has a committed result
        """
        state = self._load_state()
        created = state is None
        if created:
            state = self._new_state()

        if state['doc_ids'] != self._doc_ids:
            # Reusing an existing cleanup run with a different request is
            # ambiguous, so refuse rather than silently mixing results.
            raise ValueError(
                f'Cleanup {self._cleanup_id!r} already exists with a '
                'different list of document IDs'
            )

        results: list[dict] = state['results']
        cursor: int = state['cursor']
        total = len(self._doc_ids)

        if cursor >= total:
            if created:
                # Empty request: persist a completed state once so that
                # ``result()`` and ``is_done()`` work for empty cleanups.
                state['status'] = STATE_DONE
                self._commit(self._storage.read(), [], state)
                return self._to_result(state)
            return None

        end = min(cursor + self._batch_size, total)
        batch_positions = range(cursor, end)

        # One snapshot for the whole batch keeps all decisions of this
        # commit atomic. The raw storage keys are strings, so convert them
        # to the table's document ID class.
        tables = self._storage.read()
        raw_table = (tables or {}).get(self._table.name, {})
        existing_ids = {
            self._table.document_id_class(doc_id)
            for doc_id in raw_table.keys()
        }
        # Status of the first occurrence of every ID processed in an
        # earlier, already committed batch.
        prior_status = {
            entry['doc_id']: entry['status'] for entry in results
        }
        # Status of the first occurrence of every ID inside this batch.
        batch_status: dict[int, str] = {}
        to_remove: list[int] = []
        batch_results: list[dict] = []

        for position in batch_positions:
            doc_id = self._doc_ids[position]

            if doc_id in prior_status:
                # The first occurrence of this ID was evaluated in an
                # earlier batch. If the ID is present again now, the
                # document is a version explicitly created *after* the
                # original request -- leave it untouched.
                if doc_id in existing_ids:
                    status = STATUS_SKIPPED
                else:
                    status = prior_status[doc_id]
            elif doc_id in batch_status:
                # Duplicate inside the same batch: associate the position
                # with the first request's decision instead of evaluating
                # the (already mutated) table a second time.
                status = batch_status[doc_id]
            elif doc_id in existing_ids:
                status = STATUS_REMOVED
                batch_status[doc_id] = status
                to_remove.append(doc_id)
            else:
                status = STATUS_MISSING
                batch_status[doc_id] = status

            batch_results.append(
                {
                    'position': position,
                    'doc_id': doc_id,
                    'status': status,
                }
            )

        results.extend(batch_results)
        state['results'] = results
        state['cursor'] = end
        if end >= total:
            state['status'] = STATE_DONE

        self._commit(tables, to_remove, state)

        return self._to_result(state)

    def resume(self) -> CleanupResult:
        """
        Resume an interrupted cleanup run.

        Equivalent to :meth:`run`; provided for readability at call sites
        that only handle the recovery case.
        """
        return self.run()

    def is_done(self) -> bool:
        """Whether every source position has a committed result."""
        state = self._load_state()
        return state is not None and state['status'] == STATE_DONE

    def result(self) -> CleanupResult:
        """
        Return the current result without performing any deletion.

        :raises KeyError: if this cleanup run has never been started
        """
        state = self._load_state()
        if state is None:
            raise KeyError(
                f'No cleanup state found for cleanup_id={self._cleanup_id!r}'
            )
        return self._to_result(state)

    def discard(self) -> None:
        """
        Delete the recovery state of this cleanup run.

        Only the metadata is removed; documents deleted by the run stay
        deleted.
        """
        tables = self._storage.read()
        if tables is None:
            return
        state_table = tables.get(CLEANUP_STATE_TABLE)
        if not state_table or self._state_key not in state_table:
            return
        del state_table[self._state_key]
        if not state_table:
            # Avoid leaving an empty metadata table behind.
            del tables[CLEANUP_STATE_TABLE]
        self._storage.write(tables)

    @property
    def _state_key(self) -> str:
        # Namespacing by table name keeps identically named cleanup runs on
        # different tables from sharing state.
        return f'{self._table.name}::{self._cleanup_id}'

    def _new_state(self) -> dict:
        return {
            'cleanup_id': self._cleanup_id,
            'doc_ids': list(self._doc_ids),
            'cursor': 0,
            'status': STATE_PENDING,
            'results': [],
        }

    def _load_state(self) -> Optional[dict]:
        tables = self._storage.read()
        if tables is None:
            return None
        state_table = tables.get(CLEANUP_STATE_TABLE)
        if not state_table:
            return None
        return state_table.get(self._state_key)

    def _commit(
        self,
        tables: Optional[dict],
        to_remove: list[int],
        state: dict,
    ) -> None:
        """
        Apply the batch deletions and the new recovery state in one write.

        Sharing the storage write is what makes the recovery state and the
        actual deletions consistent within a single commit boundary for
        both JSON and memory storages.
        """
        if tables is None:
            tables = {}

        if to_remove:
            raw_table = tables.get(self._table.name, {})
            table = {
                self._table.document_id_class(doc_id): doc
                for doc_id, doc in raw_table.items()
            }

            for doc_id in to_remove:
                # ``to_remove`` is deduplicated, so every ID is deleted
                # exactly once -- duplicate source positions can no longer
                # trigger a second ``del`` and raise ``KeyError``.
                del table[doc_id]

            tables[self._table.name] = {
                str(doc_id): doc for doc_id, doc in table.items()
            }

        tables.setdefault(CLEANUP_STATE_TABLE, {})[self._state_key] = state

        self._storage.write(tables)

        # The table caches query results, not raw data; invalidate the
        # cache because the storage contents changed underneath it.
        self._table.clear_cache()

    def _to_result(self, state: dict) -> CleanupResult:
        return CleanupResult(
            self._cleanup_id,
            list(state['doc_ids']),
            [dict(entry) for entry in state['results']],
            state['status'] == STATE_DONE,
        )
