#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This code is distributed under the terms and conditions
# from the Apache License, Version 2.0
#
# http://opensource.org/licenses/apache2.0.php
#
# This code was inspired by:
#  * http://code.activestate.com/recipes/576638-draft-for-an-sqlite3-based-dbm/
#  * http://code.activestate.com/recipes/526618/

"""
A lightweight wrapper around Python's sqlite3 database, with a dict-like interface
and multi-thread access support::

>>> mydict = SqliteDict('some.db', autocommit=True) # the mapping will be persisted to file `some.db`
>>> mydict['some_key'] = any_picklable_object
>>> print mydict['some_key']
>>> print len(mydict) # etc... all dict functions work

Pickle is used internally to serialize the values. Keys are strings.

If you don't use autocommit (default is no autocommit for performance), then
don't forget to call `mydict.commit()` when done with a transaction.

"""

import sqlite3
import os
import sys
import enum
import tempfile
import threading
import logging
import traceback
from base64 import b64decode, b64encode
import weakref

__version__ = '2.2.0'


def reraise(tp, value, tb=None):
    if value is None:
        value = tp()
    if value.__traceback__ is not tb:
        raise value.with_traceback(tb)
    raise value


try:
    from cPickle import dumps, loads, HIGHEST_PROTOCOL as PICKLE_PROTOCOL
except ImportError:
    from pickle import dumps, loads, HIGHEST_PROTOCOL as PICKLE_PROTOCOL

# some Python 3 vs 2 imports
try:
    from collections import UserDict as DictClass
except ImportError:
    from UserDict import DictMixin as DictClass

try:
    from queue import Queue
except ImportError:
    from Queue import Queue


logger = logging.getLogger(__name__)

#
# There's a thread that holds the actual SQL connection (SqliteMultithread).
# We communicate with this thread via queues (request and responses).
# The requests can either be SQL commands or one of the "special" commands
# below:
#
# _REQUEST_CLOSE: request that the SQL connection be closed
# _REQUEST_COMMIT: request that any changes be committed to the DB
#
# Responses are either SQL records (e.g. results of a SELECT) or the magic
# _RESPONSE_NO_MORE command, which indicates nothing else will ever be written
# to the response queue.
#
_REQUEST_CLOSE = '--close--'
_REQUEST_COMMIT = '--commit--'
_RESPONSE_NO_MORE = '--no more--'

#
# Lifecycle of a background (write) operation when receipts are enabled.
# See OperationReceipt below for the public surface.
#
class OperationStatus(enum.Enum):
    """The states an :class:`OperationReceipt` can be in."""

    PENDING = 'pending'      # queued or being processed by the background thread
    SUCCEEDED = 'succeeded'  # the statement finished without error
    FAILED = 'failed'        # the statement raised; the original exception is retained
    CANCELLED = 'cancelled'  # never ran: the connection was closed before processing it

    @property
    def terminal(self):
        return self is not OperationStatus.PENDING


class SqliteDictError(Exception):
    """
    Raised from :meth:`OperationReceipt.raise_for_status` when the underlying
    background operation failed.

    The :class:`OperationReceipt` (with the operation's identity) is attached as
    ``receipt`` and ``__cause__`` preserves the exception raised inside the
    background thread, so the original source of the error is not lost.
    """

    def __init__(self, receipt):
        self.receipt = receipt
        super(SqliteDictError, self).__init__(
            'background operation %s %r failed: %s'
            % (receipt.operation, receipt.identity, receipt.error)
        )


class OperationReceipt(object):
    """
    A handle on a single operation submitted to the background writer thread.

    Receipts are opt-in (see ``SqliteDict(..., receipts=True)``).  They make the
    outcome of an asynchronous operation attributable to the call that submitted
    it, instead of having the background exception surface on some unrelated
    later call.

    States (see :class:`OperationStatus`):

    * ``PENDING``   - the operation is queued or running
    * ``SUCCEEDED`` - it finished without an error; use a commit receipt to wait
      for durability when autocommit is disabled
    * ``FAILED``    - it raised; :attr:`error` holds the original exception
    * ``CANCELLED`` - the connection was closed before the operation ran

    A receipt reaches exactly one terminal state.  A failed operation is only
    ever reported through its own receipt: the same error is never re-delivered
    to another request or receipt.
    """

    __slots__ = ('identity', 'operation', '_status', 'error', '_lock', '_done', '__weakref__')

    def __init__(self, identity, operation):
        self.identity = identity
        self.operation = operation
        self._status = OperationStatus.PENDING
        self.error = None
        self._lock = threading.Lock()
        self._done = threading.Event()

    @property
    def status(self):
        """:class:`OperationStatus` of the operation."""
        return self._status

    @property
    def done(self):
        """True once the receipt has reached a terminal state."""
        return self._status.terminal

    def _set_terminal(self, status, error=None):
        # Only the first transition wins, so an operation keeps a single
        # terminal outcome even if the connection is closed concurrently.
        with self._lock:
            if self._status.terminal:
                return
            self._status = status
            if error is not None:
                self.error = error
        self._done.set()

    def wait(self, timeout=None):
        """
        Block until the operation reaches a terminal state.

        :param timeout: maximum seconds to wait, or None to wait indefinitely.
        :returns: the receipt once it is terminal, or None if the wait timed out.
          Inspect :attr:`status` or call :meth:`raise_for_status` afterwards.
        """
        if self._done.wait(timeout):
            return self
        return None

    def raise_for_status(self):
        """
        Return the receipt if the operation completed, otherwise raise.

        :raises SqliteDictError: if the operation :attr:`status` is ``FAILED``.
        :raises RuntimeError: if it was ``CANCELLED`` by a close before running.
        """
        if self._status is OperationStatus.FAILED:
            raise SqliteDictError(self) from self.error
        if self._status is OperationStatus.CANCELLED:
            raise RuntimeError(
                'background operation %s %r was cancelled before it ran'
                % (self.operation, self.identity)
            )
        return self

    def __repr__(self):
        return 'OperationReceipt(operation=%r, identity=%r, status=%s)' % (
            self.operation, self.identity, self._status.value,
        )

#
# We work with weak references for better memory efficiency.
# Dereferencing, checking the referent queue still exists, and putting to it
# is boring and repetitive, so we have a _put function to handle it for us.
#
_PUT_OK, _PUT_REFERENT_DESTROYED, _PUT_NOOP = 0, 1, 2


def _put(queue_reference, item):
    if queue_reference is not None:
        queue = queue_reference()
        if queue is None:
            #
            # We got a reference to a queue, but that queue no longer exists
            #
            retval = _PUT_REFERENT_DESTROYED
        else:
            queue.put(item)
            retval = _PUT_OK

        del queue
        return retval

    #
    # We didn't get a reference to a queue, so do nothing (no-op).
    #
    return _PUT_NOOP


def open(*args, **kwargs):
    """See documentation of the SqliteDict class."""
    return SqliteDict(*args, **kwargs)


def encode(obj):
    """Serialize an object using pickle to a binary format accepted by SQLite."""
    return sqlite3.Binary(dumps(obj, protocol=PICKLE_PROTOCOL))


def decode(obj):
    """Deserialize objects retrieved from SQLite."""
    return loads(bytes(obj))


def encode_key(key):
    """Serialize a key using pickle + base64 encoding to text accepted by SQLite."""
    return b64encode(dumps(key, protocol=PICKLE_PROTOCOL)).decode("ascii")


def decode_key(key):
    """Deserialize a key retrieved from SQLite."""
    return loads(b64decode(key.encode("ascii")))


def identity(obj):
    """Identity f(x) = x function for encoding/decoding."""
    return obj


class SqliteDict(DictClass):
    VALID_FLAGS = ['c', 'r', 'w', 'n']

    def __init__(self, filename=None, tablename='unnamed', flag='c',
                 autocommit=False, journal_mode="DELETE", encode=encode,
                 decode=decode, encode_key=identity, decode_key=identity,
                 timeout=5, outer_stack=True, receipts=False):
        """
        Initialize a thread-safe sqlite-backed dictionary. The dictionary will
        be a table `tablename` in database file `filename`. A single file (=database)
        may contain multiple tables.

        If no `filename` is given, a random file in temp will be used (and deleted
        from temp once the dict is closed/deleted).

        If you enable `autocommit`, changes will be committed after each operation
        (more inefficient but safer). Otherwise, changes are committed on `self.commit()`,
        `self.clear()` and `self.close()`.

        Set `journal_mode` to 'OFF' if you're experiencing sqlite I/O problems
        or if you need performance and don't care about crash-consistency.

        Set `outer_stack` to False to disable the output of the outer exception
        to the error logs.  This may improve the efficiency of sqlitedict
        operation at the expense of a detailed exception trace.

        Set `receipts` to True to opt into operation receipts.  Write calls then
        return an :class:`OperationReceipt` that tracks the submitted statement
        through the background writer thread, attributing success or failure to
        that specific operation instead of re-raising the error on a later,
        unrelated call.  Receipts that are still pending when the connection is
        closed are given a terminal state.  When False (the default), the dict
        behaves exactly as before, including the return values of all mapping
        methods.

        The `flag` parameter. Exactly one of:
          'c': default mode, open for read/write, creating the db/table if necessary.
          'w': open for r/w, but drop `tablename` contents first (start with empty table)
          'r': open as read-only
          'n': create a new database (erasing any existing tables, not just `tablename`!).

        The `encode` and `decode` parameters are used to customize how the values
        are serialized and deserialized.
        The `encode` parameter must be a function that takes a single Python
        object and returns a serialized representation.
        The `decode` function must be a function that takes the serialized
        representation produced by `encode` and returns a deserialized Python
        object.
        The default is to use pickle.

        The `timeout` defines the maximum time (in seconds) to wait for initial Thread startup.

        """
        self.in_temp = filename is None
        if self.in_temp:
            fd, filename = tempfile.mkstemp(prefix='sqldict')
            os.close(fd)

        if flag not in SqliteDict.VALID_FLAGS:
            raise RuntimeError("Unrecognized flag: %s" % flag)
        self.flag = flag

        if flag == 'n':
            if os.path.exists(filename):
                os.remove(filename)

        dirname = os.path.dirname(filename)
        if dirname:
            if not os.path.exists(dirname):
                raise RuntimeError('Error! The directory does not exist, %s' % dirname)

        self.filename = filename

        # Use standard SQL escaping of double quote characters in identifiers, by doubling them.
        # See https://github.com/RaRe-Technologies/sqlitedict/pull/113
        self.tablename = tablename.replace('"', '""')

        self.autocommit = autocommit
        self.journal_mode = journal_mode
        self.encode = encode
        self.decode = decode
        self.encode_key = encode_key
        self.decode_key = decode_key
        self._outer_stack = outer_stack
        self.receipts = receipts

        logger.debug("opening Sqlite table %r in %r" % (tablename, filename))
        self.conn = self._new_conn()
        if self.flag == 'r':
            if self.tablename not in SqliteDict.get_tablenames(self.filename):
                msg = 'Refusing to create a new table "%s" in read-only DB mode' % tablename
                raise RuntimeError(msg)
        else:
            MAKE_TABLE = 'CREATE TABLE IF NOT EXISTS "%s" (key TEXT PRIMARY KEY, value BLOB)' % self.tablename
            self.conn.execute(MAKE_TABLE)
            self.conn.commit()
        if flag == 'w':
            self.clear()

    def _new_conn(self):
        return SqliteMultithread(
            self.filename,
            autocommit=self.autocommit,
            journal_mode=self.journal_mode,
            outer_stack=self._outer_stack,
            receipts=self.receipts,
        )

    def _submit(self, operation, identity, req, arg=()):
        """
        Queue a write statement, returning a receipt when receipts are enabled
        (and None otherwise, preserving the historical return value).
        """
        receipt = OperationReceipt(identity, operation) if self.receipts else None
        try:
            self.conn.execute(req, arg, receipt=receipt)
        except Exception:
            # The request never made it to the worker (e.g. a dead connection):
            # give the receipt a terminal state instead of leaving it pending.
            if receipt is not None:
                receipt._set_terminal(OperationStatus.FAILED, sys.exc_info()[1])
            raise
        return receipt

    def _result(self, receipt):
        """Return value of a mapping write: the receipt when tracking, else None."""
        return receipt if self.receipts else None

    def _flush_autocommit(self, receipt=None):
        # Block until the background thread catches up.  In autocommit mode the
        # durability commit already ran inside the worker (and is covered by the
        # write receipt); waiting on that receipt needs no extra SQL.  Without
        # receipts, keep the historical blocking commit so errors still surface.
        if not self.autocommit or self.conn is None:
            return
        if receipt is not None:
            receipt.wait()
        else:
            self.conn.commit(blocking=True)

    def __enter__(self):
        if not hasattr(self, 'conn') or self.conn is None:
            self.conn = self._new_conn()
        return self

    def __exit__(self, *exc_info):
        self.close()

    def __str__(self):
        return "SqliteDict(%s)" % (self.filename)

    def __repr__(self):
        return str(self)  # no need of something complex

    def __len__(self):
        # `select count (*)` is super slow in sqlite (does a linear scan!!)
        # As a result, len() is very slow too once the table size grows beyond trivial.
        # We could keep the total count of rows ourselves, by means of triggers,
        # but that seems too complicated and would slow down normal operation
        # (insert/delete etc).
        GET_LEN = 'SELECT COUNT(*) FROM "%s"' % self.tablename
        rows = self.conn.select_one(GET_LEN)[0]
        return rows if rows is not None else 0

    def __bool__(self):
        # No elements is False, otherwise True
        GET_MAX = 'SELECT MAX(ROWID) FROM "%s"' % self.tablename
        m = self.conn.select_one(GET_MAX)[0]
        # Explicit better than implicit and bla bla
        return True if m is not None else False

    def iterkeys(self):
        GET_KEYS = 'SELECT key FROM "%s" ORDER BY rowid' % self.tablename
        for key in self.conn.select(GET_KEYS):
            yield self.decode_key(key[0])

    def itervalues(self):
        GET_VALUES = 'SELECT value FROM "%s" ORDER BY rowid' % self.tablename
        for value in self.conn.select(GET_VALUES):
            yield self.decode(value[0])

    def iteritems(self):
        GET_ITEMS = 'SELECT key, value FROM "%s" ORDER BY rowid' % self.tablename
        for key, value in self.conn.select(GET_ITEMS):
            yield self.decode_key(key), self.decode(value)

    def keys(self):
        return self.iterkeys()

    def values(self):
        return self.itervalues()

    def items(self):
        return self.iteritems()

    def __contains__(self, key):
        HAS_ITEM = 'SELECT 1 FROM "%s" WHERE key = ?' % self.tablename
        return self.conn.select_one(HAS_ITEM, (self.encode_key(key),)) is not None

    def __getitem__(self, key):
        GET_ITEM = 'SELECT value FROM "%s" WHERE key = ?' % self.tablename
        item = self.conn.select_one(GET_ITEM, (self.encode_key(key),))
        if item is None:
            raise KeyError(key)
        return self.decode(item[0])

    def __setitem__(self, key, value):
        if self.flag == 'r':
            raise RuntimeError('Refusing to write to read-only SqliteDict')

        ADD_ITEM = 'REPLACE INTO "%s" (key, value) VALUES (?,?)' % self.tablename
        receipt = self._submit('setitem', key, ADD_ITEM,
                               (self.encode_key(key), self.encode(value)))
        if self.autocommit:
            self._flush_autocommit(receipt)
        return self._result(receipt)

    def __delitem__(self, key):
        if self.flag == 'r':
            raise RuntimeError('Refusing to delete from read-only SqliteDict')

        if key not in self:
            raise KeyError(key)
        DEL_ITEM = 'DELETE FROM "%s" WHERE key = ?' % self.tablename
        receipt = self._submit('delitem', key, DEL_ITEM, (self.encode_key(key),))
        if self.autocommit:
            self._flush_autocommit(receipt)
        return self._result(receipt)

    def update(self, items=(), **kwds):
        if self.flag == 'r':
            raise RuntimeError('Refusing to update read-only SqliteDict')

        try:
            items = items.items()
        except AttributeError:
            pass
        items = list(items)

        UPDATE_ITEMS = 'REPLACE INTO "%s" (key, value) VALUES (?, ?)' % self.tablename
        if self.receipts:
            # One receipt per row makes a failure traceable back to the exact
            # imported item instead of the whole batch.
            receipts = [
                self._submit('update', key, UPDATE_ITEMS,
                             (self.encode_key(key), self.encode(value)))
                for key, value in items
            ]
        else:
            encoded = [(self.encode_key(k), self.encode(v)) for k, v in items]
            self.conn.executemany(UPDATE_ITEMS, encoded)
            receipts = []
        if kwds:
            extra = self.update(kwds)
            if self.receipts:
                receipts.extend(extra)
        if self.autocommit:
            if self.receipts:
                # Each write was already committed inside the worker; just join
                # on its receipt instead of queueing another commit request.
                for item_receipt in receipts:
                    item_receipt.wait()
            else:
                self._flush_autocommit()
        return receipts if self.receipts else None

    def __iter__(self):
        return self.iterkeys()

    def clear(self):
        if self.flag == 'r':
            raise RuntimeError('Refusing to clear read-only SqliteDict')

        # avoid VACUUM, as it gives "OperationalError: database schema has changed"
        CLEAR_ALL = 'DELETE FROM "%s";' % self.tablename
        self.conn.commit()
        receipt = self._submit('clear', self.tablename, CLEAR_ALL)
        self.conn.commit()
        return self._result(receipt)

    @staticmethod
    def get_tablenames(filename):
        """get the names of the tables in an sqlite db as a list"""
        if not os.path.isfile(filename):
            raise IOError('file %s does not exist' % (filename))
        GET_TABLENAMES = 'SELECT name FROM sqlite_master WHERE type="table"'
        with sqlite3.connect(filename) as conn:
            cursor = conn.execute(GET_TABLENAMES)
            res = cursor.fetchall()

        return [name[0] for name in res]

    def commit(self, blocking=True):
        """
        Persist all data to disk.

        When `blocking` is False, the commit command is queued, but the data is
        not guaranteed persisted (default implication when autocommit=True).

        When receipt tracking is enabled, returns an :class:`OperationReceipt`
        for the commit itself (operation ``'commit'``), otherwise None.
        """
        if self.conn is None:
            return None
        receipt = OperationReceipt(self.tablename, 'commit') if self.receipts else None
        self.conn.commit(blocking, receipt=receipt)
        return receipt if self.receipts else None
    sync = commit

    def pending_receipts(self):
        """
        Receipts still queued or running in the background thread.

        Purely in-memory bookkeeping: no SQL is executed.  Returns an empty
        list once the connection has been closed.
        """
        if self.conn is None:
            return []
        return self.conn.pending_receipts()

    def all_receipts(self):
        """
        Best-effort snapshot of all receipts still referenced by their callers.

        Reads in-memory state only and never re-runs SQL, so it is safe to use
        for diagnostics after a failure.
        """
        if self.conn is None:
            return []
        return self.conn.all_receipts()

    def close(self, do_log=True, force=False):
        if do_log:
            logger.debug("closing %s" % self)
        conn = getattr(self, 'conn', None)
        if conn is not None:
            try:
                if conn.autocommit and not force:
                    # typically calls to commit are non-blocking when autocommit is
                    # used.  However, we need to block on close() to ensure any
                    # awaiting exceptions are handled and that all data is
                    # persisted to disk before returning.
                    conn.commit(blocking=True)
            finally:
                # Always shut the worker down, even if the final commit raised,
                # so that every pending receipt still reaches a terminal state.
                try:
                    conn.close(force=force)
                finally:
                    self.conn = None
        if self.in_temp:
            try:
                os.remove(self.filename)
            except Exception:
                pass

    def terminate(self):
        """Delete the underlying database file. Use with care."""
        if self.flag == 'r':
            raise RuntimeError('Refusing to terminate read-only SqliteDict')

        self.close()

        if self.filename == ':memory:':
            return

        logger.info("deleting %s" % self.filename)
        try:
            if os.path.isfile(self.filename):
                os.remove(self.filename)
        except (OSError, IOError):
            logger.exception("failed to delete %s" % (self.filename))

    def __del__(self):
        # like close(), but assume globals are gone by now (do not log!)
        try:
            self.close(do_log=False, force=True)
        except Exception:
            # prevent error log flood in case of multiple SqliteDicts
            # closed after connection lost (exceptions are always ignored
            # in __del__ method.
            pass


class SqliteMultithread(threading.Thread):
    """
    Wrap sqlite connection in a way that allows concurrent requests from multiple threads.

    This is done by internally queueing the requests and processing them sequentially
    in a separate thread (in the same order they arrived).

    When ``receipts`` is enabled, every request may carry an
    :class:`OperationReceipt`.  The worker thread attributes success or failure
    to that receipt only; a failed statement is never re-delivered to another
    request through the legacy ``exception`` slot.

    """
    def __init__(self, filename, autocommit, journal_mode, outer_stack=True, receipts=False):
        super(SqliteMultithread, self).__init__()
        self.filename = filename
        self.autocommit = autocommit
        self.journal_mode = journal_mode
        # use request queue of unlimited size
        self.reqs = Queue()
        self.daemon = True
        self._outer_stack = outer_stack
        self.receipts_enabled = receipts
        self.log = logging.getLogger('sqlitedict.SqliteMultithread')

        #
        # Parts of this object's state get accessed from different threads, so
        # we use synchronization to avoid race conditions.  For example,
        # .exception gets set inside the new daemon thread that we spawned, but
        # gets read from the main thread.  This is particularly important
        # during initialization: the Thread needs some time to actually start
        # working, and until this happens, any calls to e.g.
        # check_raise_error() will prematurely return None, meaning all is
        # well.  If the that connection happens to fail, we'll never know about
        # it, and instead wait for a result that never arrives (effectively,
        # deadlocking).  Locking solves this problem by eliminating the race
        # condition.
        #
        self._lock = threading.Lock()
        self._lock.acquire()
        self.exception = None

        # Receipt bookkeeping.  _pending_receipts holds strong references to
        # receipts whose request is still queued/running (so they can be given
        # a terminal state on close); _all_receipts is a best-effort view for
        # diagnostics and never keeps a receipt alive on its own.
        self._receipts_lock = threading.Lock()
        self._pending_receipts = set()
        self._all_receipts = weakref.WeakSet()

        self.start()

    def _track_receipt(self, receipt):
        """Register a receipt that is about to be queued."""
        if receipt is None:
            return
        with self._receipts_lock:
            self._pending_receipts.add(receipt)
            self._all_receipts.add(receipt)

    def _untrack_receipt(self, receipt):
        if receipt is None:
            return
        with self._receipts_lock:
            self._pending_receipts.discard(receipt)

    def _cancel_pending_receipts(self):
        """
        Drive every receipt that never reached a terminal state to one.

        Used when the connection closes: operations still queued will not run,
        so they are reported as CANCELLED instead of being left pending forever.
        Already-terminal receipts are untouched.
        """
        with self._receipts_lock:
            pending = list(self._pending_receipts)
            self._pending_receipts.clear()
        for receipt in pending:
            receipt._set_terminal(OperationStatus.CANCELLED)

    def pending_receipts(self):
        """Snapshot of receipts still queued or running (purely in-memory)."""
        with self._receipts_lock:
            return list(self._pending_receipts)

    def all_receipts(self):
        """
        Best-effort snapshot of all receipts known to this connection.

        Only receipts still referenced by their callers appear here.  The
        snapshot reads in-memory state and never executes SQL.
        """
        with self._receipts_lock:
            return list(self._all_receipts)

    def _connect(self):
        """Connect to the underlying database.

        Raises an exception on failure.  Returns the connection and cursor on success.
        """
        try:
            if self.autocommit:
                conn = sqlite3.connect(self.filename, isolation_level=None, check_same_thread=False)
            else:
                conn = sqlite3.connect(self.filename, check_same_thread=False)
        except Exception:
            self.log.exception("Failed to initialize connection for filename: %s" % self.filename)
            self.exception = sys.exc_info()
            raise

        try:
            conn.execute('PRAGMA journal_mode = %s' % self.journal_mode)
            conn.text_factory = str
            cursor = conn.cursor()
            conn.commit()
            cursor.execute('PRAGMA synchronous=OFF')
        except Exception:
            self.log.exception("Failed to execute PRAGMA statements.")
            self.exception = sys.exc_info()
            raise

        return conn, cursor

    def _record_exception(self, receipt, outer_stack):
        """
        Handle an exception raised while processing a request in this thread.

        With a receipt, the error is attributed to that receipt only (the shared
        legacy ``exception`` slot is left untouched, so no unrelated request can
        ever receive it).  Without a receipt, the historical behavior is kept:
        the exception is stored for the next call to re-raise.
        """
        e_type, e_value, e_tb = sys.exc_info()

        if receipt is not None:
            receipt._set_terminal(OperationStatus.FAILED, e_value)
            self._untrack_receipt(receipt)
            self.log.error(
                'background operation %s %r failed:', receipt.operation, receipt.identity,
            )
            for item in traceback.format_exception_only(e_type, e_value):
                self.log.error(item.rstrip())
            if self._outer_stack and outer_stack:
                self.log.error('Outer stack:')
                for item in traceback.format_list(outer_stack):
                    self.log.error(item)
            return

        with self._lock:
            self.exception = (e_type, e_value, e_tb)

        inner_stack = traceback.extract_stack()

        # An exception occurred in our thread, but we may not
        # immediately able to throw it in our calling thread, if it has
        # no return `res` queue: log as level ERROR both the inner and
        # outer exception immediately.
        #
        # Any iteration of res.get() or any next call will detect the
        # inner exception and re-raise it in the calling Thread; though
        # it may be confusing to see an exception for an unrelated
        # statement, an ERROR log statement from the 'sqlitedict.*'
        # namespace contains the original outer stack location.
        self.log.error('Inner exception:')
        for item in traceback.format_list(inner_stack):
            self.log.error(item)
        self.log.error('')  # deliniate traceback & exception w/blank line
        for item in traceback.format_exception_only(e_type, e_value):
            self.log.error(item)

        self.log.error('')  # exception & outer stack w/blank line

        if self._outer_stack:
            self.log.error('Outer stack:')
            for item in traceback.format_list(outer_stack):
                self.log.error(item)
            self.log.error('Exception will be re-raised at next call.')
        else:
            self.log.error(
                'Unable to show the outer stack. Pass '
                'outer_stack=True when initializing the '
                'SqliteDict instance to show the outer stack.'
            )

    def run(self):
        #
        # Nb. this is what actually runs inside the new daemon thread.
        # self._lock is locked at this stage - see the initializer function.
        #
        try:
            conn, cursor = self._connect()
        finally:
            self._lock.release()

        res_ref = None
        receipt = None
        try:
            while True:
                #
                # req: an SQL command or one of the --magic-- commands we use internally
                # arg: arguments for the command
                # res_ref: a weak reference to the queue into which responses must be placed
                # outer_stack: the outer stack, for producing more informative traces in case of error
                # receipt: optional OperationReceipt attributing this request
                #
                req, arg, res_ref, outer_stack, receipt = self.reqs.get()

                if req == _REQUEST_CLOSE:
                    assert res_ref, ('--close-- without return queue', res_ref)
                    break
                elif req == _REQUEST_COMMIT:
                    try:
                        conn.commit()
                    except Exception:
                        self._record_exception(receipt, outer_stack)
                    else:
                        if receipt is not None:
                            receipt._set_terminal(OperationStatus.SUCCEEDED)
                    finally:
                        self._untrack_receipt(receipt)
                    _put(res_ref, _RESPONSE_NO_MORE)
                    continue

                try:
                    cursor.execute(req, arg)
                    if self.autocommit:
                        # The implicit commit is part of this write, so a failure
                        # here is attributed to the same receipt (or legacy slot).
                        conn.commit()
                except Exception:
                    self._record_exception(receipt, outer_stack)
                else:
                    if receipt is not None:
                        receipt._set_terminal(OperationStatus.SUCCEEDED)
                finally:
                    self._untrack_receipt(receipt)

                if res_ref:
                    for rec in cursor:
                        if _put(res_ref, rec) == _PUT_REFERENT_DESTROYED:
                            #
                            # The queue we are sending responses to got garbage
                            # collected.  Nobody is listening anymore, so we
                            # stop sending responses.
                            #
                            break

                    _put(res_ref, _RESPONSE_NO_MORE)
        except Exception:
            # The worker is dying unexpectedly.  Give every outstanding receipt a
            # terminal state so nobody waits forever: the in-flight request owns
            # this exception, the rest never get to run and are cancelled.
            self._fail_inflight_and_cancel_rest(receipt, sys.exc_info()[1])
            raise

        self.log.debug('received: %s, send: --no more--', req)
        conn.close()

        _put(res_ref, _RESPONSE_NO_MORE)

    def _fail_inflight_and_cancel_rest(self, inflight_receipt, error):
        """Terminal-state fallback used if the worker thread dies unexpectedly."""
        with self._receipts_lock:
            pending = list(self._pending_receipts)
            self._pending_receipts.clear()
        for receipt in pending:
            if receipt is inflight_receipt:
                receipt._set_terminal(OperationStatus.FAILED, error)
            else:
                receipt._set_terminal(OperationStatus.CANCELLED)

    def check_raise_error(self):
        """
        Check for and raise exception for any previous sqlite query.

        For the `execute*` family of method calls, such calls are non-blocking and any
        exception raised in the thread cannot be handled by the calling Thread (usually
        MainThread).  This method is called on `close`, and prior to any subsequent
        calls to the `execute*` methods to check for and raise an exception in a
        previous call to the MainThread.
        """
        with self._lock:
            if self.exception:
                e_type, e_value, e_tb = self.exception

                # clear self.exception, if the caller decides to handle such
                # exception, we should not repeatedly re-raise it.
                self.exception = None

                self.log.error('An exception occurred from a previous statement, view '
                               'the logging namespace "sqlitedict" for outer stack.')

                # The third argument to raise is the traceback object, and it is
                # substituted instead of the current location as the place where
                # the exception occurred, this is so that when using debuggers such
                # as `pdb', or simply evaluating the naturally raised traceback, we
                # retain the original (inner) location of where the exception
                # occurred.
                reraise(e_type, e_value, e_tb)

    def execute(self, req, arg=None, res=None, receipt=None):
        """
        `execute` calls are non-blocking: just queue up the request and return immediately.

        :param req: The request (an SQL command)
        :param arg: Arguments to the SQL command
        :param res: A queue in which to place responses as they become available
        :param receipt: An optional :class:`OperationReceipt` that attributes this
            request. When given, any failure is reported only through the receipt
            instead of the shared legacy exception slot.
        """
        self.check_raise_error()
        stack = None

        if self._outer_stack:
            # NOTE: This might be a lot of information to pump into an input
            # queue, affecting performance.  I've also seen earlier versions of
            # jython take a severe performance impact for throwing exceptions
            # so often.
            stack = traceback.extract_stack()[:-1]

        #
        # We pass a weak reference to the response queue instead of a regular
        # reference, because we want the queues to be garbage-collected
        # more aggressively.
        #
        res_ref = None
        if res:
            res_ref = weakref.ref(res)

        self._track_receipt(receipt)
        self.reqs.put((req, arg or tuple(), res_ref, stack, receipt))

    def executemany(self, req, items):
        for item in items:
            self.execute(req, item)
        self.check_raise_error()

    def select(self, req, arg=None):
        """
        Unlike sqlite's native select, this select doesn't handle iteration efficiently.

        The result of `select` starts filling up with values as soon as the
        request is dequeued, and although you can iterate over the result normally
        (`for res in self.select(): ...`), the entire result will be in memory.
        """
        res = Queue()  # results of the select will appear as items in this queue
        self.execute(req, arg, res)
        while True:
            rec = res.get()
            self.check_raise_error()
            if rec == _RESPONSE_NO_MORE:
                break
            yield rec

    def select_one(self, req, arg=None):
        """Return only the first row of the SELECT, or None if there are no matching rows."""
        try:
            return next(iter(self.select(req, arg)))
        except StopIteration:
            return None

    def commit(self, blocking=True, receipt=None):
        if blocking:
            # by default, we await completion of commit() unless
            # blocking=False.  This ensures any available exceptions for any
            # previous statement are thrown before returning, and that the
            # data has actually persisted to disk!
            res = Queue()
            self.execute(_REQUEST_COMMIT, res=res, receipt=receipt)
            while True:
                rec = res.get()
                self.check_raise_error()
                if rec == _RESPONSE_NO_MORE:
                    break
        else:
            # otherwise, we fire and forget as usual.
            self.execute(_REQUEST_COMMIT, receipt=receipt)

    def close(self, force=False):
        try:
            if force:
                # If a SqliteDict is being killed or garbage-collected, then select_one()
                # could hang forever because run() might already have exited and therefore
                # can't process the request. Instead, push the close command to the requests
                # queue directly. If run() is still alive, it will exit gracefully. If not,
                # then there's nothing we can do anyway.
                self.reqs.put((_REQUEST_CLOSE, None, weakref.ref(Queue()), None, None))
            else:
                # we abuse 'select' to "iter" over a "--close--" statement so that we
                # can confirm the completion of close before joining the thread and
                # returning (by semaphore '--no more--'
                self.select_one(_REQUEST_CLOSE)
                self.join()
        finally:
            # Guarantee a terminal state for receipts that never got processed
            # (e.g. requests still queued when the thread exits).
            self._cancel_pending_receipts()


#
# This is here for .github/workflows/release.yml
#
if __name__ == '__main__':
    print(__version__)
