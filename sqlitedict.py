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
import io
import json
import os
import sys
import tempfile
import threading
import logging
import traceback
from base64 import b64decode, b64encode
from datetime import datetime, timezone
import weakref

__version__ = '2.1.0'


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
_REQUEST_AUDIT_SUBSCRIBE = '--audit-subscribe--'
_RESPONSE_NO_MORE = '--no more--'

# Table inside every audit backup describing the boundary it represents.
# Keeping the manifest inside the database means a single atomic rename
# exposes a fully self-describing copy, with no window where a reader could
# see a new database paired with a stale external manifest.
_AUDIT_MANIFEST_TABLE = 'sqlitedict_audit_manifest'


class AuditSnapshotError(RuntimeError):
    """Raised when a requested commit boundary can no longer be captured.

    A backup subscription targets a specific commit sequence number.  If the
    worker thread has already persisted a later commit by the time the
    subscription is processed, the boundary is gone for good: the database as
    of that earlier commit was never snapshotted and cannot be reconstructed.
    """


class AuditSnapshotHandle(object):
    """A pending audit backup, returned when a snapshot is subscribed.

    The backup itself is produced by the worker thread when the requested
    commit boundary is reached.  Call :meth:`wait` to block until it exists
    and retrieve its manifest (every reference table with its row count).
    """

    def __init__(self, commit_seq, destination, completion, result_box):
        self.commit_seq = commit_seq
        self.destination = os.path.abspath(destination)
        self.manifest_path = self.destination + '.manifest.json'
        self._completion = completion
        self._result_box = result_box
        self._manifest = None

    @property
    def ready(self):
        """True once the backup file has been swapped into place."""
        return self._completion.is_set()

    def wait(self, timeout=None):
        """Block until the snapshot completes and return its manifest.

        Raises AuditSnapshotError if the backup could not be written.
        Raises TimeoutError if ``timeout`` (seconds) elapses first.
        """
        if self._manifest is None:
            if not self._completion.wait(timeout):
                raise TimeoutError(
                    'timed out after %ss waiting for the audit snapshot of '
                    'commit %d' % (timeout, self.commit_seq)
                )
            status, payload = self._result_box['result']
            if status == 'error':
                raise payload
            self._manifest = payload
        return self._manifest

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
                 timeout=5, outer_stack=True):
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

        logger.debug("opening Sqlite table %r in %r" % (tablename, filename))
        self.conn = self._new_conn()
        if self.flag == 'r':
            if self.tablename not in SqliteDict.get_tablenames(self.filename):
                msg = 'Refusing to create a new table "%s" in read-only DB mode' % tablename
                raise RuntimeError(msg)
        else:
            MAKE_TABLE = 'CREATE TABLE IF NOT EXISTS "%s" (key TEXT PRIMARY KEY, value BLOB)' % self.tablename
            self.conn.execute(MAKE_TABLE)
            # Internal flush: must not advance the user-facing commit sequence.
            self.conn.commit(counted=False)
        if flag == 'w':
            self.clear()

    def _new_conn(self):
        return SqliteMultithread(
            self.filename,
            autocommit=self.autocommit,
            journal_mode=self.journal_mode,
            outer_stack=self._outer_stack,
        )

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
        self.conn.execute(ADD_ITEM, (self.encode_key(key), self.encode(value)))
        if self.autocommit:
            self.commit()

    def __delitem__(self, key):
        if self.flag == 'r':
            raise RuntimeError('Refusing to delete from read-only SqliteDict')

        if key not in self:
            raise KeyError(key)
        DEL_ITEM = 'DELETE FROM "%s" WHERE key = ?' % self.tablename
        self.conn.execute(DEL_ITEM, (self.encode_key(key),))
        if self.autocommit:
            self.commit()

    def update(self, items=(), **kwds):
        if self.flag == 'r':
            raise RuntimeError('Refusing to update read-only SqliteDict')

        try:
            items = items.items()
        except AttributeError:
            pass
        items = [(self.encode_key(k), self.encode(v)) for k, v in items]

        UPDATE_ITEMS = 'REPLACE INTO "%s" (key, value) VALUES (?, ?)' % self.tablename
        self.conn.executemany(UPDATE_ITEMS, items)
        if kwds:
            self.update(kwds)
        if self.autocommit:
            self.commit()

    def __iter__(self):
        return self.iterkeys()

    def clear(self):
        if self.flag == 'r':
            raise RuntimeError('Refusing to clear read-only SqliteDict')

        # avoid VACUUM, as it gives "OperationalError: database schema has changed"
        CLEAR_ALL = 'DELETE FROM "%s";' % self.tablename
        self.conn.commit(counted=False)
        self.conn.execute(CLEAR_ALL)
        self.conn.commit(counted=False)

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
        """
        if self.conn is not None:
            self.conn.commit(blocking)
    sync = commit

    @property
    def commit_seq(self):
        """Ordinal (1-based) of the last commit confirmed by this dictionary.

        Counts only commits made through this instance; a freshly opened
        dictionary starts at 0.  The value is updated by the worker thread
        once a commit has actually been persisted to disk.
        """
        if self.conn is None:
            return 0
        return self.conn.commit_seq

    def subscribe_audit_snapshot(self, target_commit, destination):
        """Schedule an atomic audit backup of commit boundary ``target_commit``.

        Subscribe *before* the Nth commit is confirmed, typically while a
        later batch of writes may already be queued behind it.  When the
        worker thread reaches that commit, the whole database is copied via
        the SQLite online backup API and swapped atomically into
        ``destination``; readers of ``destination`` only ever see the
        previous complete copy or the new complete copy, never a partial
        one.  A manifest listing every reference table and its row count is
        written next to the backup as ``destination + '.manifest.json'``.

        Returns an AuditSnapshotHandle whose ``wait()`` yields the manifest.
        Raises AuditSnapshotError if ``target_commit`` was already confirmed
        before the subscription could be registered -- the live database is
        never relabeled as an older commit.
        """
        if self.flag == 'r':
            raise RuntimeError(
                'Refusing to schedule an audit snapshot from a read-only SqliteDict'
            )
        if self.conn is None:
            raise RuntimeError(
                'cannot subscribe an audit snapshot after the SqliteDict was closed'
            )
        return self.conn.subscribe_audit_snapshot(target_commit, destination)

    def close(self, do_log=True, force=False):
        if do_log:
            logger.debug("closing %s" % self)
        if hasattr(self, 'conn') and self.conn is not None:
            if self.conn.autocommit and not force:
                # typically calls to commit are non-blocking when autocommit is
                # used.  However, we need to block on close() to ensure any
                # awaiting exceptions are handled and that all data is
                # persisted to disk before returning.
                self.conn.commit(blocking=True, counted=False)
            self.conn.close(force=force)
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

    """
    def __init__(self, filename, autocommit, journal_mode, outer_stack=True):
        super(SqliteMultithread, self).__init__()
        self.filename = filename
        self.autocommit = autocommit
        self.journal_mode = journal_mode
        # use request queue of unlimited size
        self.reqs = Queue()
        self.daemon = True
        self._outer_stack = outer_stack
        self.log = logging.getLogger('sqlitedict.SqliteMultithread')

        #
        # Audit snapshot state.  Everything below is only touched from this
        # worker thread (inside run()), so no extra locking is required.
        # _commit_seq is the ordinal of the last commit already persisted.
        # _audit_subscriptions maps commit sequence number -> list of
        # (destination path, completion event, result box) entries.
        #
        self._commit_seq = 0
        self._audit_subscriptions = {}

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

        self.start()

    @property
    def commit_seq(self):
        """Ordinal of the last commit persisted by the worker thread."""
        return self._commit_seq

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
        while True:
            #
            # req: an SQL command or one of the --magic-- commands we use internally
            # arg: arguments for the command
            # res_ref: a weak reference to the queue into which responses must be placed
            # outer_stack: the outer stack, for producing more informative traces in case of error
            #
            req, arg, res_ref, outer_stack = self.reqs.get()

            if req == _REQUEST_CLOSE:
                assert res_ref, ('--close-- without return queue', res_ref)
                self._fail_pending_audit_subscriptions(
                    'connection closed before commit boundary %d was reached'
                )
                break
            elif req == _REQUEST_AUDIT_SUBSCRIBE:
                self._handle_audit_subscribe(arg, res_ref)
            elif req == _REQUEST_COMMIT:
                counted = arg[0] if arg else True
                conn.commit()
                if counted:
                    # Only user-visible commits advance the audit sequence;
                    # internal flushes (table creation, clear(), close()) pass
                    # counted=False so the first user commit is always #1.
                    self._commit_seq += 1
                    self._fulfill_audit_subscriptions(conn, self._commit_seq)
                _put(res_ref, _RESPONSE_NO_MORE)
            else:
                try:
                    cursor.execute(req, arg)
                except Exception:
                    with self._lock:
                        self.exception = (e_type, e_value, e_tb) = sys.exc_info()

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

                if self.autocommit:
                    # Persist the write immediately, but do not advance the
                    # audit sequence here: autocommit callers follow up with an
                    # explicit counted commit request, keeping one sequence
                    # increment per user-visible commit.
                    conn.commit()

        self.log.debug('received: %s, send: --no more--', req)
        conn.close()

        _put(res_ref, _RESPONSE_NO_MORE)

    def _handle_audit_subscribe(self, arg, res_ref):
        """Register (or reject) an audit backup request from the queue.

        Runs on the worker thread, so checking the current commit sequence
        against the requested boundary is race-free.
        """
        target_seq, destination, completion, result_box = arg
        result_queue = res_ref() if res_ref else None

        if target_seq <= self._commit_seq:
            error = AuditSnapshotError(
                'commit boundary %d is no longer recoverable: the database '
                'has already reached commit %d; the current database must not '
                'be labeled as commit %d'
                % (target_seq, self._commit_seq, target_seq)
            )
            if result_queue is not None:
                result_queue.put(('error', error))
                result_queue.put(_RESPONSE_NO_MORE)
            else:
                self.log.error('%s', error)
            return

        self._audit_subscriptions.setdefault(target_seq, []).append(
            (destination, completion, result_box)
        )
        if result_queue is not None:
            result_queue.put(('subscribed', target_seq))
            result_queue.put(_RESPONSE_NO_MORE)

    def _fulfill_audit_subscriptions(self, conn, commit_seq):
        """Write all backups subscribed for this just-persisted commit."""
        subscriptions = self._audit_subscriptions.pop(commit_seq, None)
        if not subscriptions:
            return

        for destination, completion, result_box in subscriptions:
            try:
                manifest = self._write_audit_snapshot(conn, destination, commit_seq)
                result_box['result'] = ('snapshot', manifest)
            except Exception as error:
                # Keep the full traceback in the log; surface a concise
                # message to whoever is waiting on the handle.
                self.log.exception(
                    'failed to write audit snapshot for commit %d', commit_seq
                )
                result_box['result'] = ('error', AuditSnapshotError(
                    'failed to write audit snapshot for commit %d to %s: '
                    '%s: %s' % (
                        commit_seq, destination,
                        type(error).__name__, error,
                    )
                ))
            finally:
                completion.set()

    def _fail_pending_audit_subscriptions(self, message):
        """Notify every still-pending audit subscription that it cannot finish."""
        for target_seq, subscriptions in self._audit_subscriptions.items():
            for _destination, completion, result_box in subscriptions:
                result_box['result'] = (
                    'error', AuditSnapshotError(message % target_seq)
                )
                completion.set()
        self._audit_subscriptions.clear()

    def _write_audit_snapshot(self, conn, destination, commit_seq):
        """Back up the database at this commit boundary, atomically.

        The SQLite online backup API copies a consistent snapshot of the
        committed database while other connections may keep writing.  The
        copy goes to a temporary file in the destination directory, gets a
        self-describing manifest table, and is swapped into place with a
        single os.replace(), so readers only ever observe the previous
        complete copy or this complete copy -- never a partial file.
        """
        destination = os.path.abspath(destination)
        destination_dir = os.path.dirname(destination) or '.'
        if not os.path.isdir(destination_dir):
            os.makedirs(destination_dir)

        fd, tmp_path = tempfile.mkstemp(
            prefix='.audit-%d-' % commit_seq, suffix='.sqlite.tmp',
            dir=destination_dir,
        )
        os.close(fd)
        manifest_path = destination + '.manifest.json'
        tmp_manifest_path = tmp_path + '.manifest.json'
        try:
            backup_conn = sqlite3.connect(tmp_path)
            try:
                # Standard CPython returns a backup object that must be
                # stepped and finished.  Some builds (notably patched
                # sqlite3 wheels) perform the whole copy synchronously and
                # return None; support both contracts.
                backup = conn.backup(backup_conn)
                if backup is not None:
                    try:
                        backup.step()
                    finally:
                        backup.finish()

                tables = self._read_audit_manifest(backup_conn)
                manifest = {
                    'source': os.path.abspath(self.filename),
                    'backup': destination,
                    'commit_seq': commit_seq,
                    'captured_at': datetime.now(timezone.utc).isoformat(),
                    'tables': tables,
                }
                # Embed the manifest in the copy itself so the single rename
                # exposes an already self-describing database.
                backup_conn.execute(
                    'CREATE TABLE "%s" (key TEXT PRIMARY KEY, value TEXT)'
                    % _AUDIT_MANIFEST_TABLE
                )
                backup_conn.executemany(
                    'INSERT INTO "%s" (key, value) VALUES (?, ?)'
                    % _AUDIT_MANIFEST_TABLE,
                    [
                        ('source', manifest['source']),
                        ('backup', manifest['backup']),
                        ('commit_seq', str(commit_seq)),
                        ('captured_at', manifest['captured_at']),
                        ('tables', json.dumps(tables, sort_keys=True)),
                    ],
                )
                backup_conn.commit()
            finally:
                backup_conn.close()

            # Sidecar JSON for readers that do not open the sqlite file.
            # It is replaced before the database itself; the in-database
            # manifest table always remains the authoritative description.
            with io.open(tmp_manifest_path, 'w') as manifest_file:
                json.dump(manifest, manifest_file, indent=2, sort_keys=True)
            os.replace(tmp_manifest_path, manifest_path)
            os.replace(tmp_path, destination)
        finally:
            for leftover in (tmp_path, tmp_manifest_path):
                if os.path.exists(leftover):
                    try:
                        os.remove(leftover)
                    except OSError:
                        self.log.exception('failed to remove temp file %s', leftover)

        self.log.info(
            'audit snapshot of commit %d written to %s (%d tables)',
            commit_seq, destination, len(tables),
        )
        return manifest

    @staticmethod
    def _read_audit_manifest(connection):
        """Return {table_name: row_count} for every user table in a copy."""
        tables = {}
        names = [
            row[0]
            for row in connection.execute(
                'SELECT name FROM sqlite_master '
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                'ORDER BY name'
            )
        ]
        for name in names:
            if name == _AUDIT_MANIFEST_TABLE:
                continue
            quoted = name.replace('"', '""')
            count = connection.execute(
                'SELECT COUNT(*) FROM "%s"' % quoted
            ).fetchone()[0]
            tables[name] = count
        return tables

    def subscribe_audit_snapshot(self, target_commit, destination):
        """Request an atomic backup once commit ``target_commit`` is persisted.

        The subscription is queued like every other request, so it is
        registered relative to the same ordering as the commit commands:
        any commit already queued ahead of the subscription will still be
        observed by the worker when the request is processed.

        Returns an :class:`AuditSnapshotHandle`.  Call ``wait()`` on it to
        block until the backup exists and get the snapshot manifest (all
        reference tables and their row counts).

        Raises AuditSnapshotError if the requested boundary has already been
        passed -- the current database state is never relabeled as an older
        commit.
        """
        target_commit = int(target_commit)
        if target_commit < 1:
            raise ValueError(
                'target_commit must be a positive commit sequence number, got %r'
                % target_commit
            )
        if not destination:
            raise ValueError('a destination path for the audit copy is required')

        self.check_raise_error()
        completion = threading.Event()
        result_box = {}
        res = Queue()
        self.execute(
            _REQUEST_AUDIT_SUBSCRIBE,
            (target_commit, os.path.abspath(destination), completion, result_box),
            res,
        )
        status, payload = res.get()
        self.check_raise_error()
        if status == 'error':
            raise payload
        return AuditSnapshotHandle(payload, destination, completion, result_box)

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

    def execute(self, req, arg=None, res=None):
        """
        `execute` calls are non-blocking: just queue up the request and return immediately.

        :param req: The request (an SQL command)
        :param arg: Arguments to the SQL command
        :param res: A queue in which to place responses as they become available
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

        self.reqs.put((req, arg or tuple(), res_ref, stack))

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

    def commit(self, blocking=True, counted=True):
        if blocking:
            # by default, we await completion of commit() unless
            # blocking=False.  This ensures any available exceptions for any
            # previous statement are thrown before returning, and that the
            # data has actually persisted to disk!
            self.select_one(_REQUEST_COMMIT, (counted,))
        else:
            # otherwise, we fire and forget as usual.
            self.execute(_REQUEST_COMMIT, (counted,))

    def close(self, force=False):
        if force:
            # If a SqliteDict is being killed or garbage-collected, then select_one()
            # could hang forever because run() might already have exited and therefore
            # can't process the request. Instead, push the close command to the requests
            # queue directly. If run() is still alive, it will exit gracefully. If not,
            # then there's nothing we can do anyway.
            self.reqs.put((_REQUEST_CLOSE, None, weakref.ref(Queue()), None))
        else:
            # we abuse 'select' to "iter" over a "--close--" statement so that we
            # can confirm the completion of close before joining the thread and
            # returning (by semaphore '--no more--'
            self.select_one(_REQUEST_CLOSE)
            self.join()


#
# This is here for .github/workflows/release.yml
#
if __name__ == '__main__':
    print(__version__)
