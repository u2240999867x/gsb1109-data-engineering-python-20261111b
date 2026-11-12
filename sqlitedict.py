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
import io
import struct
import sys
import tempfile
import threading
import logging
import traceback
from base64 import b64decode, b64encode
from urllib.parse import quote as urlquote
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
_RESPONSE_NO_MORE = '--no more--'

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
                 timeout=5, outer_stack=True, connection=None):
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

        In read-only mode (`flag='r'`) neither a missing database file nor a
        missing table is ever created: opening fails with IOError / RuntimeError
        instead.

        The `connection` parameter accepts an existing `SqliteMultithread`
        connection to share instead of opening a new one.  Shared connections
        are only supported in read-only mode and are used by `ReadonlySnapshot`
        to pin several tables to the same committed version of the database.
        A dict opened this way does not own the connection: `close()` detaches
        from it but leaves it open.

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
        self._owns_conn = connection is None
        if connection is not None:
            if flag != 'r':
                raise RuntimeError("Shared connections are only supported in read-only mode")
            self.conn = connection
        else:
            if flag == 'r' and not self.in_temp and not os.path.exists(filename):
                # refuse to create a missing database file in read-only mode
                raise IOError('file %s does not exist' % (filename))
            self.conn = self._new_conn()
        if self.flag == 'r':
            if self._owns_conn:
                tablenames = SqliteDict.get_tablenames(self.filename)
            else:
                GET_TABLENAMES = 'SELECT name FROM sqlite_master WHERE type="table"'
                tablenames = [row[0] for row in self.conn.select(GET_TABLENAMES)]
            if self.tablename not in tablenames:
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
        )

    def __enter__(self):
        if not hasattr(self, 'conn') or self.conn is None:
            self.conn = self._new_conn()
            self._owns_conn = True
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
        self.conn.commit()
        self.conn.execute(CLEAR_ALL)
        self.conn.commit()

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

        This is a no-op in read-only mode (`flag='r'`): there is never anything
        to persist, and committing would release a pinned snapshot transaction
        shared with other tables.
        """
        if self.flag == 'r':
            return
        if self.conn is not None:
            self.conn.commit(blocking)
    sync = commit

    def close(self, do_log=True, force=False):
        if do_log:
            logger.debug("closing %s" % self)
        if hasattr(self, 'conn') and self.conn is not None:
            if getattr(self, '_owns_conn', True):
                if self.conn.autocommit and not force:
                    # typically calls to commit are non-blocking when autocommit is
                    # used.  However, we need to block on close() to ensure any
                    # awaiting exceptions are handled and that all data is
                    # persisted to disk before returning.
                    self.conn.commit(blocking=True)
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


#
# Read-only snapshots: a pinned, consistent view of an existing database,
# shareable across several tables.
#
_SQLITE_HEADER_SIZE = 100
_SQLITE_HEADER_MAGIC = b'SQLite format 3\x00'
_WAL_HEADER_SIZE = 32
_WAL_FRAME_HEADER_SIZE = 24


def _db_journal_kind(header):
    """Classify a database header as 'wal' or 'journal' (rollback journal).

    Bytes 18 and 19 of the database header hold the file format read/write
    versions: 1 for the legacy rollback journal, 2 for WAL.
    """
    if len(header) >= 20 and header[:16] == _SQLITE_HEADER_MAGIC:
        if header[18] == 2 and header[19] == 2:
            return 'wal'
    return 'journal'


def _wal_frame_count(wal_path, page_size):
    """Number of frames currently stored in a -wal file (0 if absent/empty)."""
    try:
        size = os.path.getsize(wal_path)
    except OSError:
        return 0
    if size < _WAL_HEADER_SIZE:
        return 0
    return (size - _WAL_HEADER_SIZE) // (_WAL_FRAME_HEADER_SIZE + page_size)


class ReadonlySnapshot(object):
    """A pinned, read-only view of an existing SQLite database.

    The snapshot opens the database file read-only (sqlite ``mode=ro`` plus
    ``PRAGMA query_only``), so neither a missing database nor a missing table
    is ever created, and the connection itself refuses writes.  Immediately
    after connecting, a read transaction is opened and held until `close()`:
    every table accessed through this snapshot sees exactly the same committed
    version of the data, even if other connections keep writing in the
    meantime (WAL mode) -- accessing a second table never switches to a newer
    version.

    Use `table()` to obtain `SqliteDict` objects (opened with ``flag='r'``)
    bound to this snapshot, and `version` to identify the pinned committed
    version.  The version identifier changes whenever the database contents
    change, so two partitions backfilled from snapshots with equal `version`
    strings were filled from the same committed version and can be merged.

    Databases in WAL journal mode are only accepted while their ``-wal``
    companion file is present and non-empty, i.e. for active sources whose
    last consistent commit can be recovered and authenticated.  A detached
    copy of the main database file alone cannot be authenticated (it may be
    missing transactions that were still only in the WAL), so opening such a
    copy raises RuntimeError instead of silently reading a stale version.

    Note that in rollback-journal mode the pinned read transaction holds a
    shared lock, so writers will be blocked (SQLITE_BUSY) for the lifetime of
    the snapshot.  Use WAL-mode sources for concurrent writer/reader setups.
    """

    def __init__(self, filename, outer_stack=True):
        if not os.path.isfile(filename):
            raise IOError('file %s does not exist' % (filename))
        self.filename = filename
        self._outer_stack = outer_stack

        with io.open(filename, 'rb') as f:
            self._header = f.read(_SQLITE_HEADER_SIZE)
        self.journal_kind = _db_journal_kind(self._header)
        if self.journal_kind == 'wal':
            self._authenticate_wal(filename)

        self.conn = SqliteMultithread(
            filename,
            autocommit=False,
            journal_mode=None,
            outer_stack=outer_stack,
            read_only=True,
        )
        try:
            # Pin the snapshot: the read transaction fixes the committed
            # version visible through this connection until close().
            self.conn.execute('BEGIN DEFERRED')
            self.conn.select_one('SELECT COUNT(*) FROM sqlite_master')
        except Exception:
            self.conn.close(force=True)
            self.conn = None
            raise
        self._version = self._capture_version()

    @staticmethod
    def _authenticate_wal(filename):
        wal_path = filename + '-wal'
        if os.path.isfile(wal_path) and os.path.getsize(wal_path) >= _WAL_HEADER_SIZE:
            # active source: the last consistent commit is recoverable from
            # the WAL and authenticated by its salts/checksums.
            return
        raise RuntimeError(
            'Refusing to open %r: the database is in WAL mode but its -wal '
            'companion file is missing or empty, so this copy cannot be '
            'authenticated as a complete committed version (copy the -wal '
            'file together with the database, or checkpoint the source first)'
            % filename
        )

    def _capture_version(self):
        """Identify the pinned committed version as an opaque string.

        The token combines the journal kind, the database header change
        counter (bumped by every commit in rollback-journal mode and by every
        checkpoint in WAL mode) and the number of frames in the WAL (growing
        with every commit in WAL mode).  Equal tokens imply the same committed
        version; different tokens imply different versions.
        """
        if len(self._header) >= _SQLITE_HEADER_SIZE:
            change_counter = struct.unpack('>I', self._header[24:28])[0]
            page_size = struct.unpack('>H', self._header[16:18])[0]
            if page_size == 1:
                page_size = 65536
        else:
            change_counter, page_size = 0, 512
        wal_frames = 0
        if self.journal_kind == 'wal':
            wal_frames = _wal_frame_count(self.filename + '-wal', page_size)
        return '%s:%d:%d' % (self.journal_kind, change_counter, wal_frames)

    @property
    def version(self):
        """Identifier of the committed version this snapshot is pinned to."""
        return self._version

    def tablenames(self):
        """Names of the tables visible in this snapshot."""
        GET_TABLENAMES = 'SELECT name FROM sqlite_master WHERE type="table"'
        return [row[0] for row in self.conn.select(GET_TABLENAMES)]

    def table(self, tablename='unnamed', **kwargs):
        """Open `tablename` as a read-only SqliteDict bound to this snapshot.

        All dicts handed out by one snapshot share the same connection and
        therefore the same pinned committed version.  Closing a dict detaches
        it from the snapshot without disturbing the other tables.
        """
        if self.conn is None:
            raise RuntimeError('Cannot open a table on a closed snapshot')
        kwargs.setdefault('outer_stack', self._outer_stack)
        d = SqliteDict(
            filename=self.filename, tablename=tablename, flag='r',
            connection=self.conn, **kwargs
        )
        d.snapshot_version = self._version
        return d

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def __str__(self):
        return "ReadonlySnapshot(%s, version=%s)" % (self.filename, self._version)

    def __repr__(self):
        return str(self)

    def close(self):
        if getattr(self, 'conn', None) is not None:
            # closing the connection releases the pinned read transaction
            self.conn.close()
            self.conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class SqliteMultithread(threading.Thread):
    """
    Wrap sqlite connection in a way that allows concurrent requests from multiple threads.

    This is done by internally queueing the requests and processing them sequentially
    in a separate thread (in the same order they arrived).

    """
    def __init__(self, filename, autocommit, journal_mode, outer_stack=True, read_only=False):
        super(SqliteMultithread, self).__init__()
        self.filename = filename
        self.autocommit = autocommit
        self.journal_mode = journal_mode
        self.read_only = read_only
        # use request queue of unlimited size
        self.reqs = Queue()
        self.daemon = True
        self._outer_stack = outer_stack
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

        self.start()

    def _connect(self):
        """Connect to the underlying database.

        Raises an exception on failure.  Returns the connection and cursor on success.
        """
        try:
            if self.read_only:
                # Open the database file itself in read-only mode: sqlite
                # refuses to create a missing file and rejects any write.
                uri = 'file:%s?mode=ro' % urlquote(os.path.abspath(self.filename))
                conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            elif self.autocommit:
                conn = sqlite3.connect(self.filename, isolation_level=None, check_same_thread=False)
            else:
                conn = sqlite3.connect(self.filename, check_same_thread=False)
        except Exception:
            self.log.exception("Failed to initialize connection for filename: %s" % self.filename)
            self.exception = sys.exc_info()
            raise

        try:
            if self.read_only:
                # Belt and braces: the connection itself refuses writes, so
                # accidental writes cannot leak through the application layer.
                conn.execute('PRAGMA query_only = ON')
                conn.text_factory = str
                cursor = conn.cursor()
            else:
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
                break
            elif req == _REQUEST_COMMIT:
                conn.commit()
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
                    conn.commit()

        self.log.debug('received: %s, send: --no more--', req)
        conn.close()

        _put(res_ref, _RESPONSE_NO_MORE)

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

    def commit(self, blocking=True):
        if blocking:
            # by default, we await completion of commit() unless
            # blocking=False.  This ensures any available exceptions for any
            # previous statement are thrown before returning, and that the
            # data has actually persisted to disk!
            self.select_one(_REQUEST_COMMIT)
        else:
            # otherwise, we fire and forget as usual.
            self.execute(_REQUEST_COMMIT)

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
