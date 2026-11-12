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
import tempfile
import threading
import logging
import traceback
import uuid
import hashlib
from base64 import b64decode, b64encode
import weakref

try:
    from collections.abc import Mapping
except ImportError:  # pragma: no cover - Python 2 fallback
    from collections import Mapping

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


#
# Typed-key migration
# -------------------
#
# A legacy table only ever stored raw, human-readable text keys (the default
# ``identity`` key codec).  Switching to ``encode_key``/``decode_key`` introduces
# typed keys (ints, tuples, ...) whose on-disk representations live in a
# *different* text space, so a legacy row with the text key "12" and a new row
# with the integer key 12 can both exist at once.
#
# The migration first *splits* the live table: every legacy row is moved into a
# generation-scoped ``pending`` table, leaving only typed rows reachable through
# the dict.  Pending rows are then processed in batches, each ending in a durable
# commit: an unambiguous row is published under its typed key and recorded
# (exact original bytes) in a generation-scoped ``tombstone`` table, while an
# ambiguous row is moved into a generation-scoped ``held`` table.
#
# Every migration run is pinned to a single, persisted "generation id" used in
# all its work-table names.  Queries, deletes and background writes of one run
# therefore target one mapping and can never see another generation's rows.  A
# resume adopts the persisted id, and because a moved/deleted value no longer
# lives in the live table, it cannot be read from or resurrected via the old key
# space.
#

# The on-disk key codec this migration produces (kept as constants so the
# migration is always self-contained, regardless of the instance's codecs).
_MIGRATION_ENCODE_KEY = encode_key
_MIGRATION_DECODE_KEY = decode_key
_MIGRATION_META_TABLE = 'sqlitedict_key_migration'


def looks_like_encoded_key(stored_key):
    """Return True when a stored key already belongs to the typed-key space.

    Legacy rows store raw, human-readable text keys; typed rows store the
    pickle+base64 output of :func:`encode_key`.  A stored key belongs to the
    typed space only if it round-trips exactly through the typed codec, which
    a coincidentally base64-looking legacy text key cannot do.  This is what
    stops a migration from re-processing the typed rows it just wrote into the
    same table.
    """
    try:
        return _MIGRATION_ENCODE_KEY(_MIGRATION_DECODE_KEY(stored_key)) == stored_key
    except Exception:
        return False


class MigrationClaim(object):
    """An explicit caller-provided interpretation of a legacy key.

    A claim maps a legacy *text* key (as stored in the old table) to a typed
    Python key that the new, typed source uses.  Callers build these from their
    own source manifests; they are what lets a migration decide which typed key
    should inherit a legacy value when several interpretations are possible.
    """

    __slots__ = ('legacy', 'typed')

    def __init__(self, legacy, typed):
        if not isinstance(legacy, str):
            raise TypeError('legacy claim key must be a text string, got %r' % type(legacy))
        self.legacy = legacy
        self.typed = typed

    def __eq__(self, other):
        if not isinstance(other, MigrationClaim):
            return NotImplemented
        return self.legacy == other.legacy and self.typed == other.typed

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self):
        try:
            return hash((self.legacy, self.typed))
        except TypeError:
            return hash(self.legacy)

    def __repr__(self):
        return 'MigrationClaim(legacy=%r, typed=%r)' % (self.legacy, self.typed)


class MigrationReport(object):
    """Summary of one ``migrate_typed_keys`` invocation."""

    __slots__ = ('generation', 'migrated', 'tombstoned',
                 'ambiguous', 'resumed', 'complete')

    def __init__(self, generation, migrated=0, tombstoned=0,
                 ambiguous=(), resumed=False, complete=False):
        self.generation = generation
        self.migrated = migrated
        self.tombstoned = tombstoned
        self.ambiguous = list(ambiguous)
        self.resumed = resumed
        self.complete = complete

    def __repr__(self):
        return (
            'MigrationReport(generation=%r, migrated=%d, tombstoned=%d, '
            'ambiguous=%r, resumed=%r, complete=%r)'
            % (self.generation, self.migrated, self.tombstoned,
               self.ambiguous, self.resumed, self.complete)
        )


def _claims_to_map(claims):
    """Turn a claim iterable/mapping into a {legacy_text: typed_key} dict."""
    if claims is None:
        return {}
    if isinstance(claims, Mapping):
        items = claims.items()
    else:
        items = ((claim.legacy, claim.typed) for claim in claims)
    result = {}
    for legacy, typed in items:
        if not isinstance(legacy, str):
            raise TypeError(
                'legacy claim key must be a text string, got %r' % type(legacy))
        result[legacy] = typed
    return result


class KeyMigration(object):
    """Batch, resumable migration of legacy text keys to typed keys.

    Construct one with a :class:`SqliteDict` opened using the typed key codec
    (``encode_key``/``decode_key``) and drive it with :meth:`run`, or call
    :meth:`SqliteDict.migrate_typed_keys` directly.

    The live table starts out holding only legacy, raw-text rows.  The first
    run *splits* it: every legacy row is moved into a generation-scoped
    ``pending`` table, leaving the live table containing typed rows only.  The
    pending rows are then processed in batches:

    * a row with one defensible interpretation is written to the live table
      under its typed key, recorded with its exact original bytes in a
      generation-scoped ``tombstone`` table, and removed from ``pending``;
    * a row with several plausible interpretations (text "12" vs integer 12)
      that no caller claim settles is moved into a generation-scoped ``held``
      table.  Its bytes are preserved, but it is no longer reachable through
      the dict, so neither typed key can read the old value away.  Supplying a
      claim and re-running moves it back into ``pending`` for another pass.

    Concurrency / consistency model:

    * A migration owns a unique ``generation_id``, persisted in a sidecar
      metadata table.  The pending, held and tombstone tables are all named
      with that id, so every query, delete and background write of one run
      targets the same mapping and can never be confused with another
      generation's.  Re-opening a migration for the table adopts the persisted
      generation instead of forking.
    * Every batch and every split step is durably committed, so an
      interruption only loses un-started work.  On resume the anti-join against
      ``pending`` and the tombstones make every move idempotent: a value that
      was already deleted from the live table is never copied back, i.e. it
      cannot be resurrected from the old key space.
    """

    def __init__(self, db, claims=None, batch_size=500, generation_id=None):
        if not isinstance(db, SqliteDict):
            raise TypeError('KeyMigration expects a SqliteDict instance')
        if db.flag == 'r':
            raise RuntimeError('Refusing to migrate a read-only SqliteDict')
        if db.encode_key is not _MIGRATION_ENCODE_KEY or \
                db.decode_key is not _MIGRATION_DECODE_KEY:
            raise RuntimeError(
                'Typed-key migration requires the SqliteDict to be opened with '
                'encode_key=sqlitedict.encode_key and '
                'decode_key=sqlitedict.decode_key')
        if batch_size < 1:
            raise ValueError('batch_size must be a positive integer')

        self.db = db
        self.claims = _claims_to_map(claims)
        self.batch_size = int(batch_size)
        self._generation = generation_id or uuid.uuid4().hex
        self.table = db.tablename
        # Generation-scoped work tables.  ``pending`` holds legacy rows waiting
        # to be classified, ``held`` rows parked until a claim arrives, and
        # ``tombstone`` the exact original bytes of every migrated row.
        self.pending_table = '"_km_pending_%s_%s"' % (self.table, self._generation)
        self.held_table = '"_km_held_%s_%s"' % (self.table, self._generation)
        self.tomb_table = '"_km_tomb_%s_%s"' % (self.table, self._generation)
        # Encoded typed keys already produced by the current run, used to catch
        # two legacy rows resolving to the same typed key inside one batch.
        self._claimed_targets = set()

    def _claims_signature(self):
        """Stable hash of the current claims, used to detect manifest changes."""
        items = sorted(self.claims.items(), key=lambda kv: kv[0])
        payload = dumps(
            [(legacy, dumps(typed, protocol=PICKLE_PROTOCOL))
             for legacy, typed in items],
            protocol=PICKLE_PROTOCOL)
        return hashlib.sha256(payload).hexdigest()

    # ------------------------------------------------------------------ #
    # Metadata / schema
    # ------------------------------------------------------------------ #

    def _ensure_meta(self):
        self.db.conn.execute(
            'CREATE TABLE IF NOT EXISTS "%s" ('
            'tablename TEXT PRIMARY KEY, generation TEXT NOT NULL, '
            'complete INTEGER NOT NULL DEFAULT 0, claims_sig TEXT, '
            'split_done INTEGER NOT NULL DEFAULT 0)'
            % _MIGRATION_META_TABLE)
        # Add columns introduced after the original schema.  Inspect via a
        # blocking PRAGMA rather than issuing ALTER blindly: worker-thread
        # errors only surface on the next call, so a plain try/except around
        # execute() cannot swallow a duplicate-column error.
        columns = {row[1] for row in self.db.conn.select(
            'PRAGMA table_info("%s")' % _MIGRATION_META_TABLE)}
        if 'claims_sig' not in columns:
            self.db.conn.execute(
                'ALTER TABLE "%s" ADD COLUMN claims_sig TEXT'
                % _MIGRATION_META_TABLE)
        if 'split_done' not in columns:
            self.db.conn.execute(
                'ALTER TABLE "%s" ADD COLUMN split_done INTEGER NOT NULL DEFAULT 0'
                % _MIGRATION_META_TABLE)

    def _split_already_done(self):
        row = self.db.conn.select_one(
            'SELECT split_done FROM "%s" WHERE tablename = ?'
            % _MIGRATION_META_TABLE, (self.table,))
        return bool(row and row[0])

    def _mark_split_done(self):
        self.db.conn.execute(
            'UPDATE "%s" SET split_done = 1 WHERE tablename = ?'
            % _MIGRATION_META_TABLE, (self.table,))

    def _open_state(self):
        """Return (generation, complete, created), creating state if new."""
        self._ensure_meta()
        row = self.db.conn.select_one(
            'SELECT generation, complete FROM "%s" WHERE tablename = ?'
            % _MIGRATION_META_TABLE, (self.table,))
        if row is None:
            self.db.conn.execute(
                'INSERT INTO "%s" (tablename, generation, complete) '
                'VALUES (?, ?, 0)'
                % _MIGRATION_META_TABLE, (self.table, self._generation))
            self.db.commit(blocking=True)
            return self._generation, False, True
        generation, complete = row
        return generation, bool(complete), False

    def _bind_persisted_generation(self):
        """Point this instance at the table's persisted migration state."""
        self._ensure_meta()
        row = self.db.conn.select_one(
            'SELECT generation FROM "%s" WHERE tablename = ?'
            % _MIGRATION_META_TABLE, (self.table,))
        if row is None:
            raise RuntimeError(
                'No typed-key migration state exists for table %r yet'
                % self.table)
        self._set_generation(row[0])
        self._ensure_work_tables()

    def _set_generation(self, generation):
        self._generation = generation
        self.pending_table = '"_km_pending_%s_%s"' % (self.table, generation)
        self.held_table = '"_km_held_%s_%s"' % (self.table, generation)
        self.tomb_table = '"_km_tomb_%s_%s"' % (self.table, generation)

    def _ensure_work_tables(self):
        legacy_schema = 'legacy_key TEXT PRIMARY KEY, value BLOB NOT NULL'
        # Rows still waiting to be classified.
        self.db.conn.execute(
            'CREATE TABLE IF NOT EXISTS %s (%s) WITHOUT ROWID'
            % (self.pending_table, legacy_schema))
        # Rows parked because they cannot be settled without a claim.  Their
        # exact value bytes are preserved for a later, claim-driven pass.
        self.db.conn.execute(
            'CREATE TABLE IF NOT EXISTS %s (%s) WITHOUT ROWID'
            % (self.held_table, legacy_schema))
        # The exact original bytes of every migrated row, for reconciliation
        # and to make resume idempotent.
        self.db.conn.execute(
            'CREATE TABLE IF NOT EXISTS %s (%s) WITHOUT ROWID'
            % (self.tomb_table, legacy_schema))

    def _release_held_if_claims_changed(self):
        """Move held rows back to pending when the claims manifest changes.

        Held rows are only re-evaluated when the caller supplies new claims;
        otherwise the same ambiguous rows would be re-classified on every run.
        """
        sig = self._claims_signature()
        row = self.db.conn.select_one(
            'SELECT claims_sig FROM "%s" WHERE tablename = ?'
            % _MIGRATION_META_TABLE, (self.table,))
        stored_sig = row[0] if row is not None else None
        if stored_sig != sig:
            self.db.conn.execute(
                'INSERT OR IGNORE INTO %s (legacy_key, value) '
                'SELECT legacy_key, value FROM %s'
                % (self.pending_table, self.held_table))
            self.db.conn.execute('DELETE FROM %s' % self.held_table)
            self.db.conn.execute(
                'UPDATE "%s" SET claims_sig = ? WHERE tablename = ?'
                % _MIGRATION_META_TABLE, (sig, self.table))

    def _mark_complete(self, complete):
        self.db.conn.execute(
            'UPDATE "%s" SET complete = ? WHERE tablename = ?'
            % _MIGRATION_META_TABLE, (1 if complete else 0, self.table))

    # ------------------------------------------------------------------ #
    # Key interpretation
    # ------------------------------------------------------------------ #

    def _candidate_typed_keys(self, legacy_key):
        """All typed keys whose legacy text form equals ``legacy_key``."""
        candidates = []
        # An explicit caller claim always takes precedence.
        if legacy_key in self.claims:
            candidates.append(self.claims[legacy_key])

        # The plain string interpretation is always viable because the new
        # source may keep using text keys.
        if legacy_key not in candidates:
            candidates.append(legacy_key)

        # A numeric-looking legacy text could equally be a new integer key.
        integer_value = self._as_int(legacy_key)
        if integer_value is not None and integer_value not in candidates:
            candidates.append(integer_value)
        return candidates

    @staticmethod
    def _as_int(text):
        try:
            return int(text)
        except (TypeError, ValueError):
            return None

    def _resolve(self, legacy_key):
        """Decide the destination typed key for a legacy row.

        Returns ``(typed_key, ambiguous)``.  ``ambiguous`` is True when no
        single interpretation can be justified; the caller then parks the row
        in the held table rather than assigning it to any typed key.
        """
        candidates = self._candidate_typed_keys(legacy_key)
        if len(candidates) == 1:
            return candidates[0], False
        # Multiple viable interpretations: a caller claim is the only
        # authoritative tie-breaker for the legacy key's identity.
        if legacy_key in self.claims:
            return self.claims[legacy_key], False
        return None, True

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def run(self, max_batches=None):
        """Run (or resume) the migration, one batch at a time.

        ``max_batches`` optionally caps the batches processed in this call,
        useful for interleaving work or for testing interruption/resumption.
        Returns a :class:`MigrationReport`.
        """
        generation, already_complete, created = self._open_state()
        # A run "resumes" only when prior migration state already existed and
        # the work is not finished yet.
        resumed = not created and not already_complete
        # Adopt the persisted generation so every operation continues the same
        # key mapping instead of forking a parallel one.
        self._set_generation(generation)
        self._ensure_work_tables()
        self._release_held_if_claims_changed()

        report = MigrationReport(generation=generation, resumed=resumed)
        if already_complete:
            report.complete = True
            return report

        # One-time, idempotent split: move every legacy raw-text row out of the
        # live table into this generation's pending table.  Typed rows already
        # written by the new source stay in the live table.  The persisted flag
        # lets later runs skip this O(n) scan entirely.
        if not self._split_already_done():
            self._split_legacy_rows()
            self._mark_split_done()
            self.db.commit(blocking=True)

        batches_done = 0
        while max_batches is None or batches_done < max_batches:
            batch = self._next_batch()
            if not batch:
                break
            self._process_batch(batch, report)
            batches_done += 1
            # Each batch is durably committed before the next begins, so an
            # interruption only ever loses un-started work.
            self.db.commit(blocking=True)

        # Complete once nothing is pending and nothing remains held awaiting a
        # claim.  Held rows keep complete=False so a later, claim-driven run
        # can settle them.
        if self._count(self.pending_table) == 0 and \
                self._count(self.held_table) == 0:
            self._mark_complete(True)
            self.db.commit(blocking=True)
            report.complete = True
        return report

    def _split_legacy_rows(self):
        """Move all legacy raw-text rows from the live table into ``pending``.

        Idempotent and resumable: rows already moved are absent from the live
        table, and ``INSERT OR IGNORE`` makes re-splitting a no-op.  Typed rows
        (detected via the key codec) are left alone.  Moving a row out means a
        deleted legacy value can never be read from the old key space again.
        """
        watermark = -1
        while True:
            rows = list(self.db.conn.select(
                'SELECT rowid, key, value FROM "%s" WHERE rowid > ? '
                'ORDER BY rowid LIMIT ?' % self.table,
                (watermark, self.batch_size)))
            if not rows:
                break
            for rowid, stored_key, value in rows:
                watermark = max(watermark, rowid)
                if looks_like_encoded_key(stored_key):
                    # Already a typed row from the new source; leave it in place.
                    continue
                self.db.conn.execute(
                    'INSERT OR IGNORE INTO %s (legacy_key, value) VALUES (?, ?)'
                    % self.pending_table,
                    (stored_key, sqlite3.Binary(bytes(value))))
                self.db.conn.execute(
                    'DELETE FROM "%s" WHERE key = ?' % self.table, (stored_key,))
            # Commit each chunk so an interruption never has to redo a big move.
            self.db.commit(blocking=True)

    def _next_batch(self):
        """Return the next batch of pending legacy rows, in a stable order."""
        return list(self.db.conn.select(
            'SELECT legacy_key, value FROM %s ORDER BY legacy_key LIMIT ?'
            % self.pending_table, (self.batch_size,)))

    def _count(self, table):
        return self.db.conn.select_one('SELECT COUNT(*) FROM %s' % table)[0]

    def _process_batch(self, batch, report):
        for legacy_key, value in batch:
            typed_key, ambiguous = self._resolve(legacy_key)
            if ambiguous:
                self._hold(legacy_key, value)
                report.ambiguous.append(legacy_key)
                continue
            self._migrate_one(legacy_key, value, typed_key, report)

    def _hold(self, legacy_key, value):
        """Park a row that cannot be settled yet, preserving its exact bytes."""
        self.db.conn.execute(
            'INSERT OR IGNORE INTO %s (legacy_key, value) VALUES (?, ?)'
            % self.held_table,
            (legacy_key, sqlite3.Binary(bytes(value))))
        self.db.conn.execute(
            'DELETE FROM %s WHERE legacy_key = ?' % self.pending_table,
            (legacy_key,))

    def _migrate_one(self, legacy_key, value, typed_key, report):
        value_bytes = bytes(value)
        encoded = _MIGRATION_ENCODE_KEY(typed_key)

        # A typed row already written by the new source, or already produced
        # earlier in this run, owns that target.  Hold the contested row instead
        # of overwriting new data or letting one typed key read both values.
        if encoded in self._claimed_targets:
            self._hold(legacy_key, value)
            report.ambiguous.append(legacy_key)
            return
        existing = self.db.conn.select_one(
            'SELECT 1 FROM "%s" WHERE key = ?' % self.table, (encoded,))
        if existing is not None:
            self._hold(legacy_key, value)
            report.ambiguous.append(legacy_key)
            return

        # Publish the typed row into the live table, record the exact original
        # bytes for reconciliation, and remove the row from the pending space.
        self.db.conn.execute(
            'INSERT INTO "%s" (key, value) VALUES (?, ?)' % self.table,
            (encoded, sqlite3.Binary(value_bytes)))
        self.db.conn.execute(
            'INSERT INTO %s (legacy_key, value) VALUES (?, ?)'
            % self.tomb_table,
            (legacy_key, sqlite3.Binary(value_bytes)))
        self.db.conn.execute(
            'DELETE FROM %s WHERE legacy_key = ?' % self.pending_table,
            (legacy_key,))
        self._claimed_targets.add(encoded)
        report.migrated += 1
        report.tombstoned += 1

    # ------------------------------------------------------------------ #
    # Reconciliation
    # ------------------------------------------------------------------ #

    def reconcile(self, claims=None):
        """Compare value bytes before and after migration for claimed keys.

        For every explicitly claimed legacy key, the exact original value bytes
        held in the generation's tombstone are compared, item by item, with the
        bytes now stored under the typed key.  Returns a list of
        ``(legacy_key, typed_key, status)`` tuples where status is one of
        ``'ok'``, ``'mismatch'``, ``'missing-typed'`` (the typed row is absent)
        or ``'not-migrated'`` (no tombstone exists for the legacy key).
        """
        claim_map = _claims_to_map(claims) if claims is not None else self.claims
        # Bind to whatever generation the table actually migrated under, so a
        # brand new KeyMigration instance can still reconcile a prior run.
        self._bind_persisted_generation()
        results = []
        for legacy_key, typed_key in claim_map.items():
            tomb = self.db.conn.select_one(
                'SELECT value FROM %s WHERE legacy_key = ?'
                % self.tomb_table, (legacy_key,))
            if tomb is None:
                results.append((legacy_key, typed_key, 'not-migrated'))
                continue
            live = self.db.conn.select_one(
                'SELECT value FROM "%s" WHERE key = ?' % self.table,
                (_MIGRATION_ENCODE_KEY(typed_key),))
            if live is None:
                results.append((legacy_key, typed_key, 'missing-typed'))
            elif bytes(live[0]) == bytes(tomb[0]):
                results.append((legacy_key, typed_key, 'ok'))
            else:
                results.append((legacy_key, typed_key, 'mismatch'))
        return results


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

    def migrate_typed_keys(self, claims=None, batch_size=500, max_batches=None,
                           generation_id=None):
        """Migrate legacy text-key rows of this table to typed keys.

        Runs the resumable, batched migration implemented by
        :class:`KeyMigration`.  The dict must be opened with the typed key
        codec (``encode_key=sqlitedict.encode_key`` /
        ``decode_key=sqlitedict.decode_key``) and be writable.

        ``claims`` is an optional mapping of ``{legacy_text: typed_key}`` or an
        iterable of :class:`MigrationClaim` objects built from the caller's
        source manifest.  ``max_batches`` caps the work per call; call again to
        resume.  Returns a :class:`MigrationReport`.
        """
        migration = KeyMigration(
            self, claims=claims, batch_size=batch_size,
            generation_id=generation_id)
        return migration.run(max_batches=max_batches)

    def commit(self, blocking=True):
        """
        Persist all data to disk.

        When `blocking` is False, the commit command is queued, but the data is
        not guaranteed persisted (default implication when autocommit=True).
        """
        if self.conn is not None:
            self.conn.commit(blocking)
    sync = commit

    def close(self, do_log=True, force=False):
        if do_log:
            logger.debug("closing %s" % self)
        if hasattr(self, 'conn') and self.conn is not None:
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
