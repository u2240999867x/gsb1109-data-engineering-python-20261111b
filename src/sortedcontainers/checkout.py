"""Range checkout ledger built on top of a SortedList.

A :class:`CheckoutLedger` lets backfill workers *claim* a contiguous range of
equal-key rows from a :class:`SortedList`, process them outside the main
collection, and later either *commit* them (confirm the attribution change) or
*return* them to the main collection.

Design notes
------------

Rows are never physically removed from the backing :class:`SortedList`.
Instead every row is stamped once with a stable, monotonically increasing
``seq`` (its original row number) and a status marker:

* ``STATUS_AVAILABLE`` -- claimable, held by the main collection;
* ``STATUS_PENDING`` -- claimed by a checkout, no longer claimable;
* ``STATUS_COMMITTED`` -- confirmed attribution change, no longer claimable.

Because the rows stay in sorted position, returning a range only has to touch
the rows of that range: the surrounding records are never copied. New rows
that arrive after a claim are stamped with later sequence numbers, so when an
earlier claim is returned its rows sort back *before* the newly arrived
equal-key rows, preserving the original equal-key order.

The three live views (available / pending / committed) are reconciled against
the stamped sequence numbers instead of rebuilding or copying the rest of the
collection.

"""

from itertools import count as _count

from .sortedlist import SortedKeyList, SortedList

STATUS_AVAILABLE = 0
STATUS_PENDING = 1
STATUS_COMMITTED = 2


class CheckoutError(RuntimeError):
    """Raised when a checkout cannot be claimed or resolved."""


class Checkout:
    """A single claimed key-range handle.

    Instances are created by :meth:`CheckoutLedger.claim` and are opaque to
    callers apart from their attributes. A checkout moves through the states
    ``'pending'`` -> ``'committed'`` / ``'returned'`` and never changes state
    twice, so repeating a confirmation only produces one attribution change.

    """

    __slots__ = ('id', 'lo', 'hi', 'inclusive', 'seqs', 'state', '_ledger_ref')

    def __init__(self, checkout_id, lo, hi, inclusive, seqs, ledger):
        self.id = checkout_id
        self.lo = lo
        self.hi = hi
        self.inclusive = inclusive
        self.seqs = seqs
        self.state = 'pending'
        self._ledger_ref = ledger

    @property
    def rows(self):
        """Rows captured by this checkout, in sorted order."""
        ledger = self._ledger_ref
        return [ledger._by_seq[seq].row for seq in self.seqs]

    def __repr__(self):
        return (
            f'Checkout(id={self.id!r}, lo={self.lo!r}, hi={self.hi!r}, '
            f'inclusive={self.inclusive!r}, state={self.state!r})'
        )


class CheckoutLedger:
    """Claim, commit and return contiguous key ranges of a SortedList.

    The ledger wraps a :class:`SortedList` (or :class:`SortedKeyList`) of
    rows. Rows may compare equal; equal rows keep a stable arrival order
    recorded by their sequence numbers.

    >>> ledger = CheckoutLedger(SortedList([1, 2, 2, 3]))
    >>> checkout = ledger.claim(2, 2)
    >>> checkout.rows
    [2, 2]
    >>> ledger.available()
    [1, 3]
    >>> ledger.commit(checkout)
    Checkout(id=1, lo=2, hi=2, inclusive=(True, True), state='committed')
    >>> ledger.committed()
    [2, 2]

    """

    def __init__(self, source, key=None):
        """Initialize a ledger around an existing sorted list.

        :param source: a :class:`SortedList`, :class:`SortedKeyList` or an
            iterable of initial rows
        :param key: optional key function, used when `source` is a bare
            iterable to build a :class:`SortedKeyList`

        """
        if not isinstance(source, SortedList):
            rows = list(source)
            if key is not None:
                source = SortedKeyList(rows, key=key)
            else:
                source = SortedList(rows)

        self._source = source
        self._outer_key = source.key if isinstance(source, SortedKeyList) else None
        entry_key = _Entry.key_for(self._outer_key)
        self._store = SortedKeyList(key=entry_key)

        self._seq_counter = _count()
        self._by_seq = {}
        self._checkouts = {}
        self._id_counter = _count(1)
        self._closed = False

        for row in source:
            self._admit(row)

    # ------------------------------------------------------------------
    # Intake
    # ------------------------------------------------------------------

    def _admit(self, row):
        """Stamp one new row and file it as available."""
        seq = next(self._seq_counter)
        entry = _Entry(row, seq, STATUS_AVAILABLE)
        self._store.add(entry)
        self._by_seq[seq] = entry
        return entry

    def add(self, row):
        """Accept a new row into the main collection.

        Rows added while checkouts are pending stay in the main collection:
        equal-key newcomers are stamped with a later sequence and therefore
        remain claimable and sort after any returned earlier equal-key rows.

        """
        if self._closed:
            raise CheckoutError('ledger is closed')
        self._source.add(row)
        return self._admit(row)

    @property
    def source(self):
        """The backing sorted list, kept in sync with every admitted row."""
        return self._source

    # ------------------------------------------------------------------
    # Key boundary helpers
    # ------------------------------------------------------------------

    def _bounds(self, lo, hi, inclusive):
        store = self._store
        lo_inclusive, hi_inclusive = inclusive

        # Bounds are comparison keys. For a plain SortedList the key is the
        # row itself; for a SortedKeyList the caller passes extracted keys.
        left = store.bisect_key_left(lo) if lo_inclusive else store.bisect_key_right(lo)
        right = (
            store.bisect_key_right(hi) if hi_inclusive else store.bisect_key_left(hi)
        )

        return left, right

    # ------------------------------------------------------------------
    # Claim
    # ------------------------------------------------------------------

    def claim(self, lo, hi=None, inclusive=(True, True)):
        """Claim every currently available row in the contiguous key range.

        The range is ``[lo, hi]`` by default; pass ``inclusive=(True,
        False)`` for a half-open range. Only rows currently in the available
        state are claimed -- rows held by another checkout or already
        committed are skipped. Claims therefore never overlap on the same
        rows, even when their key ranges coincide, and new equal-key rows
        that arrive after a claim can be claimed separately.

        For a :class:`SortedKeyList`, `lo` and `hi` are comparison keys.

        :raises CheckoutError: if the range contains no available row

        """
        if self._closed:
            raise CheckoutError('ledger is closed')
        if hi is None:
            hi = lo

        left, right = self._bounds(lo, hi, inclusive)

        if left >= right:
            raise CheckoutError(f'no rows in range [{lo!r}, {hi!r}]')

        claimed = [
            entry
            for entry in self._store[left:right]
            if entry.status == STATUS_AVAILABLE
        ]

        if not claimed:
            raise CheckoutError(f'no available rows in range [{lo!r}, {hi!r}]')

        for entry in claimed:
            entry.status = STATUS_PENDING

        seqs = tuple(entry.seq for entry in claimed)
        checkout = Checkout(next(self._id_counter), lo, hi, inclusive, seqs, self)
        self._checkouts[checkout.id] = checkout
        return checkout

    # ------------------------------------------------------------------
    # Resolve
    # ------------------------------------------------------------------

    def _resolve(self, checkout, terminal_state, new_status):
        if not isinstance(checkout, Checkout):
            raise TypeError('expected a Checkout handle')
        if checkout._ledger_ref is not self:
            raise CheckoutError('checkout belongs to another ledger')
        if checkout.state != 'pending':
            # Repeated confirmation is idempotent: a second call changes
            # nothing and never moves rows twice.
            return checkout

        entries = []
        for seq in checkout.seqs:
            entry = self._by_seq.get(seq)
            if entry is None or entry.status != STATUS_PENDING:
                # Validate before mutating so a failed resolution loses no
                # rows and leaves the checkout fully pending.
                raise CheckoutError(
                    f'checkout {checkout.id!r} is corrupt at seq {seq!r}'
                )
            entries.append(entry)

        for entry in entries:
            entry.status = new_status

        checkout.state = terminal_state
        return checkout

    def commit(self, checkout):
        """Confirm the attribution change for `checkout`.

        Its rows become committed. Calling :meth:`commit` or
        :meth:`return_checkout` again on the same handle is a no-op.

        """
        return self._resolve(checkout, 'committed', STATUS_COMMITTED)

    def return_checkout(self, checkout):
        """Return `checkout` rows to the main collection.

        Equal-key order is restored from the original sequence numbers:
        returned rows sort before equal-key rows that arrived later. A
        failed return raises and leaves every row intact and still pending.

        """
        return self._resolve(checkout, 'returned', STATUS_AVAILABLE)

    # ------------------------------------------------------------------
    # Reconciliation views, keyed by original row number
    # ------------------------------------------------------------------

    def _view(self, status):
        return [entry.row for entry in self._store if entry.status == status]

    def available(self):
        """Rows currently claimable or held by the main collection."""
        return self._view(STATUS_AVAILABLE)

    def pending(self):
        """Rows currently held by an unresolved checkout."""
        return self._view(STATUS_PENDING)

    def committed(self):
        """Rows whose attribution change has been confirmed."""
        return self._view(STATUS_COMMITTED)

    def pending_checkouts(self):
        """Handles still in the pending state, ordered by checkout id."""
        return [
            checkout
            for checkout in self._checkouts.values()
            if checkout.state == 'pending'
        ]

    def reconcile(self):
        """Reconcile every row by original row number.

        Returns a triple of ``(available, pending, committed)`` lists of
        ``(seq, row)`` pairs covering every admitted row exactly once. The
        three sequence sets must partition the full sequence space, which
        makes dropped or double-counted rows visible immediately.

        """
        buckets = {
            STATUS_AVAILABLE: [],
            STATUS_PENDING: [],
            STATUS_COMMITTED: [],
        }
        for entry in self._store:
            buckets[entry.status].append((entry.seq, entry.row))

        for bucket in buckets.values():
            bucket.sort(key=lambda pair: pair[0])

        seqs = [seq for bucket in buckets.values() for seq, _ in bucket]
        if sorted(seqs) != list(range(len(self._by_seq))):
            raise CheckoutError(
                'ledger reconciliation failed: rows missing or duplicated'
            )

        return (
            buckets[STATUS_AVAILABLE],
            buckets[STATUS_PENDING],
            buckets[STATUS_COMMITTED],
        )

    def close(self):
        """Reject further claims or additions; unresolved rows stay pending."""
        self._closed = True

    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return (
            f'CheckoutLedger(rows={len(self._store)}, checkouts={len(self._checkouts)})'
        )


class _Entry:
    """Internal envelope pairing a row with its sequence number and status."""

    __slots__ = ('row', 'seq', 'status')

    def __init__(self, row, seq, status):
        self.row = row
        self.seq = seq
        self.status = status

    @staticmethod
    def key_for(outer_key):
        if outer_key is None:
            return _entry_key
        return lambda entry: outer_key(entry.row)

    def __repr__(self):
        return f'_Entry(row={self.row!r}, seq={self.seq}, status={self.status})'


def _entry_key(entry):
    return entry.row
