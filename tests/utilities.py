"""utility functions"""

from collections import Counter
from hashlib import md5
from pathlib import Path

from probables.constants import UINT64_T_MAX
from probables.hashes import KeyT


def calc_file_md5(filename: str | Path) -> str:
    """calc the md5 of a file"""
    with open(filename, "rb") as filepointer:
        res = filepointer.read()
    return md5(res).hexdigest()


def different_hash(key: KeyT, depth: int) -> list[int]:
    """the default fnv-1a hashing routine, but different"""

    def __fnv_1a(key: KeyT) -> int:
        """64 bit fnv-1a hash"""
        hval = 14695981039346656074  # made minor change
        fnv_64_prime = 1099511628211
        tmp = list(key) if not isinstance(key, str) else list(map(ord, key))
        for t_str in tmp:
            hval ^= t_str
            hval *= fnv_64_prime
            hval &= UINT64_T_MAX
        return hval

    res = []
    for _ in range(depth):
        res.append(__fnv_1a(key))
    return res


class RecordingBucket(list):
    """a cuckoo bucket that records every slot overwrite into a shared trace

    Each trace entry is `(bucket_idx, slot, old_value, new_value)`; cuckoo
    kicks and their rollback writes both show up as entries."""

    def __init__(self, values, bucket_idx, trace):
        super().__init__(values)
        self._bucket_idx = bucket_idx
        self._trace = trace

    def __setitem__(self, pos, value):
        self._trace.append((self._bucket_idx, pos, self[pos], value))
        super().__setitem__(pos, value)


def trace_cuckoo_writes(cko):
    """swap recording buckets into a cuckoo filter; returns the write trace"""
    trace = []
    cko._buckets = [RecordingBucket(bucket, idx, trace) for idx, bucket in enumerate(cko.buckets)]
    return trace


def find_unpaired_kicks(trace, max_swaps):
    """find kicks in a failed-insertion trace whose rollback write is missing

    The first `max_swaps` entries are the kicks; every later entry should be a
    rollback that restores a kicked slot to its original occupant."""
    kicks = trace[:max_swaps]
    rollbacks = list(trace[max_swaps:])
    unpaired = []
    for idx, pos, old, _new in kicks:
        match = next((r for r in rollbacks if r[0] == idx and r[1] == pos and r[3] == old), None)
        if match is None:
            unpaired.append((idx, pos, old))
        else:
            rollbacks.remove(match)
    return unpaired


def reconcile_cuckoo(ledger, buckets):
    """check a plain cuckoo filter against the confirmed-event ledger

    Args:
        ledger (dict): fingerprint -> original keys merged into that unit
        buckets (list): the filter buckets holding fingerprints
    Returns:
        dict: anomalies found, keyed by failure signature:
            `missing` committed fingerprints no longer present (lost kick rollback),
            `foreign` fingerprints in the table that were never committed,
            `duplicate` committed fingerprints migrated more than once"""
    counts = Counter(finger for bucket in buckets for finger in bucket)
    return {
        "missing": [fp for fp in ledger if counts.get(fp, 0) < 1],
        "foreign": [fp for fp in counts if fp not in ledger],
        "duplicate": [fp for fp, num in counts.items() if num > 1],
    }


def reconcile_counting_cuckoo(ledger, buckets):
    """check a counting cuckoo filter against the confirmed-event ledger

    Args:
        ledger (dict): fingerprint -> committed arrival count
        buckets (list): the filter buckets holding CountingCuckooBin objects
    Returns:
        dict: anomalies found, keyed by failure signature:
            `missing` committed fingerprints no longer present,
            `foreign` fingerprints in the table that were never committed,
            `duplicate` fingerprints stored in more than one bin,
            `count_drift` membership unchanged but committed count differs"""
    positions = {}
    for idx, bucket in enumerate(buckets):
        for ccb in bucket:
            positions.setdefault(ccb.finger, []).append((idx, ccb.count))
    return {
        "missing": [fp for fp in ledger if fp not in positions],
        "foreign": [fp for fp in positions if fp not in ledger],
        "duplicate": [fp for fp, locs in positions.items() if len(locs) > 1],
        "count_drift": [
            fp for fp, locs in positions.items() if len(locs) == 1 and fp in ledger and locs[0][1] != ledger[fp]
        ],
    }
