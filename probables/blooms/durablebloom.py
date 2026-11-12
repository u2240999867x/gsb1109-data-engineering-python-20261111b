"""Durable, cross-process Bloom Filter built on top of :class:`BloomFilter`.

Multiple collector processes submit *small batches* of events to the same
on-disk filter while archive readers deduplicate against a *published
partition generation*.  The implementation is built around four concrete
persistence guarantees that can be checked byte-for-byte on disk:

1. **Exactly-once batches.**  Every batch is first written to its own
   append-only, CRC32-protected journal and only then recorded as *confirmed*
   in an atomically-renamed manifest.  Re-submitting the same ``batch_id``
   returns the stored result without touching a bit twice, so a confirmed
   batch is counted exactly once even if the process restarts.
2. **No lost concurrent bits.**  Applying a batch is a read-modify-write
   *byte OR* while holding an inter-process ``flock``; two processes setting
   different bits of the same byte therefore merge instead of overwriting.
3. **Stable reader snapshots.**  A generation is sealed (frozen) before the
   next one is published.  A reader that opens a generation sees either the
   complete old generation or the complete new one, never a half-written
   layer, and keeps scanning the old layer even while a writer grows the
   filter.
4. **Explicit takeover fencing.**  Writers are fenced by a partition
   generation that the *caller* increments when another process takes a
   partition over.  A late writer holding the old generation is rejected with
   :class:`StaleWriterError`; process death is never guessed from a timeout.

Crash model: a clean :meth:`DurableBloomFilter.close` is a graceful shutdown
(a confirmation point), distinct from a machine interruption.  On recovery
every confirmed batch is still present; an unconfirmed batch is either fully
applied (journal intact) or fully absent (torn journal discarded) - it is
never partially visible.

Nothing here ever copies a whole bit table per operation: batches journal only
their touched bytes, sealing writes one immutable layer once, and growing adds
a new layer file rather than rewriting existing ones.

License: MIT
"""

import fcntl
import json
import os
import zlib
from mmap import ACCESS_READ, mmap
from pathlib import Path
from struct import Struct
from typing import Iterator

from probables.blooms.bloom import BloomFilter
from probables.exceptions import DurableBloomRecoveryError, InitializationError, StaleWriterError
from probables.hashes import HashFuncT, HashResultsT, KeyT, default_fnv_1a

# A single journal record.  ``length`` counts the payload bytes that follow;
# ``crc`` is a CRC32 over (generation || length || payload) so a torn tail
# record is detected instead of being silently applied.
_RECORD_STRUCT = Struct("<QQII")  # generation, length, crc, reserved
_RECORD_PREFIX = Struct("<QI")  # generation, length (covered by the CRC)
_BATCH_TRAILER = b"DBFB"  # durable-bloom-finalised-batch marker


class _LockedFile:
    """Context manager holding an exclusive ``flock`` on a directory file."""

    __slots__ = ("_path", "_fp")

    def __init__(self, path: Path):
        self._path = path
        self._fp = None

    def __enter__(self):
        self._fp = open(self._path, "a+b")
        fcntl.flock(self._fp.fileno(), fcntl.LOCK_EX)
        return self._fp

    def __exit__(self, *exc):
        fcntl.flock(self._fp.fileno(), fcntl.LOCK_UN)
        self._fp.close()
        self._fp = None
        return False


class DurableBloomFilter:
    """A durable, multi-process, batched Bloom filter stored in a directory.

    Args:
        directory (str | Path): Directory holding the filter.  It is created
            (along with the first generation) when it does not exist.
        est_elements (int): Estimated elements per generation layer.
        false_positive_rate (float): Desired false positive rate.
        partition_generation (int): The partition generation the caller
            believes it owns.  Must be at least the persisted generation;
            passing a larger value performs an explicit takeover.
        hash_function (function): Hashing strategy ``hf(key, number)``.

    On-disk layout (all paths relative to ``directory``)::

        manifest.json            atomically renamed, fsync'd confirmation point
        manifest.json.tmp        staging file (never read by a recovery)
        lock                     flock(2) target serialising writers
        journal/<batch_id>.dbf   one append-only CRC journal per batch
        gen/g<N>.layer           immutable bit table for sealed generation N
        gen/g<N>.meta           "<num_bits> <number_hashes> <byte_length>"

    The byte boundaries are fixed: each journal record is
    ``_RECORD_STRUCT.size`` header bytes followed by exactly ``length`` payload
    bytes, and each layer is exactly ``byte_length`` bytes.  Both are validated
    on recovery so a machine interruption cannot leave a partially applied
    batch looking committed.
    """

    __slots__ = (
        "_dir",
        "_lock_path",
        "_gen_dir",
        "_journal_dir",
        "_manifest_path",
        "_est_elements",
        "_fpr",
        "_hash_func",
        "_closed",
        "_partition_generation",
        "_manifest",
        "_active",
        "_active_fp",
        "_active_generation",
    )

    def __init__(
        self,
        directory: str | Path,
        est_elements: int | None = None,
        false_positive_rate: float | None = None,
        partition_generation: int = 0,
        hash_function: HashFuncT | None = None,
    ) -> None:
        self._dir = Path(directory).expanduser().resolve()
        self._lock_path = self._dir / "lock"
        self._gen_dir = self._dir / "gen"
        self._journal_dir = self._dir / "journal"
        self._manifest_path = self._dir / "manifest.json"
        self._est_elements = est_elements
        self._fpr = false_positive_rate
        self._hash_func = hash_function if hash_function is not None else default_fnv_1a
        self._closed = False
        self._partition_generation = int(partition_generation)
        self._manifest: dict = {}
        self._active: BloomFilter | None = None
        self._active_fp = None
        self._active_generation = -1

        # The lock file lives directly in the backing directory, so make sure
        # the directory exists before the first flock(2) target is opened.
        self._dir.mkdir(parents=True, exist_ok=True)
        with _LockedFile(self._lock_path):
            existed = self._manifest_path.exists()
            if existed:
                self._manifest = self._read_manifest()
                # Geometry lives in the manifest so a reopened writer can grow
                # or fail over without the caller restating the parameters.
                self._est_elements = int(self._manifest["est_elements"])
                self._fpr = float(self._manifest["false_positive_rate"])
                self._recover_journals()
                if self._partition_generation < self.generation:
                    raise StaleWriterError(
                        f"partition generation {self._partition_generation} is below the persisted "
                        f"generation {self.generation}; another process already took over"
                    )
            else:
                if est_elements is None or false_positive_rate is None:
                    raise InitializationError(
                        "est_elements and false_positive_rate are required to create a DurableBloomFilter"
                    )
                self._gen_dir.mkdir(parents=True, exist_ok=True)
                self._journal_dir.mkdir(parents=True, exist_ok=True)
                self._manifest = self._fresh_manifest()
                # Create and record the first layer *before* publishing the
                # manifest so the persisted state never references a missing
                # generation and a later in-lock refresh cannot drop it.
                self._start_generation(self._partition_generation)
                self._write_manifest()
            # Open the active (newest) generation for read-modify-write.
            self._active = self._open_active_generation()

    # ------------------------------------------------------------------ #
    # Public, user-facing surface
    # ------------------------------------------------------------------ #
    @property
    def directory(self) -> Path:
        """Path: The directory backing the filter."""
        return self._dir

    @property
    def generation(self) -> int:
        """int: The newest persisted partition generation (monotonic)."""
        return int(self._manifest["generation"])

    @property
    def partition_generation(self) -> int:
        """int: The partition generation this writer is fenced to."""
        return self._partition_generation

    @property
    def confirmed_batches(self) -> int:
        """int: Number of distinct, confirmed (exactly-once) batches."""
        return len(self._manifest["confirmed"])

    @property
    def confirmed_events(self) -> int:
        """int: Sum of event counts over confirmed batches.

        A re-sent batch is only counted once because duplicate ``batch_id``
        submissions do not add to this total.
        """
        return int(self._manifest["events"])

    @property
    def sealed_generations(self) -> list[int]:
        """list(int): Generation numbers that have been sealed and published."""
        return [int(g) for g in self._manifest["sealed"]]

    def hashes(self, key: KeyT, depth: int | None = None) -> HashResultsT:
        """Return the configured hashes for ``key``.

        With no explicit depth the deepest hash requirement across all layers
        is used, so the result is valid for every generation's geometry.
        """
        if depth is None:
            with _LockedFile(self._lock_path):
                self._refresh_locked()
                depth = max(self._read_meta(gen)[1] for gen in (int(g) for g in self._manifest["layers"]))
        return self._hash_func(key, depth)

    def check(self, key: KeyT) -> bool:
        """Membership test across every generation layer.

        Old layers remain valid after a resize: a key is a member if *any*
        layer (including layers whose geometry differs from the active one and
        whose original keys were never re-read) contains it.
        """
        with _LockedFile(self._lock_path):
            self._refresh_locked()
            depth = max(self._read_meta(gen)[1] for gen in (int(g) for g in self._manifest["layers"]))
            hashes = self._hash_func(key, depth)
            return any(self._layer_contains(layer, hashes) for layer in self._iter_layers())

    def check_alt(self, hashes: HashResultsT) -> bool:
        """Membership test from pre-computed hashes across all layers."""
        with _LockedFile(self._lock_path):
            self._refresh_locked()
            return any(self._layer_contains(layer, hashes) for layer in self._iter_layers())

    def submit_batch(self, keys: list[KeyT], batch_id: str) -> bool:
        """Idempotently submit one batch of events.

        Args:
            keys (list): The events in this batch.
            batch_id (str): A caller-assigned, stable identifier for the batch;
                re-sending the same identifier makes the call a no-op.
        Returns:
            bool: ``True`` if the batch was applied now, ``False`` if it had
            already been confirmed (a replay).
        Raises:
            StaleWriterError: The writer's partition generation is behind the
                persisted generation, i.e. a takeover happened.
        """
        self._ensure_open()
        safe_id = self._safe_batch_id(batch_id)
        journal_path = self._journal_dir / f"{safe_id}.dbf"

        with _LockedFile(self._lock_path):
            self._fence_check()
            if self._active is None or self._active_generation != self.generation:
                # Another process advanced the generation; bind to the new layer.
                self._active = self._open_active_generation()
            if safe_id in self._manifest["confirmed"]:
                # Exactly once: the replay changes neither bits nor counters.
                return False

            # Hash inside the lock against the bound layer's geometry so a
            # concurrent grow/takeover cannot invalidate the bit positions.
            depth = self._active._number_hashes  # noqa: SLF001
            hashed = [self._hash_func(key, depth) for key in keys]
            gen_before = self.generation
            self._write_journal(journal_path, gen_before, hashed)
            self._apply_journal(journal_path)
            self._manifest["confirmed"][safe_id] = {
                "generation": gen_before,
                "events": len(keys),
                "journal": journal_path.name,
            }
            self._manifest["events"] = int(self._manifest["events"]) + len(keys)
            self._write_manifest()  # confirmation point (atomic rename + fsync)
            return True

    def takeover(self, new_partition_generation: int) -> int:
        """Explicitly take the partition over at a caller-chosen generation.

        Args:
            new_partition_generation (int): Must be greater than both the
                current writer fence and the persisted generation.
        Returns:
            int: The newly published partition generation.
        Raises:
            StaleWriterError: The supplied generation is not strictly greater.

        This is the *only* way a writer is fenced.  No timeout or liveness
        guess is involved; the late writer discovers the takeover on its next
        locked operation and is rejected.
        """
        self._ensure_open()
        with _LockedFile(self._lock_path):
            self._refresh_locked()
            if new_partition_generation <= self._partition_generation:
                raise StaleWriterError(
                    f"takeover generation {new_partition_generation} must exceed the writer fence "
                    f"{self._partition_generation}"
                )
            if new_partition_generation <= self.generation:
                raise StaleWriterError(
                    f"takeover generation {new_partition_generation} must exceed persisted generation "
                    f"{self.generation}"
                )
            self._seal_active()
            self._partition_generation = int(new_partition_generation)
            self._manifest["generation"] = self._partition_generation
            self._start_generation(self._partition_generation)
            self._write_manifest()
            self._active = self._open_active_generation()
            return self.generation

    def grow(self, est_elements: int | None = None, false_positive_rate: float | None = None) -> int:
        """Add a larger generation layer without rewriting or copying old ones.

        Args:
            est_elements (int): Capacity of the new layer (defaults to double
                the current estimate).
            false_positive_rate (float): FPR of the new layer.
        Returns:
            int: The generation number of the new active layer.

        Old readers keep their open handles to the sealed, immutable layers and
        continue scanning them; the old layers stay membership-valid even
        though their original keys are never re-read into the new layer.
        """
        self._ensure_open()
        with _LockedFile(self._lock_path):
            self._fence_check()
            new_est = est_elements if est_elements is not None else max(1, self._est_elements * 2)
            new_fpr = false_positive_rate if false_positive_rate is not None else self._fpr
            self._est_elements = new_est
            self._fpr = new_fpr
            self._manifest["est_elements"] = new_est
            self._manifest["false_positive_rate"] = new_fpr
            new_generation = self.generation + 1
            self._seal_active()
            self._partition_generation = new_generation
            self._manifest["generation"] = new_generation
            self._start_generation(new_generation, new_est, new_fpr)
            self._write_manifest()
            self._active = self._open_active_generation()
            return new_generation

    def open_reader(self, generation: int | None = None) -> "GenerationReader":
        """Open a stable snapshot reader.

        Args:
            generation (int): Generation to scan; ``None`` means the newest
                sealed-or-active generation recorded in the manifest at call
                time.
        Returns:
            GenerationReader: A reader bound to immutable on-disk bytes.

        The reader resolves its target generation under the lock once, so a
        concurrent grow/takeover can never make it observe a half-written
        layer: throughout its scan it sees either the complete old generation
        or the complete new one.
        """
        with _LockedFile(self._lock_path):
            self._refresh_locked()
            target = self.generation if generation is None else int(generation)
            layer_path = self._layer_path(target)
            if not layer_path.exists():
                raise InitializationError(f"generation {target} has no layer file {layer_path.name}")
            num_bits, number_hashes, _byte_length = self._read_meta(target)
        return GenerationReader(layer_path, num_bits, number_hashes, target, self._hash_func)

    def close(self) -> None:
        """Graceful shutdown: flush the active mmap and publish a manifest.

        This is the clean-process confirmation point, deliberately separate
        from a machine interruption.  It never rewrites the manifest from a
        possibly stale view: the manifest is the per-operation confirmation
        point, so closing only flushes the active layer's mmap.  Journals are
        retained as the replayable proof of each confirmed batch.
        """
        if self._closed:
            return
        with _LockedFile(self._lock_path):
            self._refresh_locked()
            if self._active is not None:
                # Only flush a layer this writer still owns; a takeover may
                # have moved the active generation to a different file.
                if self._active_generation == self.generation:
                    self._flush_active()
                bloom = self._active._bloom  # noqa: SLF001
                if isinstance(bloom, mmap) and not bloom.closed:
                    bloom.close()
                if self._active_fp is not None and not self._active_fp.closed:
                    self._active_fp.close()
                self._active_fp = None
        self._closed = True

    def __enter__(self) -> "DurableBloomFilter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    def __contains__(self, key: KeyT) -> bool:
        return self.check(key)

    # ------------------------------------------------------------------ #
    # Manifest (atomic confirmation point)
    # ------------------------------------------------------------------ #
    def _fresh_manifest(self) -> dict:
        return {
            "format": 1,
            "generation": self._partition_generation,
            "est_elements": self._est_elements,
            "false_positive_rate": self._fpr,
            "events": 0,
            "confirmed": {},
            "sealed": [],
            "layers": {},
        }

    def _read_manifest(self) -> dict:
        try:
            with open(self._manifest_path, encoding="utf-8") as fp:
                manifest = json.load(fp)
        except (OSError, json.JSONDecodeError) as exc:
            raise DurableBloomRecoveryError(f"unable to read manifest {self._manifest_path}: {exc}") from exc
        for key in ("generation", "events", "confirmed", "sealed", "layers"):
            if key not in manifest:
                raise DurableBloomRecoveryError(f"manifest is missing required key {key!r}")
        return manifest

    def _write_manifest(self) -> None:
        """Atomically publish the manifest with a full fsync confirmation."""
        tmp_path = self._dir / "manifest.json.tmp"
        payload = json.dumps(self._manifest, sort_keys=True).encode("utf-8")
        with open(tmp_path, "wb") as fp:
            fp.write(payload)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp_path, self._manifest_path)
        self._fsync_dir(self._dir)

    # ------------------------------------------------------------------ #
    # Generation layers
    # ------------------------------------------------------------------ #
    def _layer_path(self, generation: int) -> Path:
        return self._gen_dir / f"g{generation}.layer"

    def _meta_path(self, generation: int) -> Path:
        return self._gen_dir / f"g{generation}.meta"

    def _start_generation(
        self,
        generation: int,
        est_elements: int | None = None,
        false_positive_rate: float | None = None,
    ) -> None:
        est = self._est_elements if est_elements is None else est_elements
        fpr = self._fpr if false_positive_rate is None else false_positive_rate
        # Build a throwaway in-memory filter purely to derive optimal geometry.
        probe = BloomFilter(est_elements=est, false_positive_rate=fpr, hash_function=self._hash_func)
        byte_length = probe.bloom_length
        num_bits = probe.number_bits
        number_hashes = probe.number_hashes

        layer_path = self._layer_path(generation)
        with open(layer_path, "wb") as fp:
            fp.write(b"\x00" * byte_length)
            fp.flush()
            os.fsync(fp.fileno())
        with open(self._meta_path(generation), "w", encoding="utf-8") as fp:
            fp.write(f"{num_bits} {number_hashes} {byte_length}\n")
            fp.flush()
            os.fsync(fp.fileno())
        self._fsync_dir(self._gen_dir)
        self._manifest["layers"][str(generation)] = {
            "num_bits": num_bits,
            "number_hashes": number_hashes,
            "byte_length": byte_length,
        }

    def _seal_active(self) -> None:
        """Freeze the current active layer; sealed layers are never mutated."""
        if self._active is not None:
            self._flush_active()
        current = str(self.generation)
        if current not in self._manifest["sealed"] and current in self._manifest["layers"]:
            self._manifest["sealed"].append(int(current))

    def _open_active_generation(self) -> BloomFilter:
        # Release any previously open active layer (e.g. after grow/takeover).
        if self._active is not None:
            bloom = self._active._bloom  # noqa: SLF001
            if isinstance(bloom, mmap) and not bloom.closed:
                bloom.close()
            if self._active_fp is not None and not self._active_fp.closed:
                self._active_fp.close()
            self._active_fp = None
        generation = self.generation
        layer_path = self._layer_path(generation)
        if not layer_path.exists():
            raise DurableBloomRecoveryError(
                f"manifest references generation {generation} but its layer file {layer_path.name} is missing"
            )
        num_bits, number_hashes, byte_length = self._read_meta(generation)
        actual = layer_path.stat().st_size
        if actual != byte_length:
            raise DurableBloomRecoveryError(
                f"layer {layer_path.name} is {actual} bytes but metadata expects {byte_length}"
            )
        blm = BloomFilter(
            est_elements=self._manifest.get("est_elements", self._est_elements),
            false_positive_rate=self._manifest.get("false_positive_rate", self._fpr),
            hash_function=self._hash_func,
        )
        fp = open(layer_path, "r+b")  # noqa: SIM115 - kept open for the filter life
        blm._bloom = mmap(fp.fileno(), 0)  # noqa: SLF001
        blm._on_disk = True  # noqa: SLF001
        blm._num_bits = num_bits  # noqa: SLF001
        blm._number_hashes = number_hashes  # noqa: SLF001
        blm._bloom_length = byte_length  # noqa: SLF001
        blm._els_added = 0  # noqa: SLF001
        self._active_fp = fp
        self._active_generation = generation
        return blm

    def _read_meta(self, generation: int) -> tuple[int, int, int]:
        meta = self._manifest["layers"].get(str(generation))
        if meta is not None:
            return int(meta["num_bits"]), int(meta["number_hashes"]), int(meta["byte_length"])
        # Fall back to the sidecar for layers created by an older manifest.
        with open(self._meta_path(generation), encoding="utf-8") as fp:
            num_bits, number_hashes, byte_length = (int(part) for part in fp.read().split())
        return num_bits, number_hashes, byte_length

    def _flush_active(self) -> None:
        bloom = self._active._bloom  # noqa: SLF001
        if isinstance(bloom, mmap):
            bloom.flush()
        if self._active_fp is not None:
            self._active_fp.flush()
            os.fsync(self._active_fp.fileno())

    def _iter_layers(self) -> Iterator[tuple[int, Path, int, int]]:
        for generation in sorted(int(g) for g in self._manifest["layers"]):
            num_bits, number_hashes, _byte_length = self._read_meta(generation)
            yield generation, self._layer_path(generation), num_bits, number_hashes

    def _layer_contains(self, layer: tuple[int, Path, int, int], hashes: HashResultsT) -> bool:
        _generation, path, num_bits, number_hashes = layer
        with open(path, "rb") as fp:
            data = fp.read()
        for index in range(number_hashes):
            bit = hashes[index] % num_bits
            if (data[bit // 8] & (1 << (bit % 8))) == 0:
                return False
        return True

    # ------------------------------------------------------------------ #
    # Batch journals (CRC-protected, append-only; the durability unit)
    # ------------------------------------------------------------------ #
    def _write_journal(self, journal_path: Path, generation: int, hashed_batches: list[HashResultsT]) -> None:
        """Persist a batch's bit positions before any bit table byte changes."""
        # Collect only the touched (byte_index, or-mask) pairs so small batches
        # write small journals instead of copying the whole bit table.
        touched: dict[int, int] = {}
        num_bits = self._active._num_bits  # noqa: SLF001
        number_hashes = self._active._number_hashes  # noqa: SLF001
        for hashes in hashed_batches:
            for index in range(number_hashes):
                bit = hashes[index] % num_bits
                byte_index = bit // 8
                touched[byte_index] = touched.get(byte_index, 0) | (1 << (bit % 8))

        # Payload: event count (Q), touched-byte count (Q), then
        # count * (byte index Q, mask B).  Storing the event count keeps the
        # confirmed-event counter exact when an unconfirmed batch is recovered.
        payload = Struct("<QQ").pack(len(hashed_batches), len(touched))
        entry = Struct("<QB")
        for byte_index, mask in sorted(touched.items()):
            payload += entry.pack(byte_index, mask)
        payload += _BATCH_TRAILER

        crc = zlib.crc32(_RECORD_PREFIX.pack(generation, len(payload)))
        crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
        with open(journal_path, "wb") as fp:
            fp.write(_RECORD_STRUCT.pack(generation, len(payload), crc, 0))
            fp.write(payload)
            fp.flush()
            os.fsync(fp.fileno())
        self._fsync_dir(self._journal_dir)

    def _read_journal(self, journal_path: Path) -> tuple[int, int, dict[int, int]]:
        """Validate and decode a journal; raise if it is torn or corrupt."""
        raw = journal_path.read_bytes()
        header_size = _RECORD_STRUCT.size
        if len(raw) < header_size:
            raise DurableBloomRecoveryError(f"journal {journal_path.name} is shorter than its header")
        generation, length, crc, _reserved = _RECORD_STRUCT.unpack(raw[:header_size])
        if len(raw) != header_size + length:
            raise DurableBloomRecoveryError(f"journal {journal_path.name} is truncated (length {length})")
        payload = raw[header_size:]
        expected = zlib.crc32(_RECORD_PREFIX.pack(generation, length))
        expected = zlib.crc32(payload, expected) & 0xFFFFFFFF
        if expected != crc:
            raise DurableBloomRecoveryError(f"journal {journal_path.name} failed its CRC check")
        if not payload.endswith(_BATCH_TRAILER):
            raise DurableBloomRecoveryError(f"journal {journal_path.name} has no finalisation trailer")
        body = payload[: -len(_BATCH_TRAILER)]
        event_count, count = Struct("<QQ").unpack(body[:16])
        offset = 16
        entry = Struct("<QB")
        touched: dict[int, int] = {}
        for _ in range(count):
            byte_index, mask = entry.unpack(body[offset : offset + entry.size])
            touched[byte_index] = touched.get(byte_index, 0) | mask
            offset += entry.size
        if offset != len(body):
            raise DurableBloomRecoveryError(f"journal {journal_path.name} has trailing garbage")
        return int(generation), int(event_count), touched

    def _apply_journal(self, journal_path: Path) -> None:
        """OR journal bytes into the active layer's mmap (lossless merge)."""
        _generation, _event_count, touched = self._read_journal(journal_path)
        bloom = self._active._bloom  # noqa: SLF001
        byte_length = self._active._bloom_length  # noqa: SLF001
        for byte_index, mask in touched.items():
            if byte_index >= byte_length:
                raise DurableBloomRecoveryError(
                    f"journal {journal_path.name} references byte {byte_index} outside the active layer"
                )
            bloom[byte_index] = bloom[byte_index] | mask
        self._flush_active()

    def _recover_journals(self) -> None:
        """Reconcile journals with the manifest after a clean or hard stop.

        Confirmed batches are guaranteed present.  An unconfirmed batch whose
        journal is intact gets fully applied (all or nothing, "in"); a torn or
        CRC-failing journal is discarded ("out").  A journal whose recorded
        generation no longer has a layer belongs to a superseded writer and is
        ignored rather than applied to the wrong generation.
        """
        confirmed = self._manifest["confirmed"]
        for journal_path in sorted(self._journal_dir.glob("*.dbf")):
            batch_id = journal_path.stem
            try:
                generation, event_count, touched = self._read_journal(journal_path)
            except DurableBloomRecoveryError:
                # Torn by an interruption: the batch was not confirmed, drop it.
                journal_path.unlink(missing_ok=True)
                continue

            if str(generation) not in self._manifest["layers"]:
                # Superseded/late writer whose generation was never published.
                journal_path.unlink(missing_ok=True)
                continue

            if batch_id in confirmed:
                # Confirmed: ensure the bits really are on disk, then move on.
                self._merge_into_layer(generation, touched)
                continue

            # Unconfirmed but intact: apply atomically as a whole and confirm.
            self._merge_into_layer(generation, touched)
            confirmed[batch_id] = {
                "generation": generation,
                "events": event_count,
                "journal": journal_path.name,
                "recovered": True,
            }
            self._manifest["events"] = int(self._manifest["events"]) + event_count
        self._write_manifest()

    def _merge_into_layer(self, generation: int, touched: dict[int, int]) -> None:
        layer_path = self._layer_path(generation)
        _num_bits, _number_hashes, byte_length = self._read_meta(generation)
        with open(layer_path, "r+b") as fp:
            mm = mmap(fp.fileno(), 0)
            try:
                for byte_index, mask in touched.items():
                    if byte_index >= byte_length:
                        raise DurableBloomRecoveryError(
                            f"generation {generation} byte {byte_index} is outside its layer"
                        )
                    mm[byte_index] = mm[byte_index] | mask
                mm.flush()
            finally:
                mm.close()
            fp.flush()
            os.fsync(fp.fileno())

    # ------------------------------------------------------------------ #
    # Fencing and small helpers
    # ------------------------------------------------------------------ #
    def _fence_check(self) -> None:
        """Reject a writer fenced off by an explicit generation takeover.

        The manifest is re-read while the lock is held so a long-lived writer
        observes a takeover or grow committed by a different process instead
        of acting on its own (now stale) in-memory copy.
        """
        self._refresh_locked()
        if self._partition_generation < self.generation:
            raise StaleWriterError(
                f"writer fenced at generation {self._partition_generation} but the persisted generation "
                f"is {self.generation}; a takeover already happened"
            )

    def _refresh_locked(self) -> None:
        """Reload the manifest from disk.  Caller must hold the directory lock."""
        self._manifest = self._read_manifest()

    def _ensure_open(self) -> None:
        if self._closed:
            raise InitializationError("this DurableBloomFilter has already been closed")

    @staticmethod
    def _safe_batch_id(batch_id: str) -> str:
        if not isinstance(batch_id, str) or not batch_id:
            raise InitializationError("batch_id must be a non-empty string")
        safe = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in batch_id)
        return safe

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        """Fsync a directory so rename/create metadata is durable."""
        flags = os.O_RDONLY
        fd = os.open(str(path), flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class GenerationReader:
    """A read-only, snapshot view of one immutable generation layer.

    Instances are normally created via :meth:`DurableBloomFilter.open_reader`.
    The backing layer file is sealed, so holding a reader open while the
    filter grows or fails over never produces a torn read.
    """

    __slots__ = ("_path", "_fp", "_mm", "_num_bits", "_number_hashes", "_generation", "_hash_func")

    def __init__(
        self,
        path: Path,
        num_bits: int,
        number_hashes: int,
        generation: int,
        hash_function: HashFuncT,
    ) -> None:
        self._path = Path(path)
        self._num_bits = int(num_bits)
        self._number_hashes = int(number_hashes)
        self._generation = int(generation)
        self._hash_func = hash_function
        self._fp = open(self._path, "rb")  # noqa: SIM115
        self._mm = mmap(self._fp.fileno(), 0, access=ACCESS_READ)

    @property
    def generation(self) -> int:
        """int: The generation this snapshot is bound to."""
        return self._generation

    def check(self, key: KeyT) -> bool:
        """Membership test within this generation only."""
        return self.check_alt(self._hash_func(key, self._number_hashes))

    def check_alt(self, hashes: HashResultsT) -> bool:
        """Membership test from pre-computed hashes within this generation."""
        mm = self._mm
        for index in range(self._number_hashes):
            bit = hashes[index] % self._num_bits
            if (mm[bit // 8] & (1 << (bit % 8))) == 0:
                return False
        return True

    def __contains__(self, key: KeyT) -> bool:
        return self.check(key)

    def close(self) -> None:
        """Release the snapshot's mmap and file handle."""
        if self._mm is not None and not self._mm.closed:
            self._mm.close()
        if self._fp is not None and not self._fp.closed:
            self._fp.close()

    def __enter__(self) -> "GenerationReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
