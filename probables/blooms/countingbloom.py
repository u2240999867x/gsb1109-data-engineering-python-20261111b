"""CountingBloomFilter, python implementation
License: MIT
Author: Tyler Barrus (barrust@gmail.com)
URL: https://github.com/barrust/counting_bloom
"""

from array import array
from binascii import hexlify, unhexlify
from collections.abc import ByteString, Iterable
from io import IOBase
from mmap import mmap
from pathlib import Path
from struct import Struct

from probables.blooms.bloom import BloomFilter
from probables.constants import UINT32_T_MAX, UINT64_T_MAX
from probables.exceptions import InitializationError, SimilarityError
from probables.hashes import HashFuncT, HashResultsT, KeyT
from probables.utilities import is_hex_string, is_valid_file, resolve_path

MISMATCH_MSG = "The parameter second must be of type CountingBloomFilter"


def _verify_not_type_mismatch(second: "CountingBloomFilter") -> bool:
    """verify that there is not a type mismatch"""
    return isinstance(second, (CountingBloomFilter))


class CountingBloomFilter(BloomFilter):
    """Simple Counting Bloom Filter implementation for use in python;
    It can read and write the same format as the c version
    (https://github.com/barrust/counting_bloom)

    A bucket that reaches ``UINT32_T_MAX`` becomes *saturated*: it keeps
    accepting new events but its counter is clamped at ``UINT32_T_MAX`` and the
    stored value is only a lower bound of the true multiplicity. Saturated
    buckets are tracked per bucket and survive export/import so that exact
    counters can always be distinguished from lower-bound counters.

    Args:
        est_elements (int): The number of estimated elements to be added
        false_positive_rate (float): The desired false positive rate
        filepath (str): Path to file to load
        hex_string (str): Hex based representation to be loaded
        hash_function (function): Hashing strategy function to use `hf(key, number)`
    Returns:
        CountingBloomFilter: A Counting Bloom Filter object

    Note:
        Initialization order of operations:
            1) From file
            2) From Hex String
            3) From params"""

    __slots__ = ("_filepath", "_saturated")

    # Versioned on-disk/bytes header. Legacy exports (raw counter array +
    # footer) never start with this magic and have a different total length, so
    # older files remain readable while saturated buckets can be exported.
    _EXPORT_MAGIC = b"CBLM"
    _EXPORT_VERSION = 1
    _EXPORT_HEADER = _EXPORT_MAGIC + bytes((_EXPORT_VERSION,))

    def __init__(
        self,
        est_elements: int | None = None,
        false_positive_rate: float | None = None,
        filepath: str | Path | None = None,
        hex_string: str | None = None,
        hash_function: HashFuncT | None = None,
    ) -> None:
        """setup the basic values needed"""
        self._filepath = None
        self._saturated = bytearray()
        super().__init__(est_elements, false_positive_rate, filepath, hex_string, hash_function)

    def _load_init(self, filepath, hash_function, hex_string, est_elements, false_positive_rate):
        """Handle setting params and loading everything as needed"""
        self._bits_per_elm = 1.0
        self._type = "counting"
        self._typecode = "I"

        if is_valid_file(filepath):
            self._filepath = resolve_path(filepath)
            self._load(self._filepath, hash_function)
        elif is_hex_string(hex_string):
            self._load_hex(hex_string, hash_function)
        else:
            if est_elements is None or false_positive_rate is None:
                raise InitializationError("Insufecient parameters to set up the Counting Bloom Filter")
            # calc values
            fpr, n_hashes, n_bits = self._get_optimized_params(est_elements, false_positive_rate)
            self._set_values(est_elements, fpr, n_hashes, n_bits, hash_function)
            self._bloom_length = n_bits
            self._bloom = array(self._typecode, [0]) * self._bloom_length
            self._saturated = bytearray(self._bloom_length)

    _IMPT_STRUCT = Struct("I")

    @classmethod
    def frombytes(cls, b: ByteString, hash_function: HashFuncT | None = None) -> "CountingBloomFilter":
        """
        Args:
            b (ByteString): the bytes to load as a Counting Bloom Filter
            hash_function (function): Hashing strategy function to use `hf(key, number)`
        Returns:
            CountingBloomFilter: A Counting Bloom Filter object
        """
        header = len(cls._EXPORT_HEADER)
        if bytes(b[:header]) == cls._EXPORT_HEADER:
            footer_offset = cls._FOOTER_STRUCT.size
            _, _, _, _, n_bits = cls._parse_footer(cls._FOOTER_STRUCT, bytes(b[-1 * footer_offset :]))
            versioned_size = header + n_bits + cls._IMPT_STRUCT.size * n_bits + cls._FOOTER_STRUCT.size
            if len(b) == versioned_size:
                return cls._frombytes_versioned(b, hash_function)
        offset = cls._FOOTER_STRUCT.size
        est_els, els_added, fpr, n_hashes, n_bits = cls._parse_footer(cls._FOOTER_STRUCT, bytes(b[-1 * offset :]))
        blm = CountingBloomFilter(est_elements=est_els, false_positive_rate=fpr, hash_function=hash_function)
        blm._set_values(est_els, fpr, n_hashes, n_bits, hash_function)
        blm._els_added = els_added
        blm._parse_bloom_array(b, cls._IMPT_STRUCT.size * blm.bloom_length)
        blm._saturated = bytearray(blm.bloom_length)
        return blm

    @classmethod
    def _frombytes_versioned(cls, b: ByteString, hash_function: HashFuncT | None) -> "CountingBloomFilter":
        """parse the versioned export format produced by :meth:`export`"""
        header = len(cls._EXPORT_HEADER)
        offset = cls._FOOTER_STRUCT.size
        est_els, els_added, fpr, n_hashes, n_bits = cls._parse_footer(cls._FOOTER_STRUCT, bytes(b[-1 * offset :]))
        blm = CountingBloomFilter(est_elements=est_els, false_positive_rate=fpr, hash_function=hash_function)
        blm._set_values(est_els, fpr, n_hashes, n_bits, hash_function)
        length = blm.bloom_length
        blm._saturated = bytearray(bytes(b[header : header + length]))
        bloom_start = header + length
        blm._bloom = array(blm._typecode, bytes(b[bloom_start : bloom_start + cls._IMPT_STRUCT.size * length]))
        blm._els_added = els_added
        return blm

    def __str__(self) -> str:
        """string representation of the counting bloom filter"""
        on_disk = "no" if self.is_on_disk is False else "yes"

        cnt = sum(x for x in self._bloom if x > 0)
        total = sum(self._bloom)
        largest = max(self._bloom)
        largest_idx = (self._bloom).index(largest)
        fullness = cnt / self.number_bits
        els_added = total // self.number_hashes

        return (
            "CountingBloom:\n"
            f"\tbits: {self.number_bits}\n"
            f"\testimated elements: {self.estimated_elements}\n"
            f"\tnumber hashes: {self.number_hashes}\n"
            f"\tmax false positive rate: {self.false_positive_rate:.6f}\n"
            f"\telements added: {self.elements_added}\n"
            f"\tcurrent false positive rate: {self.current_false_positive_rate():.6f}\n"
            f"\tis on disk: {on_disk}\n"
            f"\tindex fullness: {fullness:.6}\n"
            f"\tmax index usage: {largest}\n"
            f"\tmax index id: {largest_idx}\n"
            f"\tcalculated elements: {els_added}\n"
        )

    def _indices_for(self, hashes: HashResultsT) -> list[int]:
        """return the distinct bucket indices a key touches; duplicate hash
        positions are collapsed so each bucket is updated at most once per
        insertion/removal"""
        indices: list[int] = []
        for i in range(self._number_hashes):
            idx = hashes[i] % self._bloom_length
            if idx not in indices:
                indices.append(idx)
        return indices

    def add(self, key: KeyT, num_els: int = 1) -> int:  # type: ignore
        """Add the key to the Counting Bloom Filter

        Args:
            key (str): The element to be inserted
            num_els (int): Number of times to insert the element
        Returns:
            int: Minimum counter value across the key's buckets after insertion"""
        return self.add_alt(self.hashes(key), num_els)

    def add_alt(self, hashes: HashResultsT, num_els: int = 1) -> int:  # type: ignore
        """Add the element represented by hashes into the Counting Bloom Filter

        Saturated buckets keep accepting events: their counter stays clamped at
        ``UINT32_T_MAX`` and their saturated flag remains set, signalling that
        the stored value is only a lower bound.

        Args:
            hashes (list): A list of integers representing the key to insert
            num_els (int): Number of times to insert the element
        Returns:
            int: Minimum counter value across the key's buckets after insertion"""
        for idx in self._indices_for(hashes):
            current = self._bloom[idx]
            if current >= UINT32_T_MAX:
                self._saturated[idx] = 1
                continue
            if num_els >= UINT32_T_MAX - current:
                self._bloom[idx] = UINT32_T_MAX
                self._saturated[idx] = 1
            else:
                self._bloom[idx] = current + num_els
        self.elements_added = min(self.elements_added + num_els, UINT64_T_MAX)
        return self.check_alt(hashes)

    def check(self, key: KeyT) -> int:  # type: ignore
        """Check if the key is likely in the Counting Bloom Filter

        Args:
            key (str): The element to be checked
        Returns:
            int: Minimum counter value across the key's buckets; use
            :meth:`check_with_status` to know whether it is exact or a lower bound"""
        return self.check_alt(self.hashes(key))

    def check_alt(self, hashes: HashResultsT) -> int:  # type: ignore
        """Check if the element represented by hashes is in the Counting
        Bloom Filter

        Args:
            hashes (list): A list of integers representing the key to check
        Returns:
            int: Minimum counter value across the key's buckets"""
        return min(self._bloom[idx] for idx in self._indices_for(hashes))

    def check_with_status(self, key: KeyT) -> tuple[int, bool]:
        """Check the key and report whether the count is exact

        Args:
            key (str): The element to be checked
        Returns:
            tuple(int, bool): ``(count, is_exact)``; when ``is_exact`` is
            ``False`` the count is only guaranteed to be a lower bound because
            one of the key's buckets saturated"""
        return self.check_with_status_alt(self.hashes(key))

    def check_with_status_alt(self, hashes: HashResultsT) -> tuple[int, bool]:
        """Check the element represented by hashes and report whether the
        count is exact

        Args:
            hashes (list): A list of integers representing the key to check
        Returns:
            tuple(int, bool): ``(count, is_exact)``"""
        indices = self._indices_for(hashes)
        count = min(self._bloom[idx] for idx in indices)
        is_exact = not any(self._saturated[idx] for idx in indices)
        return count, is_exact

    @property
    def has_saturated_buckets(self) -> bool:
        """bool: True if at least one bucket has saturated and only reports a
        lower bound"""
        return 1 in self._saturated

    @property
    def saturated_buckets(self) -> tuple[int, ...]:
        """tuple(int): Indices of the buckets that have saturated and therefore
        only provide lower-bound counters"""
        return tuple(i for i, flag in enumerate(self._saturated) if flag)

    def is_saturated(self, key: KeyT) -> bool:
        """Return whether any bucket backing the key has saturated, meaning the
        count returned by :meth:`check` is only a lower bound

        Args:
            key (str): The element to be checked
        Returns:
            bool: True if any of the key's buckets is saturated"""
        return self.is_saturated_alt(self.hashes(key))

    def is_saturated_alt(self, hashes: HashResultsT) -> bool:
        """Return whether any bucket backing the hashes has saturated

        Args:
            hashes (list): A list of integers representing the key to check
        Returns:
            bool: True if any of the key's buckets is saturated"""
        return any(self._saturated[idx] for idx in self._indices_for(hashes))

    def remove(self, key: KeyT, num_els: int = 1) -> int:
        """Remove the element from the counting bloom

        Saturated buckets are never decremented: their true multiplicity is
        unknown, so removing from them could make another still-present key
        sharing the bucket disappear.

        Args:
            key (str): The element to be removed
            num_els (int): Number of times to remove the element
        Returns:
            int: Minimum counter value across the key's buckets after removal"""
        return self.remove_alt(self.hashes(key), num_els)

    def remove_alt(self, hashes: HashResultsT, num_els: int = 1) -> int:
        """Remove the element represented by hashes from the Counting Bloom Filter

        Args:
            hashes (list): A list of integers representing the key to remove
            num_els (int): Number of times to remove the element
        Returns:
            int: Minimum counter value across the key's buckets after removal"""
        indices = self._indices_for(hashes)
        vals = [self._bloom[idx] for idx in indices]
        min_val = min(vals)
        if min_val == 0:
            return 0

        if any(self._saturated[idx] for idx in indices):
            # Only exact buckets may be decremented; saturated buckets hold an
            # unknown true count and must keep their lower bound intact.
            editable = [self._bloom[idx] for idx in indices if not self._saturated[idx]]
            to_remove = min([num_els, *editable]) if editable else 0
            for idx in indices:
                if not self._saturated[idx]:
                    self._bloom[idx] -= to_remove
            self.elements_added -= to_remove
            return self.check_alt(hashes)

        to_remove = num_els if min_val > num_els else min_val
        for idx in indices:
            self._bloom[idx] -= to_remove
        self.elements_added -= to_remove
        return min_val - to_remove

    def reconcile(self, keys: Iterable[KeyT], num_els: int = 1) -> int:
        """Reconcile a sealed input segment against its complete key stream

        The caller supplies the full key stream belonging to a previously sealed
        segment. Only buckets touched by that stream are recomputed and only the
        segment's own multiplicity is subtracted, so contributions from later
        batches sharing a bucket are left untouched. Duplicate hash positions of
        a key count once and shared buckets accumulate the real update
        multiplicity, which lets a small collision sample be reconciled back to
        the ledger incrementally.

        Args:
            keys (iterable): The complete key stream of the sealed segment
            num_els (int): Multiplicity each key in the stream was inserted with
        Returns:
            int: Number of buckets that were recomputed
        Raises:
            InitializationError: When ``num_els`` is smaller than one"""
        return self.reconcile_alt((self.hashes(key) for key in keys), num_els)

    def reconcile_alt(self, hash_lists: Iterable[HashResultsT], num_els: int = 1) -> int:
        """Reconcile a sealed segment using pre-computed hashes; see
        :meth:`reconcile` for the semantics

        Args:
            hash_lists (iterable): One hash result per key in the sealed segment
            num_els (int): Multiplicity each key was inserted with
        Returns:
            int: Number of buckets that were recomputed
        Raises:
            InitializationError: When ``num_els`` is smaller than one"""
        if num_els < 1:
            raise InitializationError("Counting Bloom Filter: num_els must be greater than 0")

        deltas: dict[int, int] = {}
        n_keys = 0
        for hashes in hash_lists:
            n_keys += 1
            for idx in self._indices_for(hashes):
                deltas[idx] = deltas.get(idx, 0) + num_els

        for idx, delta in deltas.items():
            current = self._bloom[idx]
            self._bloom[idx] = current - min(current, delta)
            # A saturated bucket was clamped, so its residual is only guaranteed
            # to be a lower bound of later batches' contribution; the flag is
            # therefore retained rather than reporting false precision.

        self.elements_added = max(0, self.elements_added - n_keys * num_els)
        return len(deltas)

    def clear(self) -> None:
        """Clear or reset the Counting Bloom Filter, including saturation flags"""
        super().clear()
        for idx in range(self._bloom_length):
            self._saturated[idx] = 0

    def export_hex(self) -> str:
        """Export the Counting Bloom Filter as a hex string

        The exported payload carries the per-bucket saturation flags so that an
        importer can still distinguish exact counters from lower-bound ones.

        Return:
            str: Hex representation of the Counting Bloom Filter"""
        footer_bytes = self._FOOTER_STRUCT_BE.pack(
            self.estimated_elements,
            self.elements_added,
            self.false_positive_rate,
        )
        payload = self._EXPORT_HEADER + bytes(self._saturated) + bytearray(self._bloom[: self.bloom_length])
        payload += footer_bytes
        return str(hexlify(payload), "utf-8")

    def export(self, file: Path | str | IOBase | mmap) -> None:
        """Export the Counting Bloom Filter to disk

        The layout is a versioned header followed by the saturation flags, the
        counter array and the standard footer. Files written by older versions
        (without the header) can still be loaded.

        Args:
            file (str): The file or filepath to which the filter will be written."""
        if not isinstance(file, IOBase | mmap):
            path = resolve_path(file)
            with open(path, "wb") as filepointer:
                self._export_stream(filepointer)
        else:
            self._export_stream(file)

    def _export_stream(self, filepointer) -> None:
        """write the versioned format to an open binary stream"""
        filepointer.write(self._EXPORT_HEADER)
        filepointer.write(bytes(self._saturated))
        self._bloom.tofile(filepointer)
        filepointer.write(
            self._FOOTER_STRUCT.pack(
                self.estimated_elements,
                self.elements_added,
                self.false_positive_rate,
            )
        )

    def export_size(self) -> int:
        """Calculate the size of the bloom on disk

        Returns:
            int: Size of the Counting Bloom Filter when exported to disk"""
        return (
            len(self._EXPORT_HEADER)
            + self.bloom_length
            + self.bloom_length * self._IMPT_STRUCT.size
            + self._FOOTER_STRUCT.size
        )

    def _versioned_length(self, n_bits: int) -> int:
        """expected byte length of a versioned export for a filter of this size"""
        return len(self._EXPORT_HEADER) + n_bits + self._IMPT_STRUCT.size * n_bits + self._FOOTER_STRUCT.size

    def _load_hex(self, hex_string: str, hash_function: HashFuncT | None = None) -> None:
        """load from a hex string, supporting both the versioned and legacy format"""
        footer_size = self._FOOTER_STRUCT_BE.size
        est_els, els_added, fpr, n_hashes, n_bits = self._parse_footer(
            self._FOOTER_STRUCT_BE, unhexlify(hex_string[-footer_size * 2 :])
        )
        prefix = str(hexlify(self._EXPORT_HEADER), "utf-8")
        is_versioned = hex_string.startswith(prefix) and len(hex_string) == 2 * self._versioned_length(n_bits)
        if is_versioned:
            raw = unhexlify(hex_string)
            header = len(self._EXPORT_HEADER)
            self._set_values(est_els, fpr, n_hashes, n_bits, hash_function)
            length = self._bloom_length
            self._saturated = bytearray(raw[header : header + length])
            bloom_start = header + length
            self._bloom = array(self._typecode, raw[bloom_start : bloom_start + self._IMPT_STRUCT.size * length])
            self._els_added = els_added
        else:
            super()._load_hex(hex_string, hash_function)
            self._saturated = bytearray(self._bloom_length)

    def _load(
        self,
        file: Path | str | IOBase | mmap | ByteString,
        hash_function: HashFuncT | None = None,
    ) -> None:
        """load the Counting Bloom Filter from file or bytes, dispatching on
        the versioned header"""
        if isinstance(file, str | Path):
            with open(resolve_path(file), "rb") as filepointer:
                data = filepointer.read()
        elif isinstance(file, IOBase):
            data = file.read()
        else:
            data = bytes(file)  # bytes, bytearray, memoryview or mmap

        footer_size = self._FOOTER_STRUCT.size
        est_els, els_added, fpr, n_hashes, n_bits = self._parse_footer(
            self._FOOTER_STRUCT, data[-footer_size:]
        )
        is_versioned = (
            data[: len(self._EXPORT_HEADER)] == self._EXPORT_HEADER and len(data) == self._versioned_length(n_bits)
        )
        if is_versioned:
            self._set_values(est_els, fpr, n_hashes, n_bits, hash_function)
            header = len(self._EXPORT_HEADER)
            length = self._bloom_length
            self._saturated = bytearray(data[header : header + length])
            bloom_start = header + length
            self._bloom = array(self._typecode, data[bloom_start : bloom_start + self._IMPT_STRUCT.size * length])
        else:
            self._set_values(est_els, fpr, n_hashes, n_bits, hash_function)
            self._parse_bloom_array(data, self._IMPT_STRUCT.size * self.bloom_length)
            self._saturated = bytearray(self._bloom_length)
        self._els_added = els_added

    def intersection(self, second: "CountingBloomFilter") -> "CountingBloomFilter":  # type: ignore
        """Take the intersection of two Counting Bloom Filters

        Args:
            second (CountingBloomFilter): The Bloom Filter with which to take the intersection
        Returns:
            CountingBloomFilter: The new Counting Bloom Filter containing the union
        Raises:
            TypeError: When second is not a :class:`CountingBloomFilter`
            SimilarityError: When second is not of the same size (false_positive_rate and est_elements)
        Note:
            The elements_added property will be set to the estimated number of unique elements \
                added as found in estimate_elements()"""
        if not _verify_not_type_mismatch(second):
            raise TypeError(MISMATCH_MSG)

        if self._verify_bloom_similarity(second) is False:
            raise SimilarityError("Counting Bloom Filters are not similar enough to calculate similarity")

        res = CountingBloomFilter(
            est_elements=self.estimated_elements,
            false_positive_rate=self.false_positive_rate,
            hash_function=self.hash_function,
        )

        for i in range(self.bloom_length):
            if self._bloom[i] > 0 and second._bloom[i] > 0:
                tmp = self._bloom[i] + second._bloom[i]
                res.bloom[i] = min(tmp, UINT32_T_MAX)
                # a saturated operand or an arithmetic overflow makes the sum
                # a lower bound
                res._saturated[i] = self._saturated[i] | second._saturated[i] | int(tmp > UINT32_T_MAX)
        res.elements_added = res.estimate_elements()
        return res

    def jaccard_index(self, second: "CountingBloomFilter") -> float:  # type: ignore
        """Take the Jaccard Index of two Counting Bloom Filters

        Args:
            second (CountingBloomFilter): The Bloom Filter with which to take the jaccard index
        Returns:
            float: A numeric value between 0 and 1 where 1 is identical and 0 means completely different
        Raises:
            TypeError: When second is not a :class:`CountingBloomFilter`
            SimilarityError: When second is not of the same size (false_positive_rate and est_elements)
        Note:
            The Jaccard Index is based on the unique set of elements added and not the number of each element added"""
        if not _verify_not_type_mismatch(second):
            raise TypeError(MISMATCH_MSG)

        if self._verify_bloom_similarity(second) is False:
            raise SimilarityError("Counting Bloom Filters are not similar enough to calculate similarity")

        count_union = 0
        count_inter = 0
        for i in range(self.bloom_length):
            if self._bloom[i] > 0 or second._bloom[i] > 0:
                count_union += 1
            if self._bloom[i] > 0 and second._bloom[i] > 0:
                count_inter += 1
        if count_union == 0:
            return 1.0
        return count_inter / count_union

    def union(self, second: "CountingBloomFilter") -> "CountingBloomFilter":  # type:ignore
        """Return a new Countiong Bloom Filter that contains the union of
        the two

        Args:
            second (CountingBloomFilter): The Counting Bloom Filter with which to calculate the union
        Returns:
            CountingBloomFilter: The new Counting Bloom Filter containing the union
        Raises:
            TypeError: When second is not a :class:`CountingBloomFilter`
            SimilarityError: When second is not of the same size (false_positive_rate and est_elements)
        Note:
            The elements_added property will be set to the estimated number of unique elements added as \
                found in estimate_elements()"""
        if not _verify_not_type_mismatch(second):
            raise TypeError(MISMATCH_MSG)

        if self._verify_bloom_similarity(second) is False:
            raise SimilarityError("Counting Bloom Filters are not similar enough to calculate similarity")

        res = CountingBloomFilter(
            est_elements=self.estimated_elements,
            false_positive_rate=self.false_positive_rate,
            hash_function=self.hash_function,
        )
        for i in range(self.bloom_length):
            tmp = self._bloom[i] + second._bloom[i]
            res._bloom[i] = min(tmp, UINT32_T_MAX)
            res._saturated[i] = self._saturated[i] | second._saturated[i] | int(tmp > UINT32_T_MAX)
        res.elements_added = res.estimate_elements()
        return res

    def _cnt_number_bits_set(self) -> int:
        """calculate the total number of set bits in the bloom"""
        return sum(1 for x in self._bloom if x > 0)
