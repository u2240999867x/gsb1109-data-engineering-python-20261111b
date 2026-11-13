"""Expanding and Rotating BloomFilter, python implementations
License: MIT
Author: Tyler Barrus (barrust@gmail.com)
URL: https://github.com/barrust/pyprobables
"""

import math
from array import array
from collections.abc import ByteString
from io import BytesIO, IOBase
from mmap import mmap
from numbers import Number
from pathlib import Path
from struct import Struct
from typing import NamedTuple

from probables.blooms.bloom import BloomFilter
from probables.exceptions import InitializationError, RotatingBloomFilterError
from probables.hashes import HashFuncT, HashResultsT, KeyT, default_fnv_1a
from probables.utilities import MMap, is_valid_file, resolve_path


class _ExpandingBloomMetadata(NamedTuple):
    """Metadata stored in an expanding Bloom Filter footer"""

    size: int
    estimated_elements: int
    elements_added: int
    overall_false_positive_rate: float | None
    layer_false_positive_rates: tuple[float, ...]
    data_length: int
    budget_mode: int
    next_layer_index: int


class ExpandingBloomFilter:
    """Simple expanding Bloom Filter implementation for use in python; the
    Bloom Filter will automatically expand, or grow, if the false
    positive rate is about to become greater than the desired false
    positive rate.

    By default ``false_positive_rate`` is the target rate of an individual
    layer. Once more than one layer exists the rate for the whole filter is
    higher because a query succeeds when *any* layer reports a match. A per
    layer rate therefore must not be treated as an overall guarantee once the
    filter has expanded.

    Set ``overall_false_positive_rate`` (or set ``budget_mode=True`` and pass
    the budget as ``false_positive_rate``) to enable explicit overall budget
    mode. Layer ``i`` is targeted at ``P / 2 ** (i + 1)`` where ``P`` is the
    overall budget (less a small margin so the stored rates always fit inside
    ``P``). The series sums to at most ``P``, so the union bound for any number
    of layers never exceeds ``P``; for a finite number of layers it is
    strictly smaller. A new layer is opened as soon as the newest layer has
    received ``est_elements`` keys. Queries always cover every active layer.

    Args:
        est_elements (int): The number of estimated elements to be added to each layer
        false_positive_rate (float): The desired false positive rate. When budget \
            mode is enabled this value is used as the overall budget.
        filepath (str): Path to file to load
        hash_function (function): Hashing strategy function to use `hf(key, number)`
        overall_false_positive_rate (float): Explicit overall false positive budget; \
            enables budget mode
        overall_fpr (float): Alias for ``overall_false_positive_rate``
        overall_budget (float): Alias for ``overall_false_positive_rate``
        budget (float): Alias for ``overall_false_positive_rate``
        budget_mode (bool): Treat ``false_positive_rate`` as the overall budget when \
            an explicit numeric budget is not supplied
        use_overall_budget (bool): Alias for ``budget_mode``
        overall (bool): Alias for ``budget_mode``
        mode (str): ``"overall"`` or ``"budget"`` to enable budget mode; \
            ``"per-layer"`` or ``"layer"`` to keep per layer semantics
    Returns:
        ExpandingBloomFilter: An expanding Bloom Filter object
    Note:
        Initialization order of operations:
            1) Filepath
            2) est_elements, false_positive_rate / overall_false_positive_rate

        Layer ``i`` in budget mode targets ``P / 2 ** (i + 1)`` where ``i``
        starts at zero (a small allocation margin keeps the float32 rates
        stored by the layers inside ``P``). Later layers are therefore larger
        and need more hash permutations. Hashes are always generated at the
        largest depth required by the layers involved in the operation. A
        caller supplying pre-computed hashes that are too short receives a
        ``ValueError`` instead of silently querying a layer with missing
        values.

        Exported files preserve the budget mode, the overall budget, the target
        rate of every active layer, and the next layer index used by the
        allocation schedule. When a file is loaded, those values take
        precedence over constructor arguments."""

    __slots__ = (
        "_blooms",
        "_fpr",
        "_est_elements",
        "_hash_func",
        "_added_elements",
        "_budget_mode",
        "_overall_fpr",
        "_layer_fprs",
        "_next_layer_index",
    )

    _BUDGET_NONE = 0
    _BUDGET_GEOMETRIC = 1
    _BUDGET_NAMES = {_BUDGET_GEOMETRIC: "geometric"}
    _BUDGET_ENABLE_NAMES = ("overall", "budget", "geometric", "overall-budget", "overall_budget")
    _BUDGET_DISABLE_NAMES = ("per-layer", "per_layer", "layer", "none", "legacy")
    _BUDGET_ALLOCATION_MARGIN = 2**-20

    def __init__(
        self,
        est_elements: int | None = None,
        false_positive_rate: float | None = None,
        filepath: str | Path | None = None,
        hash_function: HashFuncT | None = None,
        overall_false_positive_rate: float | None = None,
        *,
        overall_fpr: float | None = None,
        overall_budget: float | bool | None = None,
        budget: float | bool | None = None,
        budget_mode: bool | str = False,
        use_overall_budget: bool | str = False,
        overall: bool = False,
        mode: str | None = None,
        fpr_mode: str | None = None,
    ):
        """initialize"""
        self._blooms: list[BloomFilter] = []
        self._budget_mode = self._BUDGET_NONE
        self._overall_fpr: float | None = None
        self._fpr = false_positive_rate if false_positive_rate is not None else 0.0
        self._est_elements = est_elements if est_elements is not None else 100
        self._hash_func: HashFuncT = default_fnv_1a if hash_function is None else hash_function
        self._added_elements = 0
        self._layer_fprs: list[float] = []
        self._next_layer_index = 0

        explicit_overall = self._coalesce_overall_budget(
            overall_false_positive_rate, overall_fpr, overall_budget, budget
        )
        flag_mode = self._budget_flag_enabled(
            budget_mode, use_overall_budget, overall, budget, overall_budget, mode, fpr_mode
        )
        if explicit_overall is None and flag_mode:
            if false_positive_rate is None:
                raise InitializationError(
                    "ExpandingBloomFilter: overall budget mode requires false_positive_rate or an overall budget"
                )
            explicit_overall = false_positive_rate
            false_positive_rate = None

        if explicit_overall is not None:
            self._overall_fpr = self._validate_fpr(explicit_overall, "overall false positive rate")
            self._budget_mode = self._BUDGET_GEOMETRIC
            self._fpr = self._budget_layer_fpr(0)
        elif false_positive_rate is not None:
            self._fpr = false_positive_rate

        if filepath is not None and is_valid_file(filepath):
            self._load(filepath)
        else:
            # add in the initial bloom filter!
            self._add_bloom_filter()

    __FOOTER_STRUCT = Struct("QQQf")
    __S_INT64_STRUCT = Struct("Q")
    __BUDGET_FPR_STRUCT = Struct("<d")
    _BLOOM_ELEMENT_SIZE = Struct("B").size

    __BUDGET_MAGIC = b"EBF2"
    __BUDGET_FOOTER_STRUCT = Struct("<QQQQdBB4s")

    @staticmethod
    def _validate_fpr(value: float, name: str) -> float:
        valid = isinstance(value, Number) and not isinstance(value, bool) and 0.0 < float(value) < 1.0
        if not valid:
            raise InitializationError(f"ExpandingBloomFilter: {name} must be between 0.0 and 1.0")
        return float(value)

    @staticmethod
    def _coalesce_overall_budget(*values: float | bool | None) -> float | None:
        found: list[float] = []
        for value in values:
            if value is None or isinstance(value, bool):
                continue
            found.append(float(value))
        if not found:
            return None
        if any(item != found[0] for item in found[1:]):
            raise InitializationError("ExpandingBloomFilter: conflicting overall false positive budgets provided")
        return found[0]

    @classmethod
    def _budget_flag_enabled(cls, *values: bool | str | float | None) -> bool:
        enabled = False
        for value in values:
            if value is None or isinstance(value, bool):
                enabled = enabled or bool(value)
                continue
            if isinstance(value, Number):
                continue
            if not isinstance(value, str):
                raise InitializationError(f"ExpandingBloomFilter: unknown false positive rate mode: {value}")
            lowered = value.lower().replace(" ", "-")
            if lowered in cls._BUDGET_ENABLE_NAMES:
                enabled = True
            elif lowered not in cls._BUDGET_DISABLE_NAMES:
                raise InitializationError(f"ExpandingBloomFilter: unknown false positive rate mode: {value}")
        return enabled

    @property
    def budget_mode(self) -> bool:
        """bool: Is the filter using an explicit overall false positive budget"""
        return self._budget_mode != self._BUDGET_NONE

    @property
    def uses_overall_budget(self) -> bool:
        """bool: Alias for :attr:`budget_mode`"""
        return self.budget_mode

    @property
    def overall_mode(self) -> bool:
        """bool: Alias for :attr:`budget_mode`"""
        return self.budget_mode

    @property
    def budget_strategy(self) -> str | None:
        """str: The layer budget allocation strategy, or None when not in budget mode"""
        return self._BUDGET_NAMES.get(self._budget_mode)

    @property
    def allocation_strategy(self) -> str | None:
        """str: Alias for :attr:`budget_strategy`"""
        return self.budget_strategy

    @property
    def overall_false_positive_rate(self) -> float | None:
        """float: The overall false positive budget, or None when not in budget mode"""
        return self._overall_fpr

    @property
    def overall_fpr(self) -> float | None:
        """float: Alias for :attr:`overall_false_positive_rate`"""
        return self._overall_fpr

    @property
    def overall_budget(self) -> float | None:
        """float: Alias for :attr:`overall_false_positive_rate`"""
        return self._overall_fpr

    @property
    def budget(self) -> float | None:
        """float: Alias for :attr:`overall_false_positive_rate`"""
        return self._overall_fpr

    @property
    def false_positive_rate(self) -> float:
        """float: The desired false positive rate of the expanding Bloom Filter

        In explicit budget mode this is the overall budget. Otherwise it is the
        target rate of an individual layer."""
        if self._overall_fpr is not None:
            return self._overall_fpr
        return self._fpr

    @property
    def false_positive_rate_bound(self) -> float:
        """float: Mathematical union bound for the current layers

        The probability that any active layer reports a false match is at most
        the sum of the layer target rates stored in the layers. This is a
        bound, not a measurement."""
        return min(1.0, sum(blm.false_positive_rate for blm in self._blooms))

    @property
    def guaranteed_false_positive_rate(self) -> float:
        """float: Alias for :attr:`false_positive_rate_bound`"""
        return self.false_positive_rate_bound

    @property
    def theoretical_false_positive_rate(self) -> float:
        """float: Alias for :attr:`false_positive_rate_bound`"""
        return self.false_positive_rate_bound

    @property
    def allocated_false_positive_rate(self) -> float:
        """float: Total target rate currently allocated to active layers"""
        return sum(blm.false_positive_rate for blm in self._blooms)

    @property
    def layer_false_positive_rates(self) -> tuple[float, ...]:
        """tuple(float): The target false positive rate of each active layer"""
        return tuple(blm.false_positive_rate for blm in self._blooms)

    @property
    def layer_fprs(self) -> tuple[float, ...]:
        """tuple(float): Alias for :attr:`layer_false_positive_rates`"""
        return self.layer_false_positive_rates

    @property
    def per_layer_false_positive_rates(self) -> tuple[float, ...]:
        """tuple(float): Alias for :attr:`layer_false_positive_rates`"""
        return self.layer_false_positive_rates

    def layer_false_positive_rate(self, index: int) -> float:
        """Return the target false positive rate assigned to an active layer

        Args:
            index (int): Zero based layer index
        Returns:
            float: The target rate for the layer"""
        return self._blooms[index].false_positive_rate

    def layer_fpr(self, index: int) -> float:
        """Return the target false positive rate assigned to an active layer
        Args:
            index (int): Zero based layer index
        Returns:
            float: The target rate for the layer"""
        return self._blooms[index].false_positive_rate

    @property
    def next_layer_index(self) -> int:
        """int: The index used to allocate the next layer's budget"""
        return self._next_layer_index

    @property
    def next_layer_false_positive_rate(self) -> float:
        """float: The target rate the next layer would receive"""
        return self._next_layer_target_fpr()

    @property
    def next_layer_fpr(self) -> float:
        """float: Alias for :attr:`next_layer_false_positive_rate`"""
        return self._next_layer_target_fpr()

    def layer_metrics(self) -> list[dict[str, int | float]]:
        """Return per-layer allocation and usage information

        Returns:
            list(dict): One entry per active layer with its target rate, \
                number of bits, number of hashes, elements added, and capacity"""
        result: list[dict[str, int | float]] = []
        for idx, blm in enumerate(self._blooms):
            result.append(
                {
                    "index": idx,
                    "false_positive_rate": blm.false_positive_rate,
                    "number_bits": blm.number_bits,
                    "number_hashes": blm.number_hashes,
                    "elements_added": blm.elements_added,
                    "estimated_elements": blm.estimated_elements,
                }
            )
        return result

    @property
    def expansions(self) -> int:
        """int: The number of expansions"""
        return len(self._blooms) - 1

    @property
    def estimated_elements(self) -> int:
        """int: The original number of elements estimated to be in each layer of the Bloom Filter"""
        return self._est_elements

    @property
    def elements_added(self) -> int:
        """int: The total number of elements added"""
        return self._added_elements

    @property
    def hash_function(self) -> HashFuncT:
        """function: The hash function used"""
        return self._hash_func

    @property
    def hash_depth(self) -> int:
        """int: The largest number of hash permutations required by an active layer"""
        return self._max_hash_depth()

    @property
    def number_hashes(self) -> int:
        """int: Alias for :attr:`hash_depth`"""
        return self._max_hash_depth()

    @classmethod
    def frombytes(cls, b: ByteString, hash_function: HashFuncT | None = None) -> "ExpandingBloomFilter":
        """
        Args:
            b (ByteString): The bytes to load as a Expanding Bloom Filter
            hash_function (function): Hashing strategy function to use `hf(key, number)`
        Returns:
            ExpandingBloomFilter: A Bloom Filter object
        """
        metadata = cls._parse_metadata(b)
        if metadata.budget_mode:
            blm = ExpandingBloomFilter(
                est_elements=metadata.estimated_elements,
                overall_false_positive_rate=metadata.overall_false_positive_rate,
                hash_function=hash_function,
            )
        else:
            blm = ExpandingBloomFilter(
                est_elements=metadata.estimated_elements,
                false_positive_rate=metadata.layer_false_positive_rates[0],
                hash_function=hash_function,
            )
        blm._parse_blooms(b, metadata)
        blm._added_elements = metadata.elements_added
        return blm

    def __contains__(self, key: KeyT) -> bool:
        """setup the `in` functionality"""
        return self.check(key)

    def __bytes__(self) -> bytes:
        """Export bloom filter to `bytes`"""

        with BytesIO() as f:
            self.export(f)
            return f.getvalue()

    def _budget_layer_fpr(self, index: int) -> float:
        """Target false positive rate assigned to layer index"""
        if self._overall_fpr is None:
            return self._fpr
        rate = math.ldexp(self._overall_fpr, -(index + 1)) * (1.0 - self._BUDGET_ALLOCATION_MARGIN)
        if rate < 1e-45:
            raise InitializationError(
                "ExpandingBloomFilter: overall budget is exhausted; the next layer's rate is not representable"
            )
        return rate

    def push(self) -> None:
        """Push a new expansion onto the Bloom Filter"""
        self._add_bloom_filter()

    def hashes(self, key: KeyT, depth: int | None = None) -> HashResultsT:
        """Return the hashes needed to query or extend the filter

        Args:
            key (str|bytes): The key to hash
            depth (int): Number of hash permutations; defaults to the largest \
                number required by active layers or by the next layer's budget
        Returns:
            list(int): A list of hashes for the key"""
        tmp = depth if depth is not None else max(self._max_hash_depth(), self._next_layer_hash_depth())
        return self._hash_func(key, tmp)

    def check(self, key: KeyT) -> bool:
        """Check to see if the key is in the Bloom Filter

        Every active layer is queried. Hashes are generated at the largest
        depth required by those layers so that a layer with a tighter false
        positive budget never receives a short hash list.

        Args:
            key (str): The key to check for in the Bloom Filter
        Returns:
            bool: `True` if the element is likely present; `False` if definately not present"""
        return self.check_alt(self._hash_func(key, self._max_hash_depth()))

    def check_alt(self, hashes: HashResultsT) -> bool:
        """Check to see if the hashes are in the Bloom Filter

        Args:
            hashes (list): The hash representation to check for in the Bloom Filter
        Returns:
            bool: `True` if the element is likely present; `False` if definately not present
        Raises:
            ValueError: If the hash list is shorter than required by any active layer"""
        self._validate_hashes(hashes, self._max_hash_depth())
        return any(blm.check_alt(hashes) for blm in self._blooms)

    def add(self, key: KeyT, force: bool = False) -> None:
        """Add the key to the Bloom Filter

        Args:
            key (str): The element to be inserted
            force (bool): `True` will force it to be inserted, even if it likely has been inserted \
                before `False` will only insert if not found in the Bloom Filter"""
        self.add_alt(self._hash_func(key, self._required_hash_depth_for_insert(force)), force)

    def add_alt(self, hashes: HashResultsT, force: bool = False) -> None:
        """Add the element represented by hashes into the Bloom Filter

        Args:
            hashes (list): A list of integers representing the key to insert
            force (bool): `True` will force it to be inserted, even if it likely has been inserted \
                before `False` will only insert if not found in the Bloom Filter
        Raises:
            ValueError: If the hash list is too short for the layer that will receive the value"""
        self._validate_hashes(hashes, self._required_hash_depth_for_insert(force))
        self._added_elements += 1
        if force or not self.check_alt(hashes):
            self._check_for_growth()
            self._validate_hashes(hashes, self._blooms[-1].number_hashes)
            self._blooms[-1].add_alt(hashes)

    def _max_hash_depth(self) -> int:
        if not self._blooms:
            return 0
        return max(blm.number_hashes for blm in self._blooms)

    def _new_layer_index(self) -> int:
        """Index in the budget schedule used for the next layer"""
        return self._next_layer_index

    def _next_layer_target_fpr(self) -> float:
        if not self.budget_mode:
            return self._fpr
        return self._budget_layer_fpr(self._new_layer_index())

    def _next_layer_hash_depth(self) -> int:
        if not self.budget_mode:
            return self._max_hash_depth()
        _, hashes, _ = BloomFilter._get_optimized_params(self._est_elements, self._next_layer_target_fpr())
        return hashes

    def _required_hash_depth_for_insert(self, force: bool) -> int:
        depth = self._max_hash_depth()
        newest_is_full = bool(self._blooms) and self._blooms[-1].elements_added >= self._est_elements
        if force or newest_is_full:
            depth = max(depth, self._next_layer_hash_depth())
        return depth

    @staticmethod
    def _validate_hashes(hashes: HashResultsT, depth: int) -> None:
        if len(hashes) < depth:
            raise ValueError(
                f"ExpandingBloomFilter: hash list has {len(hashes)} values but a layer requires {depth}; "
                "regenerate the hashes at the required depth"
            )

    def _add_bloom_filter(self):
        """build a new bloom and add it on!"""
        layer_index = self._new_layer_index()
        layer_fpr = self._next_layer_target_fpr()
        blm = BloomFilter(
            est_elements=self._est_elements,
            false_positive_rate=layer_fpr,
            hash_function=self._hash_func,
        )
        self._blooms.append(blm)
        self._layer_fprs.append(layer_fpr)
        self._next_layer_index = max(self._next_layer_index, layer_index + 1)

    def _check_for_growth(self):
        """determine if the bloom filter should automatically grow"""
        if self._blooms[-1].elements_added >= self._est_elements:
            self._add_bloom_filter()

    def export(self, file: Path | str | IOBase | mmap) -> None:
        """Export an expanding Bloom Filter, or subclass, to disk

        Args:
            filepath (str): The path to the file to import"""
        if not isinstance(file, IOBase | mmap):
            file = resolve_path(file)
            with open(file, "wb") as filepointer:
                self.export(filepointer)  # type:ignore
        else:
            filepointer = file  # type:ignore
            # add all the different Bloom bit arrays...
            for blm in self._blooms:
                filepointer.write(self.__S_INT64_STRUCT.pack(blm.elements_added))
                blm.bloom.tofile(filepointer)
            if self.budget_mode:
                for rate in self._layer_fprs:
                    filepointer.write(self.__BUDGET_FPR_STRUCT.pack(rate))
                filepointer.write(
                    self.__BUDGET_FOOTER_STRUCT.pack(
                        len(self._blooms),
                        self.estimated_elements,
                        self.elements_added,
                        self._next_layer_index,
                        self.false_positive_rate,
                        self._budget_mode,
                        self._BUDGET_GEOMETRIC,
                        self.__BUDGET_MAGIC,
                    )
                )
            else:
                filepointer.write(
                    self.__FOOTER_STRUCT.pack(
                        len(self._blooms),
                        self.estimated_elements,
                        self.elements_added,
                        self.false_positive_rate,
                    )
                )

    def _load(self, file: Path | str | IOBase | mmap):
        """load a file"""
        if not isinstance(file, IOBase | mmap):
            file = resolve_path(file)
            with MMap(file) as filepointer:
                self._load(filepointer)
        else:
            metadata = self._parse_metadata(file)  # type: ignore
            self._blooms = []
            self._added_elements = metadata.elements_added
            self._est_elements = metadata.estimated_elements
            self._budget_mode = metadata.budget_mode
            self._overall_fpr = metadata.overall_false_positive_rate
            self._fpr = metadata.layer_false_positive_rates[0]
            self._parse_blooms(file, metadata)  # type:ignore

    @classmethod
    def _parse_metadata(cls, b: ByteString) -> _ExpandingBloomMetadata:
        raw = bytes(b)
        if len(raw) >= cls.__BUDGET_FOOTER_STRUCT.size and raw[-4:] == cls.__BUDGET_MAGIC:
            offset = cls.__BUDGET_FOOTER_STRUCT.size
            size, est_els, els_added, next_idx, overall, mode, strategy, _ = cls.__BUDGET_FOOTER_STRUCT.unpack(
                raw[-1 * offset :]
            )
            if mode != cls._BUDGET_GEOMETRIC or strategy != cls._BUDGET_GEOMETRIC:
                raise InitializationError("ExpandingBloomFilter: unsupported budget mode in file")
            if size < 1:
                raise InitializationError("ExpandingBloomFilter: invalid layer count in file")
            if not 0.0 < float(overall) < 1.0:
                raise InitializationError("ExpandingBloomFilter: invalid overall budget in file")
            rate_size = cls.__BUDGET_FPR_STRUCT.size
            fpr_start = len(raw) - offset - (int(size) * rate_size)
            if fpr_start < 0:
                raise InitializationError("ExpandingBloomFilter: invalid budget footer in file")
            layer_fprs = tuple(
                float(cls.__BUDGET_FPR_STRUCT.unpack_from(raw, fpr_start + (idx * rate_size))[0])
                for idx in range(int(size))
            )
            if any(not 0.0 < rate < 1.0 for rate in layer_fprs):
                raise InitializationError("ExpandingBloomFilter: invalid layer rate in file")
            if int(next_idx) < int(size):
                raise InitializationError("ExpandingBloomFilter: invalid layer schedule in file")
            return _ExpandingBloomMetadata(
                int(size),
                int(est_els),
                int(els_added),
                float(overall),
                layer_fprs,
                fpr_start,
                int(mode),
                int(next_idx),
            )

        offset = cls.__FOOTER_STRUCT.size
        size, est_els, els_added, fpr = cls.__FOOTER_STRUCT.unpack(raw[-1 * offset :])
        layer_fprs = (float(fpr),) * int(size)
        return _ExpandingBloomMetadata(
            int(size),
            int(est_els),
            int(els_added),
            None,
            layer_fprs,
            len(raw) - offset,
            cls._BUDGET_NONE,
            int(size),
        )

    @classmethod
    def _parse_footer(cls, b: ByteString) -> tuple[int, int, int, float]:
        """Parse the legacy footer; kept for compatibility with existing callers"""
        offset = cls.__FOOTER_STRUCT.size
        size, est_els, els_added, fpr = cls.__FOOTER_STRUCT.unpack(bytes(b[-1 * offset :]))
        return int(size), int(est_els), int(els_added), float(fpr)

    def _parse_blooms(self, b: ByteString, metadata: _ExpandingBloomMetadata) -> None:
        # reset the bloom list
        self._blooms = []
        self._layer_fprs = []
        start = 0
        data = bytes(b)
        for idx in range(metadata.size):
            layer_fpr = metadata.layer_false_positive_rates[idx]
            blm = BloomFilter(
                est_elements=self._est_elements,
                false_positive_rate=layer_fpr,
                hash_function=self._hash_func,
            )
            blm_size = self._BLOOM_ELEMENT_SIZE * blm.bloom_length
            end = start + self.__S_INT64_STRUCT.size + blm_size
            if end > metadata.data_length:
                raise InitializationError("ExpandingBloomFilter: truncated or invalid bloom data in file")
            blm._els_added = int(self.__S_INT64_STRUCT.unpack(data[start : start + self.__S_INT64_STRUCT.size])[0])
            blm._bloom = array("B", data[start + self.__S_INT64_STRUCT.size : end])
            self._blooms.append(blm)
            self._layer_fprs.append(layer_fpr)
            start = end
        self._next_layer_index = metadata.next_layer_index


class RotatingBloomFilter(ExpandingBloomFilter):
    """Simple Rotating Bloom Filter implementation that allows for the "older"
    elements added to be removed, in chunks. As the queue fills up, those
    elements inserted earlier will be bulk removed. This also provides the
    user with the oportunity to force the removal instead of it being time
    based.

    Args:
        est_elements (int): The number of estimated elements to be added
        false_positive_rate (float): The desired false positive rate
        max_queue_size (int): This is the number is used to determine the maximum number of Bloom Filters. \
            Total elements added is based on `max_queue_size * est_elements`
        filepath (str): Path to file to load
        hash_function (function): Hashing strategy function to use `hf(key, number)`
        overall_false_positive_rate (float): Explicit overall false positive budget; \
            enables budget mode for each layer in the queue
    Note:
        Initialization order of operations:
            1) Filepath
            2) est_elements and false_positive_rate

        In budget mode each layer uses the geometric rate for its position in
        the queue. The active union bound therefore stays below the overall
        budget as layers rotate in and out.
    """

    __slots__ = ("_queue_size",)

    def __init__(
        self,
        est_elements: int | None = None,
        false_positive_rate: float | None = None,
        max_queue_size: int = 10,
        filepath: str | Path | None = None,
        hash_function: HashFuncT | None = None,
        overall_false_positive_rate: float | None = None,
        **kwargs,
    ) -> None:
        """initialize"""
        self._queue_size = max_queue_size
        super().__init__(
            est_elements=est_elements,
            false_positive_rate=false_positive_rate,
            filepath=filepath,
            hash_function=hash_function,
            overall_false_positive_rate=overall_false_positive_rate,
            **kwargs,
        )

    @classmethod
    def frombytes(  # type:ignore
        cls, b: ByteString, max_queue_size: int, hash_function: HashFuncT | None = None
    ) -> "RotatingBloomFilter":
        """
        Args:
            b (ByteString): The bytes to load as a Expanding Bloom Filter
            max_queue_size (int): This is the number is used to determine the maximum number \
                of Bloom Filters. Total elements added is based on `max_queue_size * est_elements`
            hash_function (function): Hashing strategy function to use `hf(key, number)`
        Returns:
            RotatingBloomFilter: A Bloom Filter object
        """
        metadata = cls._parse_metadata(b)
        if metadata.budget_mode:
            blm = RotatingBloomFilter(
                est_elements=metadata.estimated_elements,
                overall_false_positive_rate=metadata.overall_false_positive_rate,
                max_queue_size=max_queue_size,
                hash_function=hash_function,
            )
        else:
            blm = RotatingBloomFilter(
                est_elements=metadata.estimated_elements,
                false_positive_rate=metadata.layer_false_positive_rates[0],
                max_queue_size=max_queue_size,
                hash_function=hash_function,
            )
        blm._parse_blooms(b, metadata)
        blm._added_elements = metadata.elements_added
        return blm

    @property
    def max_queue_size(self) -> int:
        """int: The maximum size for the queue"""
        return self._queue_size

    @property
    def current_queue_size(self) -> int:
        """int: The current size of the queue"""
        return len(self._blooms)

    def _new_layer_index(self) -> int:
        """Allocate by queue position so the active union bound stays within budget"""
        if not self.budget_mode:
            return self._next_layer_index
        return min(len(self._blooms), self._queue_size - 1)

    def add_alt(self, hashes: HashResultsT, force: bool = False) -> None:
        """Add the element represented by hashes into the Bloom Filter

        Args:
            hashes (list): A list of integers representing the key to insert
            force (bool): `True` will force it to be inserted, even if it likely has been inserted \
                before `False` will only insert if not found in the Bloom Filter
        Raises:
            ValueError: If the hash list is too short for the layer that will receive the value"""
        self._validate_hashes(hashes, self._required_hash_depth_for_insert(force))
        self._added_elements += 1
        if force or not self.check_alt(hashes):
            self.__rotate_bloom_filter()
            self._validate_hashes(hashes, self._blooms[-1].number_hashes)
            self._blooms[-1].add_alt(hashes)

    def pop(self) -> None:
        """Pop the oldest Bloom Filter off of the queue without pushing a new
        Bloom Filter onto the queue

        Raises:
            RotatingBloomFilterError: Unable to rotate the Bloom Filter"""
        if self.current_queue_size == 1:
            msg = "Popping a Bloom Filter will result in an unusable system!"
            raise RotatingBloomFilterError(msg)
        self._blooms.pop(0)
        self._layer_fprs.pop(0)

    def push(self) -> None:
        """Push a new bloom filter onto the queue and rotate if necessary"""
        self.__rotate_bloom_filter(force=True)

    def __rotate_bloom_filter(self, force: bool = False):
        """handle determining if/when the Bloom Filter queue needs to be rotated"""
        blm = self._blooms[-1]
        ready_to_rotate = blm.elements_added >= blm.estimated_elements
        no_need_to_pop = self.current_queue_size < self._queue_size
        if force and no_need_to_pop:
            self._add_bloom_filter()
        elif force:  # must need to be pop'd first!
            self._blooms.pop(0)
            self._layer_fprs.pop(0)
            self._add_bloom_filter()
        elif ready_to_rotate and no_need_to_pop:
            self._add_bloom_filter()
        elif ready_to_rotate:
            self._blooms.pop(0)
            self._layer_fprs.pop(0)
            self._add_bloom_filter()
