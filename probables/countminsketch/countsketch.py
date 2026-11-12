"""Count-Sketch: a signed sketch for frequency and inner-product estimation

License: MIT
"""

import math
from array import array
from numbers import Number
from statistics import median

from probables.constants import INT64_T_MAX, INT64_T_MIN
from probables.exceptions import CountMinSketchError, InitializationError
from probables.hashes import HashFuncT, HashResultsT, KeyT, default_fnv_1a, fnv_1a

SignHashFuncT = HashFuncT  # same signature; expected to return only +/- 1 values

# offset used to keep the default sign hash independent from the bucket hash
_SIGN_SEED_OFFSET = 0x9E3779B9


def default_sign_hash(key: KeyT, depth: int = 1) -> HashResultsT:
    """The default sign hashing routine; maps a key to +/- 1 per row

    Args:
        key (str): The element to be hashed
        depth (int): The number of hash permutations to compute
    Returns:
        list(int): List of size depth of either 1 or -1"""
    return [1 if fnv_1a(key, seed=_SIGN_SEED_OFFSET + idx) & 1 == 0 else -1 for idx in range(depth)]


class CountSketchInnerProduct:
    """The result of a Count-Sketch inner-product estimation

    Note:
        Rows of the sketch are only independent observations when they use
        independent hash functions. When two reports reuse the same hash
        rows, the caller declares the independent groups explicitly and each
        group contributes a single observation (the mean of its rows) so that
        shared hash rows are not counted as extra independent observations."""

    __slots__ = ("__row_estimates", "__group_estimates", "__estimate")

    def __init__(self, row_estimates: HashResultsT, group_estimates: list[float], estimate: float) -> None:
        self.__row_estimates = row_estimates
        self.__group_estimates = group_estimates
        self.__estimate = estimate

    def __str__(self) -> str:
        """string representation of the inner-product estimate"""
        msg = "Count-Sketch Inner Product:\n\tEstimate: {0}\n\tRow Estimates: {1}\n\tGroup Estimates: {2}"
        return msg.format(self.estimate, self.row_estimates, self.group_estimates)

    @property
    def row_estimates(self) -> HashResultsT:
        """list(int): The inner-product estimate of each hash row

        Note:
            Not settable"""
        return self.__row_estimates

    @property
    def group_estimates(self) -> list[float]:
        """list(float): One estimate per independent hash-row group

        Note:
            Not settable"""
        return self.__group_estimates

    @property
    def estimate(self) -> float:
        """float: The aggregated inner-product estimate; the median of the
        independent group estimates

        Note:
            Not settable"""
        return self.__estimate


class CountSketch:
    """Signed Count-Sketch implementation for use in python; supports
    negative updates (e.g., refunds or revisions) so the net value of a key
    may be negative, and supports estimating the inner product of two
    sketches aligned by key without retaining the original keys

    Args:
        width (int): The width of the count-sketch
        depth (int): The depth of the count-sketch
        confidence (float): The level of confidence desired
        error_rate (float): The desired error rate
        hash_function (function): Hashing strategy function to use `hf(key, number)`
        sign_hash_function (function): Hashing strategy mapping keys to +/- 1 per row
    Returns:
        CountSketch: A Count-Sketch object
    Note:
        Initialization order of operations:
            1) Width and depth
            2) Confidence and error rate
    Note:
        Values are stored as signed 64-bit integers; scale fractional
        amounts (e.g., to cents) before inserting
    Note:
        For width and depth, width may realistically be in the thousands while depth \
            is in the single digit to teens"""

    __slots__ = (
        "__width",
        "__depth",
        "__confidence",
        "__error_rate",
        "__elements_added",
        "_bins",
        "_hash_function",
        "_sign_hash_function",
    )

    def __init__(
        self,
        width: int | None = None,
        depth: int | None = None,
        confidence: float | None = None,
        error_rate: float | None = None,
        hash_function: HashFuncT | None = None,
        sign_hash_function: SignHashFuncT | None = None,
    ) -> None:
        """default initialization function"""
        self.__width = 0
        self.__depth = 0
        self.__confidence = 0.0
        self.__error_rate = 0.0
        self.__elements_added = 0

        if width is not None and depth is not None:
            valid_prms = isinstance(width, Number) and width > 0 and isinstance(depth, Number) and depth > 0
            if not valid_prms:
                msg = "CountSketch: width and depth must be greater than 0"
                raise InitializationError(msg)
            self.__width = int(width)
            self.__depth = int(depth)
            self.__confidence = 1 - (1 / math.pow(2, self.depth))
            self.__error_rate = 2 / self.width
        elif confidence is not None and error_rate is not None:
            valid_prms = (
                isinstance(confidence, Number) and confidence > 0 and isinstance(error_rate, Number) and error_rate > 0
            )
            if not valid_prms:
                msg = "CountSketch: confidence and error rate must be greater than 0"
                raise InitializationError(msg)
            self.__confidence = confidence
            self.__error_rate = error_rate
            self.__width = math.ceil(2 / error_rate)
            numerator = -1 * math.log(1 - confidence)
            self.__depth = math.ceil(numerator / 0.6931471805599453)
        else:
            msg = (
                "Must provide one of the following to initialize the "
                "Count-Sketch:\n"
                "    The width and depth,\n"
                "    OR confidence and error rate"
            )
            raise InitializationError(msg)
        self._bins = array("q", [0]) * (self.width * self.depth)

        if hash_function is None:
            self._hash_function = default_fnv_1a
        else:
            self._hash_function = hash_function
        if sign_hash_function is None:
            self._sign_hash_function = default_sign_hash
        else:
            self._sign_hash_function = sign_hash_function

    def __str__(self) -> str:
        """string representation of the count-sketch"""
        msg = "Count-Sketch:\n\tWidth: {0}\n\tDepth: {1}\n\tConfidence: {2}\n\tError Rate: {3}\n\tElements Added: {4}"
        return msg.format(
            self.width,
            self.depth,
            self.confidence,
            self.error_rate,
            self.elements_added,
        )

    def __contains__(self, key: KeyT) -> bool:
        """setup the `in` keyword"""
        return self.check(key) != 0

    @property
    def width(self) -> int:
        """int: The width of the count-sketch

        Note:
            Not settable"""
        return self.__width

    @property
    def depth(self) -> int:
        """int: The depth of the count-sketch

        Note:
            Not settable"""
        return self.__depth

    @property
    def confidence(self) -> float:
        """float: The confidence of the count-sketch

        Note:
            Not settable"""
        return self.__confidence

    @property
    def error_rate(self) -> float:
        """float: The error rate of the count-sketch

        Note:
            Not settable"""
        return self.__error_rate

    @property
    def elements_added(self) -> int:
        """int: The net number of elements added to the count-sketch;
        negative updates (e.g., refunds) reduce this value

        Note:
            Not settable"""
        return self.__elements_added

    def clear(self) -> None:
        """Reset the count-sketch to an empty state"""
        self.__elements_added = 0
        for i, _ in enumerate(self._bins):
            self._bins[i] = 0

    def hashes(self, key: KeyT, depth: int | None = None) -> HashResultsT:
        """Return the bucket hashes based on the provided key

        Args:
            key (str): The element to be hashed
            depth (int): Number of permutations of the hash to generate; if None, generate `depth` hashes
        Returns:
            List(int): A list of the hashes for the key in int form"""
        return self._hash_function(key, self.depth if depth is None else depth)

    def sign_hashes(self, key: KeyT, depth: int | None = None) -> HashResultsT:
        """Return the sign hashes (+/- 1) based on the provided key

        Args:
            key (str): The element to be hashed
            depth (int): Number of permutations of the hash to generate; if None, generate `depth` hashes
        Returns:
            List(int): A list of the signs for the key"""
        return self._sign_hash_function(key, self.depth if depth is None else depth)

    def add(self, key: KeyT, num_els: int = 1) -> int | float:
        """Insert the element `key` into the count-sketch; `num_els` may be
        negative to record refunds or revisions

        Args:
            key (str): The element to insert
            num_els (int): The signed number of times to insert the element
        Returns:
            int: The estimated net number of times the element was inserted"""
        return self.add_alt(self.hashes(key), self.sign_hashes(key), num_els)

    def add_alt(self, hashes: HashResultsT, signs: HashResultsT, num_els: int = 1) -> int | float:
        """Insert an element by using the hash and sign representation

        Args:
            hashes (list): The hashes representing the element to insert
            signs (list): The signs (+/- 1) representing the element to insert
            num_els (int): The signed number of times to insert the element
        Returns:
            int: The estimated net number of times the element was inserted"""
        bins = [(val % self.width) + (i * self.width) for i, val in enumerate(hashes)]
        estimates = []
        for i, idx in enumerate(bins):
            val = self._bins[idx] + (signs[i] * num_els)
            if val > INT64_T_MAX:
                val = INT64_T_MAX
            elif val < INT64_T_MIN:
                val = INT64_T_MIN
            self._bins[idx] = val
            estimates.append(signs[i] * val)
        self.__elements_added = self.__clamp_elements(self.__elements_added + num_els)
        return median(estimates)

    def remove(self, key: KeyT, num_els: int = 1) -> int | float:
        """Remove element 'key' from the count-sketch

        Args:
            key (str): The element to remove
            num_els (int): The number of times to remove the element
        Returns:
            int: The estimated net number of times the element was inserted"""
        return self.add(key, -1 * num_els)

    def remove_alt(self, hashes: HashResultsT, signs: HashResultsT, num_els: int = 1) -> int | float:
        """Remove an element by using the hash and sign representation

        Args:
            hashes (list): The hashes representing the element to remove
            signs (list): The signs (+/- 1) representing the element to remove
            num_els (int): The number of times to remove the element
        Returns:
            int: The estimated net number of times the element was inserted"""
        return self.add_alt(hashes, signs, -1 * num_els)

    def check(self, key: KeyT) -> int | float:
        """Check the net number of times element 'key' is in the count-sketch;
        the result may be negative

        Args:
            key (str): The key to check the net number of times inserted
        Returns:
            int: The estimated net number of times the element was inserted"""
        return self.check_alt(self.hashes(key), self.sign_hashes(key))

    def check_alt(self, hashes: HashResultsT, signs: HashResultsT) -> int | float:
        """Check the count-sketch for an element by using the hash and sign
        representation

        Args:
            hashes (list): The hashes representing the element to check
            signs (list): The signs (+/- 1) representing the element to check
        Returns:
            int: The estimated net number of times the element was inserted"""
        bins = [(val % self.width) + (i * self.width) for i, val in enumerate(hashes)]
        return median([signs[i] * self._bins[idx] for i, idx in enumerate(bins)])

    def join(self, second: "CountSketch") -> None:
        """Join two count-sketches into a single count-sketch; the calling
        count-sketch will have the resulting combined data. This is the
        supported way to aggregate partitions (e.g., dates): the sketches
        must share the same hash and sign convention so that the joined
        sketch reflects the final effective input

        Args:
            second (CountSketch): The count-sketch to merge
        Raises:
            TypeError: When second is not a :class:`CountSketch`
            CountMinSketchError: Raised when the count-sketches are not of the same dimensions \
                or do not share the same hash and sign convention
        Note:
            The calling count-sketch will contain the combined data
            once complete
        """
        self._validate_compatible(second)

        size = self.width * self.depth
        for i in range(size):
            if self._bins[i] in (INT64_T_MIN, INT64_T_MAX):
                continue
            tmp_els = self._bins[i] + second._bins[i]
            if tmp_els > INT64_T_MAX:
                self._bins[i] = INT64_T_MAX
            elif tmp_els < INT64_T_MIN:
                self._bins[i] = INT64_T_MIN
            else:
                self._bins[i] = tmp_els

        self.__elements_added = self.__clamp_elements(self.__elements_added + second.elements_added)

    def inner_product(
        self, second: "CountSketch", row_groups: list[list[int]] | None = None
    ) -> CountSketchInnerProduct:
        """Estimate the inner product of two count-sketches aligned by key
        (e.g., the net amounts of two reports per SKU) without needing the
        original keys

        Args:
            second (CountSketch): The other count-sketch
            row_groups (list(list(int))): The independent groups of hash rows, as explicitly \
                declared by the caller; each group must be a non-empty list of row indices and \
                the groups must partition `range(depth)`. Rows within a group are dependent \
                (e.g., two reports reusing the same hash rows) and contribute a single \
                observation; if None, every row is treated as its own independent group
        Returns:
            CountSketchInnerProduct: The per-row estimates, the per-group estimates, and the \
                aggregated estimate
        Raises:
            TypeError: When second is not a :class:`CountSketch`
            CountMinSketchError: Raised when the count-sketches are not of the same dimensions \
                or do not share the same hash and sign convention; since the original data is \
                not retained, mismatched sketches cannot be reconciled and the comparison is \
                refused
            InitializationError: Raised when row_groups is not a partition of `range(depth)`
        Note:
            The aggregate is the median of the independent group estimates; it is \
                never computed by taking a median per partition and summing. To aggregate \
                partitions (e.g., dates), first :func:`join` the sketches of each report \
                under the same hash and sign convention and then estimate the inner product
        """
        self._validate_compatible(second)
        groups = self.__validate_row_groups(row_groups)

        row_estimates = []
        for row in range(self.depth):
            offset = row * self.width
            est = 0
            for col in range(self.width):
                est += self._bins[offset + col] * second._bins[offset + col]
            row_estimates.append(est)

        group_estimates = [sum(row_estimates[row] for row in group) / len(group) for group in groups]
        return CountSketchInnerProduct(row_estimates, group_estimates, median(group_estimates))

    def _validate_compatible(self, second: "CountSketch") -> None:
        """validate that two count-sketches share dimensions and the same
        hash and sign convention"""
        if not isinstance(second, CountSketch):
            msg = f"Unable to merge a count-sketch with {type(second)}"
            raise TypeError(msg)
        if (
            (self.width != second.width)
            or (self.depth != second.depth)
            or (self.hashes("test") != second.hashes("test"))
            or (self.sign_hashes("test") != second.sign_hashes("test"))
        ):
            msg = (
                "Unable to compare as the count-sketches are mismatched and "
                "the original data is not retained to reconcile them"
            )
            raise CountMinSketchError(msg)

    def __validate_row_groups(self, row_groups: list[list[int]] | None) -> list[list[int]]:
        """validate that row_groups partitions the hash rows into the
        caller-declared independent groups"""
        if row_groups is None:
            return [[row] for row in range(self.depth)]
        seen = []
        for group in row_groups:
            if not group:
                msg = "CountSketch: row groups must be non-empty"
                raise InitializationError(msg)
            for row in group:
                if not isinstance(row, int) or row < 0 or row >= self.depth:
                    msg = f"CountSketch: row group index {row} is out of range"
                    raise InitializationError(msg)
                seen.append(row)
        if sorted(seen) != list(range(self.depth)):
            msg = "CountSketch: row groups must partition the hash rows exactly once"
            raise InitializationError(msg)
        return [list(group) for group in row_groups]

    @staticmethod
    def __clamp_elements(elements: int) -> int:
        """clamp the elements added to the int64 range"""
        if elements > INT64_T_MAX:
            return INT64_T_MAX
        if elements < INT64_T_MIN:
            return INT64_T_MIN
        return elements
