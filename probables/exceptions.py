"""PyProbables Exceptions"""


class ProbablesBaseException(Exception):
    """Base ProbablesBaseException

    Args:
        message (str): The error message to be reported"""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InitializationError(ProbablesBaseException):
    """Initialization Exception

    Args:
        message (str): The initialization error messge"""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class NotSupportedError(ProbablesBaseException):
    """Not Supported Functionality Exception

    Args:
        message (str): The error message to be reported"""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class SimilarityError(ProbablesBaseException):
    """Similarity Exception

    Args:
        message (str): The error message to be reported"""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class CuckooFilterFullError(ProbablesBaseException):
    """Cuckoo Filter Full Exception

    Args:
        message (str): The error message to be reported"""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class RotatingBloomFilterError(ProbablesBaseException):
    """RotatingBloomFilter unable to rotate Blooms Exceptions

    Args:
        message (str): The error message to be reported"""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class CountMinSketchError(ProbablesBaseException):
    """CountMinSketch Exception

    Args:
        message (str): The error message to be reported"""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class QuotientFilterError(ProbablesBaseException):
    """Quotient Filter Exception

    Args:
        message (str): The error message to be reported"""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class DurableBloomError(ProbablesBaseException):
    """Durable (on-disk, cross process) Bloom Filter base Exception

    Args:
        message (str): The error message to be reported"""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class StaleWriterError(DurableBloomError):
    """Raised when a writer fenced off by a partition-generation takeover tries
    to commit. The caller explicitly incremented the partition generation when
    another process took over, so the late writer is rejected rather than being
    detected by a timeout.

    Args:
        message (str): The error message to be reported"""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class DurableBloomRecoveryError(DurableBloomError):
    """Raised when on-disk batch journals or generation layers fail validation
    while recovering the durable Bloom Filter.

    Args:
        message (str): The error message to be reported"""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)
