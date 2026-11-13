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


class WindowSummaryError(ProbablesBaseException):
    """Windowed summary Exception, raised when an event cannot be placed into
    a mergeable windowed summary (for example, a late event arriving after the
    partition has been sealed with a rejecting late-event policy) or when two
    summaries with incompatible parameters are merged or re-granulated.

    Args:
        message (str): The error message to be reported"""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)
