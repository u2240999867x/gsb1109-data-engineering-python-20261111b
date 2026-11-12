"""
Exception classes.

.. versionadded:: 1.0.0
"""

from __future__ import annotations


__all__ = ("InvalidMethod", "StaleCursorError")


# NOTE: This needs to subclass AttributeError due to compatibility with typing.Protocol and
#  runtime_checkable. See https://github.com/dgilland/pydash/issues/165
class InvalidMethod(AttributeError):
    """
    Raised when an invalid pydash method is invoked through :func:`pydash.chaining.chain`.

    .. versionadded:: 1.0.0
    """

    pass


class StaleCursorError(ValueError):
    """
    Raised when a pagination cursor was issued under a different
    :class:`pydash.sorting.SortPolicy` than the one decoding it.

    .. versionadded:: 8.2.0
    """

    pass
