"""Incremental tracking of quoting state.

The statement splitter needs to know whether a semicolon sits inside a quoted
literal.  :class:`QuoteState` answers that for one quoting profile without
holding on to the whole text, which also lets a consumer stop in the middle
of a literal and resume later without borrowing state from another stream.

Only the delimiters that matter for statement boundaries are tracked:
single-quoted literals and, because the lexer also treats them as quoted,
double-quoted ones.  Comments, parenthesised expressions and dollar-quoted
bodies are left to the full lexer; ``QuoteState`` is a fast pre-check, not a
replacement for it.
"""

from sqlparse import profiles


class QuoteState:
    """Track whether a position is inside a quoted literal.

    :param profile: Quoting profile (name or :class:`QuoteProfile`).
    """

    __slots__ = ('_close_quote', '_escaped', '_pending', 'profile')

    def __init__(self, profile=None):
        self.profile = profiles.get_profile(profile)
        self.reset()

    def reset(self):
        """Forget the current literal and return to the initial state."""
        self._close_quote = None
        self._escaped = False
        self._pending = False

    @property
    def in_string(self):
        """True while a quoted literal is still open.

        A quote that may be either a closing quote or the first half of a
        doubled quote counts as closed until the next character decides.
        """
        return self._close_quote is not None and not self._pending

    @property
    def quote(self):
        """The opening character of the current literal, or ``None``."""
        return self._close_quote

    def copy(self):
        """Return an independent copy of the current state."""
        clone = type(self).__new__(type(self))
        clone.profile = self.profile
        clone._close_quote = self._close_quote
        clone._escaped = self._escaped
        clone._pending = self._pending
        return clone

    def feed(self, text):
        """Consume *text* and return the state after its last character."""
        for char in text:
            self.step(char)
        return self

    def step(self, char):
        """Consume a single character.

        Returns ``True`` if the character belongs to a quoted literal and
        ``False`` if it is ordinary text.  A closing quote itself reports
        ``True``; a following semicolon reports ``False``.
        """
        if self._close_quote is None:
            if char in ("'", '"'):
                self._close_quote = char
                self._escaped = False
                self._pending = False
                return True
            return False

        if self._escaped:
            # The character after a backslash never closes the literal.
            self._escaped = False
            self._pending = False
            return True

        if self.profile.backslash_escapes and char == '\\':
            self._escaped = True
            self._pending = False
            return True

        if char == self._close_quote:
            if not self.profile.doubled_quotes:
                self._close_quote = None
                self._pending = False
                return True
            if self._pending:
                # The second of two consecutive quotes: stay in the literal.
                self._pending = False
                return True
            # Wait for the next character to decide whether this is the
            # first half of a doubled quote or the actual close.
            self._pending = True
            return True

        if self._pending:
            self._pending = False
            self._close_quote = None
            # The pending quote closed the literal; this character is
            # ordinary text and has to be classified as such.
            return self.step(char)

        return True

    def close_pending(self):
        """Resolve state at end of text; returns True if it changed."""
        if self._pending:
            self._pending = False
            self._close_quote = None
            return True
        return False
