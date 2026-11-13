"""Quoting rules for SQL dialects.

A :class:`QuoteProfile` describes how quoted literals and identifiers are
delimited.  ``sqlparse`` itself still accepts both the SQL standard and the
common backslash extension when no profile is given.  Passing a profile makes
the lexer follow exactly one interpretation, which is what data ingestion
needs: the same bytes can then no longer be read as a standard literal in one
code path and as a backslash-escaped literal in another.
"""

import re
from dataclasses import dataclass

__all__ = [
    'ANSI',
    'BACKSLASH',
    'DEFAULT',
    'MYSQL',
    'POSTGRES',
    'PROFILES',
    'QuoteProfile',
    'decode_string',
    'get_profile',
    'unquote_identifier',
]


@dataclass(frozen=True)
class QuoteProfile:
    """Description of the quoting rules of one source dialect.

    :param name: Name used in manifests and error messages.
    :param backslash_escapes: If true, a backslash escapes the next
        character inside single- and double-quoted literals (MySQL and
        PostgreSQL ``E'...'`` style).  If false, backslash is an ordinary
        character.
    :param doubled_quotes: If true, two consecutive quote characters encode
        one quote character (the SQL standard form).
    :param backtick_escapes: If true, a backslash escapes characters inside
        backtick-quoted identifiers as well.
    """

    name: str
    backslash_escapes: bool = False
    doubled_quotes: bool = True
    backtick_escapes: bool = False

    def quote_regex(self, quote, backslash=None):
        """Return a regex matching a complete quoted span."""
        if backslash is None:
            backslash = self.backslash_escapes
        alternatives = []
        if self.doubled_quotes:
            alternatives.append(re.escape(quote) * 2)
        if backslash:
            alternatives.append(r'\\[\s\S]')
        excluded = re.escape(quote)
        if backslash:
            excluded += r'\\'
        alternatives.append(f'[^{excluded}]')
        body = '|'.join(alternatives)
        return f'{re.escape(quote)}(?:{body})*{re.escape(quote)}'

    @property
    def label(self):
        return self.name


#: Behaviour without an explicit profile: both the standard doubled quote
#: and the backslash escape are accepted.
DEFAULT = QuoteProfile('default', backslash_escapes=True)
#: SQL standard quoting: no backslash escapes, doubled quotes only.
ANSI = QuoteProfile('ansi')
#: MySQL style quoting: backslash escapes inside strings.
MYSQL = QuoteProfile('mysql', backslash_escapes=True)
#: PostgreSQL with ``standard_conforming_strings = off`` / ``E'...'``.
POSTGRES = QuoteProfile('postgres', backslash_escapes=True)
#: Explicit alias for the backslash dialect used by dump writers.
BACKSLASH = QuoteProfile('backslash', backslash_escapes=True)

PROFILES = {
    'default': DEFAULT,
    'ansi': ANSI,
    'standard': ANSI,
    'sql': ANSI,
    'sqlite': ANSI,
    'postgres': POSTGRES,
    'postgresql': POSTGRES,
    'mysql': MYSQL,
    'backslash': BACKSLASH,
    'backslash-escapes': BACKSLASH,
}


def get_profile(profile):
    """Resolve *profile* to a :class:`QuoteProfile`.

    ``None`` selects :data:`DEFAULT`.  A string is looked up in
    :data:`PROFILES`; any other object is expected to be a
    :class:`QuoteProfile` already.
    """
    if profile is None:
        return DEFAULT
    if isinstance(profile, QuoteProfile):
        return profile
    try:
        return PROFILES[profile.lower()]
    except (AttributeError, KeyError):
        raise ValueError(f'Unknown quoting profile: {profile!r}') from None


def unquote_identifier(value, profile=None):
    """Remove quoting from an identifier.

    Doubled closing characters are collapsed according to *profile*.
    """
    if value is None:
        return None
    if len(value) < 2:
        return value
    profile = get_profile(profile)
    open_char = value[0]
    close_char = ']' if open_char == '[' else open_char
    if open_char not in ('"', "'", '`', '[') or value[-1] != close_char:
        return value
    inner = value[1:-1]
    if profile.doubled_quotes and open_char != '[':
        inner = inner.replace(close_char * 2, close_char)
    return inner


_BACKSLASH_ESCAPES = {
    '0': '\0',
    'b': '\b',
    'n': '\n',
    'r': '\r',
    't': '\t',
    'Z': '\x1a',
    '\\': '\\',
    "'": "'",
    '"': '"',
}


def decode_string(value, profile=None):
    """Decode a quoted SQL string literal.

    The input must include its surrounding quotes.  A short character set
    introducer (``N'...'``, ``E'...'`` or MySQL's ``_utf8mb4'...'``) is
    accepted and ignored; the profile decides how escapes are handled.
    """
    profile = get_profile(profile)
    match = re.fullmatch(
        r"(?:_[A-Za-z0-9]+|[NnEe])?"
        r"(?P<quote>['\"])(?P<body>.*)(?P=quote)",
        value,
        re.DOTALL,
    )
    if match is None:
        return value
    quote = match.group('quote')
    body = match.group('body')
    result = []
    index = 0
    length = len(body)
    while index < length:
        char = body[index]
        if (profile.doubled_quotes and char == quote
                and index + 1 < length and body[index + 1] == quote):
            result.append(quote)
            index += 2
            continue
        if profile.backslash_escapes and char == '\\' and index + 1 < length:
            escaped = body[index + 1]
            result.append(_BACKSLASH_ESCAPES.get(escaped, escaped))
            index += 2
            continue
        result.append(char)
        index += 1
    return ''.join(result)
