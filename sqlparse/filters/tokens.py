#
# Copyright (C) 2009-2020 the sqlparse authors and contributors
# <see AUTHORS file>
#
# This module is part of python-sqlparse and is released under
# the BSD License: https://opensource.org/licenses/BSD-3-Clause

from sqlparse import tokens as T


class _CaseFilter:
    ttype = None

    def __init__(self, case=None):
        case = case or 'upper'
        self.convert = getattr(str, case)

    def process(self, stream):
        for item in stream:
            ttype, value = item[0], item[1]
            if ttype in self.ttype:
                value = self.convert(value)
            # Lineage mode carries (ttype, value, start, end) tuples; the
            # source range survives the in-place case conversion.
            if len(item) == 4:
                yield ttype, value, item[2], item[3]
            else:
                yield ttype, value


class KeywordCaseFilter(_CaseFilter):
    ttype = T.Keyword


class IdentifierCaseFilter(_CaseFilter):
    ttype = T.Name, T.String.Symbol

    def process(self, stream):
        for item in stream:
            ttype, value = item[0], item[1]
            if ttype in self.ttype and value.strip()[0] != '"':
                value = self.convert(value)
            if len(item) == 4:
                yield ttype, value, item[2], item[3]
            else:
                yield ttype, value


class TruncateStringFilter:
    def __init__(self, width, char):
        self.width = width
        self.char = char

    def process(self, stream):
        for item in stream:
            ttype, value = item[0], item[1]
            if ttype == T.Literal.String.Single:
                if value[:2] == "''":
                    inner = value[2:-2]
                    quote = "''"
                else:
                    inner = value[1:-1]
                    quote = "'"

                if len(inner) > self.width:
                    value = ''.join((quote, inner[:self.width], self.char, quote))
            if len(item) == 4:
                yield ttype, value, item[2], item[3]
            else:
                yield ttype, value
