#
# Copyright (C) 2009-2020 the sqlparse authors and contributors
# <see AUTHORS file>
#
# This module is part of python-sqlparse and is released under
# the BSD License: https://opensource.org/licenses/BSD-3-Clause

"""filter"""

from sqlparse import lexer
from sqlparse.engine import grouping
from sqlparse.engine.statement_splitter import StatementSplitter
from sqlparse.exceptions import SQLParseError
from sqlparse.filters import StripTrailingSemicolonFilter


class FilterStack:
    def __init__(self, strip_semicolon=False):
        self.preprocess = []
        self.stmtprocess = []
        self.postprocess = []
        self._grouping = False
        #: When True every token carries its source range, even if no
        #: verbatim region was resolved.
        self._lineage = False
        #: Resolved verbatim regions.
        self.directives = ()
        if strip_semicolon:
            self.stmtprocess.append(StripTrailingSemicolonFilter())

    def enable_grouping(self):
        self._grouping = True

    def enable_lineage(self, directives=()):
        """Switch to offset-tracked tokenization for verbatim regions."""
        self._lineage = True
        self.directives = tuple(directives)

    def run(self, sql, encoding=None):
        try:
            if self._lineage:
                stream = lexer.tokenize_with_offsets(
                    sql, self.directives, encoding)
                splitter = StatementSplitter(track_offsets=True)
            else:
                stream = lexer.tokenize(sql, encoding)
                splitter = StatementSplitter()

            # Process token stream
            for filter_ in self.preprocess:
                stream = filter_.process(stream)

            stream = splitter.process(stream)

            # Output: Stream processed Statements
            for stmt in stream:
                if self._grouping:
                    stmt = grouping.group(stmt)

                for filter_ in self.stmtprocess:
                    filter_.process(stmt)

                for filter_ in self.postprocess:
                    stmt = filter_.process(stmt)

                yield stmt
        except RecursionError as err:
            raise SQLParseError('Maximum recursion depth exceeded') from err
