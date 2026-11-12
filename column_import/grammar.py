"""Shared column-mapping grammar for the two data-import streams.

Both import streams build on this single grammar instance; it is never copied.
All code-table awareness lives in parse actions that read the active
:class:`~column_import.ParseSession`, so one grammar can serve batches with
different semantic versions in turn.

Row format (one row per line)::

    CODE,CODE,...[,{CODE,CODE,...}]

The part before ``{`` is the parent row: every short code is resolved against
the session's ``"columns"`` code table.  The braced part is an attached table
resolved recursively against the separate ``"attachment_columns"`` table, so a
child parse can never pick up a value meant for the parent code table.  A
failure inside the attached table is recorded on that row's ``attachment``
field instead of aborting the outer parse, so the parent row's already
resolved columns survive.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pyparsing as pp

from . import CodeLookupError, code_lookup, nested_parse, require_session

COLUMNS_TABLE = "columns"
ATTACHMENT_TABLE = "attachment_columns"

# Newlines separate rows, so they must not be treated as ignorable whitespace
# anywhere inside the grammar.  This is set locally on the grammar elements
# rather than changing pyparsing's global defaults.
_INLINE_WHITESPACE = " \t"


def _resolve_column_code(tokens: pp.ParseResults) -> Any:
    return code_lookup(COLUMNS_TABLE, tokens[0])


def _resolve_attachment_code(tokens: pp.ParseResults) -> Any:
    return code_lookup(ATTACHMENT_TABLE, tokens[0])


# A short code as found in a raw row.
_code_token = pp.Word(pp.alphanums + "_-.")
mapped_code = _code_token.copy().set_parse_action(_resolve_column_code)
attachment_code = _code_token.copy().set_parse_action(_resolve_attachment_code)

# Attached table: a comma-separated list of short codes inside braces.  It uses
# its own code table via attachment_code, never the parent column table.
child_table = pp.DelimitedList(attachment_code).set_name("child_table")

# Parent row columns before the optional braced attachment.
parent_columns = pp.DelimitedList(mapped_code).set_name("parent_columns")


def _parse_attachment(tokens: pp.ParseResults) -> Dict[str, Any]:
    raw = tokens.raw.strip()
    try:
        child_result = nested_parse(child_table, raw, parse_all=True)
    except (pp.ParseBaseException, CodeLookupError) as exc:
        # Inner failure must not kill the outer row.  Keep the raw text and
        # mark the attachment as failed; already computed parent columns are
        # left untouched.
        return {"raw": raw, "items": None, "error": str(exc)}
    return {"raw": raw, "items": child_result.as_list(), "error": None}


_attachment_body = (
    pp.Suppress("{")
    + pp.SkipTo(pp.Suppress("}"), include=True)
    .set_name("attachment_raw")("raw")
).set_parse_action(_parse_attachment)

# The leading comma belongs to the attachment: the parent's delimited list
# backtracks over it once the next column fails to match the opening brace.
_attachment = pp.Suppress(",") + _attachment_body

row = pp.Group(
    parent_columns("columns") + pp.Optional(_attachment)("attachment")
).set_name("row")

document = pp.DelimitedList(row, delim=pp.line_end()).set_name("document")
document.ignore(pp.python_style_comment)


def _grammar_elements(root: pp.ParserElement) -> List[pp.ParserElement]:
    """Return every parser element reachable from ``root`` (once each)."""
    seen: set[int] = set()
    ordered: List[pp.ParserElement] = []

    def visit(element: pp.ParserElement) -> None:
        if id(element) in seen:
            return
        seen.add(id(element))
        ordered.append(element)
        for child in element.recurse():
            visit(child)

    visit(root)
    return ordered


# Restrict whitespace locally; importing this module leaves pyparsing's global
# DEFAULT_WHITE_CHARS untouched for unrelated parsers in the same process.
for _element in _grammar_elements(document):
    _element.set_whitespace_chars(_INLINE_WHITESPACE)


def parse_document(text: str) -> List[Dict[str, Any]]:
    """Parse a full import document on the currently active session.

    Raises :class:`RuntimeError` if called outside a :class:`ParseSession`.
    Returns one dict per row with the resolved output columns.
    """
    require_session()
    result = document.parse_string(text, parse_all=True)
    rows: List[Dict[str, Any]] = []
    for parsed_row in result:
        columns = list(parsed_row.columns) if parsed_row.columns else []
        if parsed_row.attachment:
            # The attachment parse action returns one dict; unwrap the
            # ParseResults layer so callers see plain data structures.
            attachment: Optional[Dict[str, Any]] = parsed_row.attachment[0]
        else:
            attachment = None
        rows.append({"columns": columns, "attachment": attachment})
    return rows


# The two import streams are identical grammatically; they only differ by the
# ``source`` recorded on the session that runs them.
parse_primary_stream = parse_document
parse_secondary_stream = parse_document
