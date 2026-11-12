#
# Copyright (C) 2009-2020 the sqlparse authors and contributors
# <see AUTHORS file>
#
# This module is part of python-sqlparse and is released under
# the BSD License: https://opensource.org/licenses/BSD-3-Clause

"""Incremental lexical diagnostics.

A local text replacement (for example closing a previously unclosed quote)
can change how everything behind the edit is tokenized: a string literal may
start or stop swallowing semicolons, a block comment may merge several
statements, and dollar-quoted bodies may pair with different partners.
Re-running the lexer over the whole document on every keystroke is wasteful,
while guessing a resynchronization point from the nearest semicolon is wrong
-- that semicolon may live inside a literal the edit just opened or closed.

This module keeps a *lexical state catalog* of a document:

* :class:`LexSpan` records the source interval and the pending (unclosed)
  state of ordinary single/double-quoted strings, multiline comments and
  paired dollar-quoted literals;
* :class:`LexState` adds the lexical error intervals and the pending flag
  at end of input;
* :class:`StatementSpan` is the statement directory, one entry per
  statement as the :class:`~sqlparse.engine.StatementSplitter` would emit.

:func:`incremental_relex` re-tokenizes only the region that may actually
have changed.  The unchanged suffix directory is reused -- its intervals
translated by the replacement delta -- only after a genuine statement
boundary at which both sides are back at root lexical state (no pending
span starts before it and no old pending span crosses it) *and* the
remaining raw text is identical.  The nearest semicolon alone is never
accepted as a sync point, since a semicolon inside a literal is not a
statement boundary.  If no such boundary is found, the whole (typically
short) affected region is re-lexed to end of input and the earliest
ingested statement touched by the edit is reported; the document can
additionally be rebuilt with :func:`build_lex_state` /
:func:`build_statement_catalog` to reconcile the result.
"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from sqlparse import keywords
from sqlparse import tokens as T
from sqlparse.engine.statement_splitter import StatementSplitter
from sqlparse.lexer import Lexer
from sqlparse.utils import _STRING_CONTENT

# Kind markers for the span stack.
_STRING = 'string'
_COMMENT = 'comment'
_DOLLAR = 'dollar'

# Raw text that has to match behind a realignment point before the old
# suffix directory may be reused.
_ALIGN_WINDOW = 64
# The re-lexed window starts with this size and doubles until the lexical
# states realign or end of input is reached.
_INITIAL_WINDOW = 4096


@dataclass(frozen=True)
class LexSpan:
    """A lexical construct with a source interval.

    :param ttype: token type of the construct when closed, or the type it
        would have once closed.
    :param start: inclusive start offset.
    :param end: exclusive end offset; for a pending span this reaches end of
        input.
    :param kind: one of the ``_STRING`` / ``_COMMENT`` / ``_DOLLAR`` markers.
    :param tag: delimiter tag (dollar-quote tag or ``'/*'``), else ``None``.
    :param pending: ``True`` if the closing delimiter is missing.
    """

    ttype: object
    start: int
    end: int
    kind: str
    tag: str | None = None
    pending: bool = False


@dataclass(frozen=True)
class LexState:
    """Serializable lexical state of a whole document."""

    spans: tuple[LexSpan, ...]
    errors: tuple[tuple[int, int], ...]
    incomplete: bool

    def pending_span(self):
        """Return the first pending span reaching end of input, or None."""
        for span in self.spans:
            if span.pending:
                return span
        return None


@dataclass(frozen=True)
class StatementSpan:
    """Source interval of one statement in the directory."""

    start: int
    end: int

    def slice(self, text):
        return text[self.start:self.end]


@dataclass(frozen=True)
class IncrementalResult:
    """Result of an incremental re-lex.

    :param catalog: statement directory of the new document.
    :param state: lexical state (spans, errors, pending flag) of the new
        document.
    :param changed_from: index of the first old statement whose interval
        the replacement overlaps; equals ``len(old_catalog)`` when the edit
        only trails the directory.
    :param earliest_affected: index in ``catalog`` of the earliest ingested
        statement that had to be rebuilt (merged, split or restored);
        ``None`` when no ingested statement is affected.  When
        :attr:`realigned` is ``False`` this is the first statement to
        rebuild in order to reconcile the document.
    :param shifted_from: when the suffix was reused, index into the old
        directory of the first statement that was kept and only shifted by
        the replacement delta; ``None`` when nothing was reused.  Old
        statements from :attr:`changed_from` up to this index are the ones
        whose shape (merge/split/restore) actually changed.
    :param realigned: ``True`` once new and old token streams plus the
        remaining raw text matched at a statement boundary.
    :param suffix_reused: ``True`` when directory entries behind the
        realignment point were translated instead of being re-tokenized.
    :param recovered: ``True`` when a pending span / error overlapping the
        edited region no longer exists in the new document.
    :param relex_end: exclusive offset up to which the new document was
        actually re-tokenized.
    """

    catalog: list[StatementSpan]
    state: LexState
    changed_from: int
    earliest_affected: int | None
    shifted_from: int | None
    realigned: bool
    suffix_reused: bool
    recovered: bool
    relex_end: int


def _tokenize_window(lexer, text, lo, hi):
    """Tokenize ``text[lo:hi]`` with absolute offsets."""
    positioned = []
    pos = lo
    for ttype, value in lexer.get_tokens(text[lo:hi]):
        positioned.append((ttype, value, pos, pos + len(value)))
        pos += len(value)
    return positioned


def _split_catalog(tokens, base):
    """Run the statement splitter over positioned tokens.

    The window always starts at a statement boundary, so a fresh splitter
    has exactly the state a full-document run would have there.
    """
    splitter = StatementSplitter()
    cursor = base
    catalog = []
    stream = ((ttype, value) for ttype, value, _start, _end in tokens)
    for statement in splitter.process(stream):
        length = sum(len(token.value) for token in statement.tokens)
        cursor += length
        catalog.append(StatementSpan(cursor - length, cursor))
    return catalog


def _merge_error(errors, start, end):
    if errors and errors[-1][1] == start:
        errors[-1] = (errors[-1][0], end)
    else:
        errors.append((start, end))


def _window_openers(text, lo, hi):
    """Delimited-span openers of ``text[lo:hi]`` with absolute offsets.

    Only openers are needed: pairing is decided by the lexer run itself,
    and an opener the lexer could not resolve is the pending span.
    """
    delimited = keywords.find_delimited_spans(text[lo:hi])
    return {pos + lo: (opener_end + lo, tag, ttype)
            for pos, (opener_end, tag, ttype)
            in delimited.openers.items()}


def _is_pending_string(text, start, quote, hi):
    """True when no closing quote exists in ``text[start:hi]``.

    The check is window-local: a closer beyond ``hi`` is invisible to a
    lexer run truncated at ``hi`` and must not be used to prove a
    realignment point inside the window.
    """
    end = _STRING_CONTENT[quote].match(text, start + 1).end()
    return end >= hi or text[end] != quote


def _inside(intervals, pos):
    idx = bisect_right(intervals, pos, key=lambda interval: interval[0])
    return idx > 0 and pos < intervals[idx - 1][1]


def _collect_spans(tokens, text, lo, hi):
    """Collect spans, error intervals and pending state for one window."""
    spans = []
    errors = []
    covered = []

    for ttype, value, start, end in tokens:
        if ttype in T.Error:
            _merge_error(errors, start, end)
        if ttype in (T.String.Single, T.String.Symbol):
            spans.append(LexSpan(ttype, start, end, _STRING))
            covered.append((start, end))
        elif ttype in T.Comment.Multiline:
            spans.append(LexSpan(ttype, start, end, _COMMENT,
                                 keywords._COMMENT_TAG))
            covered.append((start, end))
        elif ttype is T.Literal and value.startswith('$'):
            tag = value[:value.index('$', 1) + 1]
            spans.append(LexSpan(ttype, start, end, _DOLLAR, tag))
            covered.append((start, end))

    classified = sorted(covered)

    # Openers the lexer reached but could not resolve stay open until end of
    # input.  Openers are visited in document order; once a pending span is
    # recorded every later opener is ordinary text inside of it.
    for pos, (_end, tag, ttype) in sorted(
            _window_openers(text, lo, hi).items()):
        if _inside(classified, pos):
            continue
        if any(span.start <= pos < span.end for span in spans):
            continue
        if ttype is T.Literal:
            spans.append(LexSpan(ttype, pos, len(text), _DOLLAR, tag,
                                 pending=True))
        else:
            spans.append(LexSpan(ttype, pos, len(text), _COMMENT,
                                 keywords._COMMENT_TAG, pending=True))

    # An unmatched ordinary quote is an Error token; without a partner it
    # leaves the document inside a string literal.  Each quote kind has at
    # most one pending literal.
    pending_quotes = set()
    for ttype, value, start, _end in tokens:
        if ttype is not T.Error or value not in ("'", '"'):
            continue
        if _inside(classified, start):
            continue
        if any(span.start <= start < span.end for span in spans):
            continue
        if value in pending_quotes:
            continue
        if _is_pending_string(text, start, value, hi):
            pending_quotes.add(value)
            spans.append(LexSpan(
                T.String.Single if value == "'" else T.String.Symbol,
                start, len(text), _STRING, pending=True))

    spans.sort(key=lambda span: span.start)
    return tuple(spans), tuple(errors)


def build_lex_state(text, lexer=None):
    """Build the full :class:`LexState` of ``text`` from scratch.

    Reference implementation used to reconcile incremental results on short
    documents.
    """
    lexer = lexer or Lexer.get_default_instance()
    tokens = _tokenize_window(lexer, text, 0, len(text))
    spans, errors = _collect_spans(tokens, text, 0, len(text))
    return LexState(spans, errors,
                    any(span.pending for span in spans))


def build_statement_catalog(text, lexer=None):
    """Build the full statement directory of ``text`` from scratch."""
    lexer = lexer or Lexer.get_default_instance()
    tokens = _tokenize_window(lexer, text, 0, len(text))
    return _split_catalog(tokens, 0)


def _statement_start(catalog_ends, offset):
    """Start offset of the statement a fresh splitter cannot skip.

    Statement boundaries are not clean cut points: the splitter attaches
    the whitespace and single-line comments directly behind a ``;`` to the
    statement that just ended (its ``consume_ws`` state).  An edit sitting
    exactly on a boundary -- or at end of input, where the trailing
    statement is still open -- therefore rebuilds the statement on its
    left as well.
    """
    idx = bisect_left(catalog_ends, offset)
    if idx == 0:
        return 0
    return catalog_ends[idx - 1]


def _region_start(old_state, old_catalog, start, old_end):
    """Boundaries at which a fresh splitter/lexer run is equivalent."""
    old_ends = [entry.end for entry in old_catalog]
    region_start = _statement_start(old_ends, start)
    for span in old_state.spans:
        if span.start >= old_end:
            continue
        # A pending span reaches end of input and overlaps every later edit.
        if not span.pending and span.end <= start:
            continue
        region_start = min(region_start,
                           _statement_start(old_ends, span.start))
    return region_start


def _find_realignment(new_tokens, old_tokens, new_catalog, new_spans,
                      old_pending_starts, old_boundaries, new_text, old_text,
                      delta, region_start, new_hi, old_hi):
    """Locate the earliest boundary where both lexical views realign.

    A boundary qualifies only when:

    * it is a complete statement end on both sides (the statement
      splitter resets to its root state there);
    * no window-local pending span starts before it: an opener the window
      could not resolve would pair with a delimiter outside the window in a
      full run and might swallow this boundary, so a truncated window can
      never prove it;
    * no *old* pending span crosses it.  A pending span reaches end of
      input, so suffix content behind the boundary was classified as part
      of that span; an edit before the boundary may resolve the opener
      differently and change that classification, making the old suffix
      unsafe to translate;
    * the remaining raw text is identical.  The lexer is stateless apart
      from delimiter positions, which are derived from the text itself, so
      equal text behind equal root states yields identical token streams --
      there is no need, and it would be wrong, to guess a sync point from
      the nearest semicolon.
    """
    old_local_ends = {entry.end for entry in
                      _split_catalog(old_tokens, region_start)}
    for entry in new_catalog:
        candidate = entry.end
        # A window cut inside the next statement cannot be judged.
        if candidate == new_hi and new_hi < len(new_text):
            continue
        mapped = candidate - delta
        if mapped < region_start or mapped > old_hi:
            continue
        if mapped not in old_boundaries or mapped not in old_local_ends:
            continue
        if any(span.pending and span.start < candidate
               for span in new_spans):
            continue
        if any(start < mapped for start in old_pending_starts):
            continue
        window = min(_ALIGN_WINDOW,
                     len(new_text) - candidate, len(old_text) - mapped)
        if new_text[candidate:candidate + window] != \
                old_text[mapped:mapped + window]:
            continue
        return candidate, mapped
    return None, None


def _shift_spans(spans, delta, cutoff):
    return tuple(LexSpan(span.ttype, span.start + delta, span.end + delta,
                         span.kind, span.tag, span.pending)
                 for span in spans if span.start >= cutoff)


def _shift_errors(errors, delta, cutoff):
    return tuple((start + delta, end + delta)
                 for start, end in errors if start >= cutoff)


def incremental_relex(text, old_state, old_catalog, start, old_end,
                      replacement, lexer=None):
    """Re-lex a document after one local replacement.

    :param text: the original document.
    :param old_state: :class:`LexState` produced from ``text``.
    :param old_catalog: statement directory produced from ``text``.
    :param start: inclusive start offset of the replaced region.
    :param old_end: exclusive end offset of the replaced region.
    :param replacement: new text substituting ``text[start:old_end]``.
    :returns: :class:`IncrementalResult`.
    """
    if not 0 <= start <= old_end <= len(text):
        raise ValueError('replacement range outside of document')

    lexer = lexer or Lexer.get_default_instance()
    new_text = text[:start] + replacement + text[old_end:]
    delta = len(replacement) - (old_end - start)

    old_ends = [entry.end for entry in old_catalog]
    changed_from = bisect_right(old_ends, start)

    region_start = _region_start(old_state, old_catalog, start, old_end)
    prefix_catalog = [entry for entry in old_catalog
                      if entry.end <= region_start]
    prefix_spans = tuple(span for span in old_state.spans
                         if span.end <= region_start)
    prefix_errors = tuple((lo, hi) for lo, hi in old_state.errors
                          if hi <= region_start)

    realigned = False
    align_new = align_old = len(new_text)
    window_size = _INITIAL_WINDOW
    new_tokens = old_tokens = []
    new_catalog = []
    while True:
        new_hi = min(len(new_text), region_start + window_size)
        old_hi = min(len(text), max(region_start, new_hi - delta))
        new_tokens = _tokenize_window(lexer, new_text, region_start, new_hi)
        old_tokens = _tokenize_window(lexer, text, region_start, old_hi)
        new_catalog = _split_catalog(new_tokens, region_start)
        new_spans, _ = _collect_spans(new_tokens, new_text,
                                      region_start, new_hi)
        old_pending_starts = [span.start for span in old_state.spans
                              if span.pending]

        align_new, align_old = _find_realignment(
            new_tokens, old_tokens, new_catalog, new_spans,
            old_pending_starts, old_ends, new_text, text, delta,
            region_start, new_hi, old_hi)
        if align_new is not None:
            realigned = True
            break
        if new_hi >= len(new_text) and old_hi >= len(text):
            break
        window_size *= 2

    local_spans, local_errors = _collect_spans(
        new_tokens, new_text, region_start, new_hi)

    if realigned:
        local_catalog = [entry for entry in new_catalog
                         if entry.end <= align_new]
        suffix_catalog = [
            StatementSpan(entry.start + delta, entry.end + delta)
            for entry in old_catalog if entry.start >= align_old]
        shifted_from = next(
            (idx for idx, entry in enumerate(old_catalog)
             if entry.start >= align_old), len(old_catalog))
        catalog = prefix_catalog + local_catalog + suffix_catalog
        spans = (prefix_spans
                 + tuple(span for span in local_spans
                         if span.start < align_new)
                 + _shift_spans(old_state.spans, delta, align_old))
        errors = (prefix_errors
                  + tuple((lo, hi) for lo, hi in local_errors
                          if hi <= align_new)
                  + _shift_errors(old_state.errors, delta, align_old))
        relex_end = align_new
    else:
        local_catalog = new_catalog
        shifted_from = None
        catalog = prefix_catalog + new_catalog
        spans = prefix_spans + local_spans
        errors = prefix_errors + local_errors
        relex_end = len(new_text)

    state = LexState(tuple(sorted(spans, key=lambda span: span.start)),
                     tuple(sorted(errors)),
                     any(span.pending for span in spans))

    # The earliest rebuilt, ingested statement.  When realignment failed
    # this is exactly the first statement a full-document rebuild would
    # have to touch to reconcile the directory.
    if local_catalog:
        earliest_affected = len(prefix_catalog)
    else:
        earliest_affected = None

    had_trouble = any(span.pending and span.start < old_end
                      for span in old_state.spans) or any(
        lo < old_end and hi > start for lo, hi in old_state.errors)
    region_clean = not state.incomplete and not any(
        region_start <= lo < relex_end for lo, _hi in state.errors)
    recovered = had_trouble and region_clean

    return IncrementalResult(
        catalog=catalog,
        state=state,
        changed_from=changed_from,
        earliest_affected=earliest_affected,
        shifted_from=shifted_from,
        realigned=realigned,
        suffix_reused=realigned,
        recovered=recovered,
        relex_end=relex_end,
    )
