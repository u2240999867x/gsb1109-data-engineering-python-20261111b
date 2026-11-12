"""
Isolated document drafts with three-way merge support.

This module implements *document drafts*: an editing session works on an
isolated deep copy of a stored document and, on save, only the fields the
editor actually touched are committed back.  Changes are merged three-way
(baseline / currently stored / draft) so two editors starting from the same
baseline can both save as long as they did not touch the same data.

Conflicting situations are reported via :class:`MergeConflict`, for example:

- both sides modified the same field with different values,
- one side removed a parent object while the other side modified a nested
  field below it,
- a nested list changed on both sides and cannot be aligned safely because
  it has no stable business key (see ``list_keys``) and contains duplicate
  values.

Drafts never touch the table, its query cache or the underlying storage
until :meth:`DocumentDraft.save` succeeds, so unsaved edits cannot pollute
query results or stored data.
"""

from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from .table import Table

__all__ = ('Conflict', 'MergeConflict', 'DocumentDraft')


#: Sentinel for "no value present" (distinct from ``None``)
_MISSING: Any = object()

#: Sentinel for list items that cannot be aligned (missing business key)
_UNALIGNABLE: Any = object()

#: A change recorded by the diff: ``('set', value)`` or ``('delete',)``
_Change = tuple

#: A path to a value inside a document, e.g. ``('specs', 'weight')``
_Path = tuple


class Conflict:
    """
    Describes a single merge conflict.

    ``path`` locates the conflicting value inside the document (as a tuple
    of keys, e.g. ``('specs', 'weight')``) and ``reason`` explains why the
    merge could not be performed automatically.
    """

    def __init__(self, path: _Path, reason: str):
        self.path = tuple(path)
        self.reason = reason

    def __repr__(self):
        return 'Conflict(path={!r}, reason={!r})'.format(self.path, self.reason)

    def __eq__(self, other):
        if not isinstance(other, Conflict):
            return NotImplemented
        return self.path == other.path and self.reason == other.reason

    def __hash__(self):
        return hash((self.path, self.reason))


class MergeConflict(Exception):
    """
    Raised when a draft cannot be saved because its changes conflict with
    changes that were stored after the draft's baseline was taken.

    The individual conflicts are available via :attr:`conflicts`.  When this
    exception is raised, nothing has been written: the stored document, the
    query cache and the ``doc_id`` mapping are all left untouched.
    """

    def __init__(self, doc_id: int, conflicts: list[Conflict]):
        self.doc_id = doc_id
        self.conflicts = list(conflicts)
        details = ', '.join(
            '{} ({})'.format('.'.join(map(str, c.path)) or '<document>',
                             c.reason)
            for c in self.conflicts
        )
        super().__init__(
            'Cannot save draft for document {}: {}'.format(doc_id, details)
        )


def _diff(base: Any, new: Any, path: _Path = ()) -> dict[_Path, _Change]:
    """
    Compute the changes that turn ``base`` into ``new``.

    Returns a dict mapping paths to ``('set', value)`` / ``('delete',)``
    changes.  Dictionaries are compared key by key; every other value
    (including lists) is compared as a whole.
    """
    changes: dict[_Path, _Change] = {}

    if isinstance(base, Mapping) and isinstance(new, Mapping):
        for key in base:
            if key not in new:
                changes[path + (key,)] = ('delete',)
        for key in new:
            if key not in base:
                changes[path + (key,)] = ('set', deepcopy(new[key]))
            else:
                changes.update(_diff(base[key], new[key], path + (key,)))
    elif base != new:
        changes[path] = ('set', deepcopy(new))

    return changes


def _get_path(doc: Any, path: _Path) -> Any:
    """
    Return the value at ``path`` in ``doc`` or ``_MISSING``.
    """
    node = doc
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return _MISSING
        node = node[key]
    return node


def _apply_change(doc: dict, path: _Path, change: _Change) -> None:
    """
    Apply a single diff change to ``doc`` in place.
    """
    parent = doc
    for key in path[:-1]:
        parent = parent[key]
    if change[0] == 'set':
        parent[path[-1]] = deepcopy(change[1])
    else:
        parent.pop(path[-1], None)


def _is_prefix(prefix: _Path, path: _Path) -> bool:
    """
    Check whether ``prefix`` is a proper ancestor path of ``path``.
    """
    return len(prefix) < len(path) and path[:len(prefix)] == prefix


def _list_key(list_keys: Mapping[str, str], path: _Path) -> Optional[str]:
    """
    Look up the business key configured for the list at ``path``.
    """
    return list_keys.get('.'.join(map(str, path)))


def _find(identities: list, identity: Any) -> Optional[int]:
    """
    Return the index of ``identity`` in ``identities`` or ``None``.

    Uses equality comparison so unhashable identities (e.g. dicts when no
    business key is configured) work as well.
    """
    for index, candidate in enumerate(identities):
        if candidate == identity:
            return index
    return None


def _has_duplicates(identities: list) -> bool:
    for index, candidate in enumerate(identities):
        if _find(identities[index + 1:], candidate) is not None:
            return True
    return False


def _merge_changed(
    path: _Path,
    base_val: Any,
    current_val: Any,
    draft_val: Any,
    list_keys: Mapping[str, str],
) -> tuple[Any, list[Conflict]]:
    """
    Three-way merge of a single value that both sides changed.
    """
    if base_val is not _MISSING:
        if current_val == base_val:
            return deepcopy(draft_val), []
        if draft_val == base_val:
            return deepcopy(current_val), []
    if current_val == draft_val:
        return deepcopy(current_val), []
    if isinstance(current_val, Mapping) and isinstance(draft_val, Mapping):
        base = base_val if isinstance(base_val, Mapping) else {}
        return _merge_documents(base, current_val, draft_val, list_keys, path)
    if isinstance(current_val, list) and isinstance(draft_val, list):
        base = base_val if isinstance(base_val, list) else []
        return _merge_lists(path, base, current_val, draft_val,
                            _list_key(list_keys, path))
    return None, [Conflict(path, 'both_modified')]


def _pair_items(
    base_ids: list,
    side_ids: list,
    positional: bool,
) -> tuple[dict[int, int], set[int]]:
    """
    Pair baseline items with items of one side of the merge.

    Returns a mapping of baseline indices to side indices plus the set of
    side indices that were paired.  Items are paired by identity first;
    without a business key, still-unpaired items at the same position are
    paired as well so in-place edits of scalar values are recognized.
    """
    pairs: dict[int, int] = {}
    used: set[int] = set()
    for index, ident in enumerate(base_ids):
        for side_index, candidate in enumerate(side_ids):
            if side_index not in used and candidate == ident:
                pairs[index] = side_index
                used.add(side_index)
                break
    if positional:
        for index in range(len(base_ids)):
            if index not in pairs and index < len(side_ids) \
                    and index not in used:
                pairs[index] = index
                used.add(index)
    return pairs, used


def _merge_lists(
    path: _Path,
    base: list,
    current: list,
    draft: list,
    key: Optional[str],
) -> tuple[Any, list[Conflict]]:
    """
    Three-way merge of a nested list that both sides changed.

    Items are aligned by the configured business ``key`` or, without one, by
    their value (falling back to their position).  Alignment is refused
    (reported as a conflict) when items lack the key or when duplicate
    identities make the alignment ambiguous.
    """
    def identity(item: Any) -> Any:
        if key is None:
            return item
        if isinstance(item, Mapping) and key in item:
            return item[key]
        return _UNALIGNABLE

    base_ids = [identity(item) for item in base]
    current_ids = [identity(item) for item in current]
    draft_ids = [identity(item) for item in draft]

    if any(ident is _UNALIGNABLE
           for ident in base_ids + current_ids + draft_ids):
        return None, [Conflict(path, 'items_missing_key')]

    if (_has_duplicates(base_ids) or _has_duplicates(current_ids)
            or _has_duplicates(draft_ids)):
        # Without unique identities we cannot tell which item an edit or a
        # removal refers to, so we refuse to guess.
        return None, [Conflict(path, 'ambiguous_list_alignment')]

    positional = key is None
    current_pairs, current_used = _pair_items(base_ids, current_ids, positional)
    draft_pairs, draft_used = _pair_items(base_ids, draft_ids, positional)

    conflicts: list[Conflict] = []
    surviving: dict[int, Any] = {}

    for index in range(len(base)):
        current_index = current_pairs.get(index)
        draft_index = draft_pairs.get(index)

        if current_index is None and draft_index is None:
            # Removed on both sides
            continue
        if current_index is None:
            if draft[draft_index] != base[index]:
                conflicts.append(Conflict(path, 'removed_vs_modified'))
            continue
        if draft_index is None:
            if current[current_index] != base[index]:
                conflicts.append(Conflict(path, 'removed_vs_modified'))
            continue

        item, item_conflicts = _merge_changed(
            path, base[index], current[current_index], draft[draft_index], {}
        )
        conflicts.extend(item_conflicts)
        surviving[index] = item

    # Additions: items that have no counterpart in the baseline
    additions: list[Any] = []
    addition_ids: list[Any] = []
    for i in range(len(current)):
        if i not in current_used:
            additions.append(current[i])
            addition_ids.append(current_ids[i])
    for i in range(len(draft)):
        if i in draft_used:
            continue
        match = _find(addition_ids, draft_ids[i])
        if match is None:
            additions.append(draft[i])
            addition_ids.append(draft_ids[i])
        elif draft[i] == additions[match]:
            # Identical additions collapse into one copy
            continue
        elif isinstance(additions[match], Mapping) \
                and isinstance(draft[i], Mapping):
            item, item_conflicts = _merge_documents(
                {}, additions[match], draft[i], {}, path
            )
            conflicts.extend(item_conflicts)
            additions[match] = item
        else:
            conflicts.append(Conflict(path, 'both_added'))

    if conflicts:
        return None, conflicts

    # Determine the order of the surviving items: if exactly one side
    # reordered them, that side wins; if both did, the orders must agree.
    base_order = [i for i in range(len(base)) if i in surviving]
    current_order = sorted(base_order, key=lambda i: current_pairs[i])
    draft_order = sorted(base_order, key=lambda i: draft_pairs[i])

    if current_order == base_order:
        order = draft_order
    elif draft_order == base_order or current_order == draft_order:
        order = current_order
    else:
        return None, [Conflict(path, 'conflicting_order')]

    merged = [surviving[i] for i in order]
    merged.extend(additions)

    return merged, []


def _merge_documents(
    base: Mapping,
    current: Mapping,
    draft: Mapping,
    list_keys: Mapping[str, str],
    prefix: _Path = (),
) -> tuple[dict, list[Conflict]]:
    """
    Three-way merge of two documents that both derive from ``base``.

    Only the changes present in ``draft`` are applied on top of ``current``;
    paths that both sides changed are merged value by value and reported as
    conflicts when they cannot be reconciled.
    """
    draft_diff = _diff(base, draft)
    other_diff = _diff(base, current)

    merged = deepcopy(dict(current))
    conflicts: list[Conflict] = []
    other_paths = list(other_diff.keys())

    for path, change in draft_diff.items():
        full_path = prefix + path

        if path in other_diff:
            other_change = other_diff[path]
            if other_change == change:
                # Both sides made the very same change
                continue
            if change[0] == 'set' and other_change[0] == 'set':
                value, sub_conflicts = _merge_changed(
                    full_path, _get_path(base, path),
                    other_change[1], change[1], list_keys
                )
                conflicts.extend(sub_conflicts)
                if not sub_conflicts:
                    _apply_change(merged, path, ('set', value))
            else:
                conflicts.append(Conflict(full_path, 'both_modified'))
            continue

        descendants = [p for p in other_paths if _is_prefix(path, p)]
        if descendants:
            # The draft replaced/removed a subtree that the other side
            # changed below.  Pure deletions on both sides are compatible.
            if change[0] == 'delete' and all(
                    other_diff[p][0] == 'delete' for p in descendants):
                _apply_change(merged, path, change)
            else:
                conflicts.append(Conflict(full_path, 'modified_descendant'))
            continue

        ancestors = [p for p in other_paths if _is_prefix(p, path)]
        if ancestors:
            ancestor = min(ancestors, key=len)
            if change[0] == 'delete' and other_diff[ancestor][0] == 'delete':
                # The ancestor is already gone; nothing left to do
                continue
            conflicts.append(Conflict(prefix + ancestor, 'modified_ancestor'))
            continue

        _apply_change(merged, path, change)

    return merged, conflicts


class DocumentDraft:
    """
    An isolated, editable copy of a single stored document.

    The draft starts from a *baseline* (a deep copy of the document at
    creation time).  All edits happen on :attr:`document`, which is fully
    detached from the table: neither the query cache nor the storage observe
    draft changes.  On :meth:`save`, only the fields that actually differ
    from the baseline are committed, three-way merged against the currently
    stored document.

    ``list_keys`` maps dotted paths of nested lists to the name of a stable
    business key (e.g. ``{'items': 'sku'}``).  Keyed lists can be aligned
    item by item even after reordering; lists without a key are aligned by
    value and refuse to merge when duplicate values make alignment
    ambiguous.

    :param table: the table the document lives in
    :param doc_id: the ID of the document to edit
    :param list_keys: mapping of dotted list paths to business key fields
    """

    def __init__(
        self,
        table: 'Table',
        doc_id: int,
        list_keys: Optional[Mapping[str, str]] = None
    ):
        stored = table.get(doc_id=doc_id)
        if stored is None:
            raise ValueError(f'Document with ID {str(doc_id)} does not exist')

        self._table = table
        self._doc_id = doc_id
        self._list_keys = dict(list_keys) if list_keys else {}
        self._baseline = deepcopy(dict(stored))
        self._working = deepcopy(self._baseline)
        self._active = True

    def __repr__(self):
        return '<{} doc_id={!r}, active={!r}>'.format(
            type(self).__name__, self.doc_id, self.is_active
        )

    @property
    def doc_id(self) -> int:
        """
        The ID of the document this draft edits.
        """
        return self._doc_id

    @property
    def is_active(self) -> bool:
        """
        Whether the draft can still be saved (``False`` after ``discard``).
        """
        return self._active

    @property
    def baseline(self) -> dict:
        """
        A copy of the document as it was when the draft was created.
        """
        return deepcopy(self._baseline)

    @property
    def document(self) -> dict:
        """
        The editable working copy of the document.
        """
        return self._working

    def changes(self) -> dict[_Path, _Change]:
        """
        The changes made so far, as a mapping of paths to changes.

        Only these paths will be committed by :meth:`save`.
        """
        return _diff(self._baseline, self._working)

    def save(self) -> int:
        """
        Commit the draft's changes to the table.

        The changes are three-way merged against the currently stored
        document, so concurrent edits to other fields are preserved.  On
        success the draft is rebased onto the merged document and can keep
        being edited.  If the changes conflict with concurrently stored
        edits, :class:`MergeConflict` is raised and nothing is written.

        :returns: the document's ID
        """
        if not self._active:
            raise RuntimeError('Cannot save a draft that has been discarded')

        if self._working == self._baseline:
            # Nothing to commit
            return self._doc_id

        stored = self._table.get(doc_id=self._doc_id)
        if stored is None:
            raise MergeConflict(
                self._doc_id, [Conflict((), 'document_deleted')]
            )

        merged, conflicts = _merge_documents(
            self._baseline, dict(stored), self._working, self._list_keys
        )
        if conflicts:
            raise MergeConflict(self._doc_id, conflicts)

        doc_id = self._doc_id

        def updater(table: dict):
            if doc_id not in table:
                raise MergeConflict(
                    doc_id, [Conflict((), 'document_deleted')]
                )
            table[doc_id] = merged

        self._table._update_table(updater)

        # Rebase so editing can continue from the merged state
        self._baseline = deepcopy(merged)
        self._working = deepcopy(merged)

        return self._doc_id

    def discard(self) -> None:
        """
        Discard the draft, dropping all uncommitted changes.

        The stored document is left untouched; afterwards the draft can no
        longer be saved.
        """
        self._working = deepcopy(self._baseline)
        self._active = False
