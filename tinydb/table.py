"""
This module implements tables, the central place for accessing and manipulating
data in TinyDB.
"""

from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from copy import deepcopy
from typing import (
    NoReturn,
    Optional,
    Union,
    cast,
    overload
)

from .queries import QueryLike
from .storages import Storage
from .utils import LRUCache

__all__ = ('Document', 'Table', 'UniqueConstraintError')


class UniqueConstraintError(ValueError):
    """
    Raised when a write operation would violate a declared unique constraint.

    The error describes the conflict in detail:

    - ``fields``: the field names of the violated constraint,
    - ``key``: the conflicting field values,
    - ``existing_doc_id`` / ``existing_document``: the document that is
      already stored (or the earlier batch entry) holding the key,
    - ``offending_doc_id``: the document ID of the rejected batch entry,
    - ``batch_position``: the zero-based position of the rejected entry
      within the current batch (``None`` if it was not part of a batch).

    When this error is raised, the offending write operation has been
    aborted as a whole: nothing from the batch was written to the storage.
    """

    def __init__(
        self,
        fields: tuple,
        key: tuple,
        existing_doc_id,
        existing_document: Mapping,
        offending_doc_id=None,
        batch_position: Optional[int] = None,
    ):
        self.fields = fields
        self.key = key
        self.existing_doc_id = existing_doc_id
        self.existing_document = existing_document
        self.offending_doc_id = offending_doc_id
        self.batch_position = batch_position

        message = (
            f'Unique constraint on fields {fields!r} violated: '
            f'key {key!r} is already held by document '
            f'{existing_doc_id!r}'
        )
        if batch_position is not None:
            message += f'; rejected batch entry at position {batch_position}'
        if offending_doc_id is not None:
            message += f' (document ID {offending_doc_id!r})'

        super().__init__(message)


class Document(dict):
    """
    A document stored in the database.

    This class provides a way to access both a document's content and
    its ID using ``doc.doc_id``.
    """

    def __init__(self, value: Mapping, doc_id: int):
        super().__init__(value)
        self.doc_id = doc_id


class Table:
    """
    Represents a single TinyDB table.

    It provides methods for accessing and manipulating documents.

    .. admonition:: Query Cache

        As an optimization, a query cache is implemented using a
        :class:`~tinydb.utils.LRUCache`. This class mimics the interface of
        a normal ``dict``, but starts to remove the least-recently used entries
        once a threshold is reached.

        The query cache is updated on every search operation. When writing
        data, the whole cache is discarded as the query results may have
        changed.

    .. admonition:: Customization

        For customization, the following class variables can be set:

        - ``document_class`` defines the class that is used to represent
          documents,
        - ``document_id_class`` defines the class that is used to represent
          document IDs,
        - ``query_cache_class`` defines the class that is used for the query
          cache
        - ``default_query_cache_capacity`` defines the default capacity of
          the query cache

        .. versionadded:: 4.0


    :param storage: The storage instance to use for this table
    :param name: The table name
    :param cache_size: Maximum capacity of query cache
    :param persist_empty: Store new table even with no operations on it
    """

    #: The class used to represent documents
    #:
    #: .. versionadded:: 4.0
    document_class = Document

    #: The class used to represent a document ID
    #:
    #: .. versionadded:: 4.0
    document_id_class = int

    #: The class used for caching query results
    #:
    #: .. versionadded:: 4.0
    query_cache_class = LRUCache

    #: The default capacity of the query cache
    #:
    #: .. versionadded:: 4.0
    default_query_cache_capacity = 10

    def __init__(
        self,
        storage: Storage,
        name: str,
        cache_size: int = default_query_cache_capacity,
        persist_empty: bool = False
    ):
        """
        Create a table instance.
        """

        self._storage = storage
        self._name = name
        self._query_cache: LRUCache[QueryLike, list[Document]] \
            = self.query_cache_class(capacity=cache_size)

        self._unique_constraints: list[tuple[str, ...]] = []

        self._next_id: Optional[int] = None
        # Only persist a brand-new empty table. Never clear existing documents
        # when reopen/construct with persist_empty=True (see issue #636).
        if persist_empty and not self._read_table():
            self._update_table(lambda table: table.clear())

    def __repr__(self):
        args = [
            'name={!r}'.format(self.name),
            'total={}'.format(len(self)),
            'storage={}'.format(self._storage),
        ]

        return '<{} {}>'.format(type(self).__name__, ', '.join(args))

    @property
    def name(self) -> str:
        """
        Get the table name.
        """
        return self._name

    @property
    def storage(self) -> Storage:
        """
        Get the table storage instance.
        """
        return self._storage

    @property
    def unique_constraints(self) -> tuple[tuple[str, ...], ...]:
        """
        Get the unique constraints declared for this table.
        """
        return tuple(self._unique_constraints)

    def declare_unique_constraint(self, *fields: str) -> None:
        """
        Declare a unique constraint on the given field(s).

        Once declared, every write operation (``insert``,
        ``insert_multiple``, ``update``, ``update_multiple`` and
        ``upsert``) makes sure that no two documents share the same
        combination of values for these fields. A constraint may span
        multiple fields to model composite identities, e.g. an external
        identity composed of ``source`` and ``number``::

            table.declare_unique_constraint('source', 'number')

        Constraints are validated against the *final* state of a write
        operation, not after every single document. This means a batch
        that swaps the values of two documents is accepted as long as the
        end result is unique. If the final state violates a constraint,
        a :class:`UniqueConstraintError` is raised and the whole batch is
        aborted: nothing is written to the storage.

        Documents that lack at least one of the constrained fields are
        not subject to the constraint (similar to SQL ``NULL`` handling).

        .. admonition:: Scope of the guarantee

            Enforcement happens in this process, at the moment the table
            data is written. It protects a single, controlled writer --
            it is *not* a multi-process transaction mechanism. If
            multiple processes or uncoordinated writers access the same
            database file, constraints can be bypassed between a read
            and the corresponding write.

        :param fields: one or more field names that must be unique
                       together
        """
        if not fields:
            raise ValueError('A unique constraint needs at least one field')
        if not all(isinstance(field, str) for field in fields):
            raise ValueError('Unique constraint fields must be strings')
        if len(set(fields)) != len(fields):
            raise ValueError(
                f'Duplicate field in unique constraint: {fields!r}'
            )

        constraint = tuple(fields)
        if constraint in self._unique_constraints:
            raise ValueError(
                f'Unique constraint already declared: {constraint!r}'
            )

        self._unique_constraints.append(constraint)

    def _check_unique_constraints(
        self,
        table: dict,
        batch_order: Optional[list] = None,
    ) -> None:
        """
        Validate the final state of a table against all unique constraints.

        ``batch_order`` lists the document IDs touched by the current
        write operation in batch order; it is used to report the batch
        position of a rejected entry. Raises :class:`UniqueConstraintError`
        on the first conflict found.
        """
        if not self._unique_constraints:
            return

        positions: dict = {}
        if batch_order:
            for index, doc_id in enumerate(batch_order):
                positions.setdefault(doc_id, index)

        for fields in self._unique_constraints:
            seen: dict = {}
            for doc_id, doc in table.items():
                if any(field not in doc for field in fields):
                    # Documents missing a constrained field are exempt
                    continue

                key = tuple(doc[field] for field in fields)
                if key not in seen:
                    seen[key] = doc_id
                    continue

                # Conflict between ``first_id`` (seen earlier) and
                # ``doc_id``. Prefer blaming a batch entry so the error
                # can point at the batch position; the other document is
                # reported as the already-existing one.
                first_id = seen[key]
                first_pos = positions.get(first_id)
                second_pos = positions.get(doc_id)

                if first_pos is None and second_pos is None:
                    existing_id, offending_id, batch_pos = \
                        first_id, doc_id, None
                elif first_pos is None:
                    existing_id, offending_id, batch_pos = \
                        first_id, doc_id, second_pos
                elif second_pos is None:
                    existing_id, offending_id, batch_pos = \
                        doc_id, first_id, first_pos
                elif first_pos <= second_pos:
                    existing_id, offending_id, batch_pos = \
                        first_id, doc_id, second_pos
                else:
                    existing_id, offending_id, batch_pos = \
                        doc_id, first_id, first_pos

                raise UniqueConstraintError(
                    fields=fields,
                    key=key,
                    existing_doc_id=existing_id,
                    existing_document=table[existing_id],
                    offending_doc_id=offending_id,
                    batch_position=batch_pos,
                )

    def insert(self, document: Mapping) -> int:
        """
        Insert a new document into the table.

        :param document: the document to insert
        :returns: the inserted document's ID
        """

        # Make sure the document implements the ``Mapping`` interface
        if not isinstance(document, Mapping):
            raise ValueError('Document is not a Mapping')

        # First, we get the document ID for the new document
        if isinstance(document, self.document_class):
            # For a `Document` object we use the specified ID
            doc_id = document.doc_id

            # We also reset the stored next ID so the next insert won't
            # re-use document IDs by accident when storing an old value
            self._next_id = None
        else:
            # In all other cases we use the next free ID
            doc_id = self._get_next_id()

        # Now, we update the table and add the document
        def updater(table: dict):
            if doc_id in table:
                raise ValueError(f'Document with ID {str(doc_id)} '
                                 f'already exists')

            # By calling ``dict(document)`` we convert the data we got to a
            # ``dict`` instance even if it was a different class that
            # implemented the ``Mapping`` interface
            table[doc_id] = dict(document)

        # See below for details on ``Table._update``
        self._update_table(updater, [doc_id])

        return doc_id

    def insert_multiple(self, documents: Iterable[Mapping]) -> list[int]:
        """
        Insert multiple documents into the table.

        :param documents: an Iterable of documents to insert
        :returns: a list containing the inserted documents' IDs
        """
        doc_ids = []

        def updater(table: dict):
            for document in documents:

                # Make sure the document implements the ``Mapping`` interface
                if not isinstance(document, Mapping):
                    raise ValueError('Document is not a Mapping')

                if isinstance(document, self.document_class):
                    # Check if document does not override an existing document
                    if document.doc_id in table:
                        raise ValueError(
                            f'Document with ID {str(document.doc_id)} '
                            f'already exists'
                        )

                    # Store the doc_id, so we can return all document IDs
                    # later. Then save the document with its doc_id and
                    # skip the rest of the current loop
                    doc_id = document.doc_id
                    doc_ids.append(doc_id)
                    table[doc_id] = dict(document)
                    # Same as insert(): don't reuse this ID for later
                    # auto-assigned documents in this batch.
                    self._next_id = None
                    continue

                # Generate new document ID for this document.
                # Allocate from the in-progress table, not storage, so a
                # Document inserted earlier in this batch is visible.
                next_id = self._next_id
                if next_id is None:
                    if table:
                        next_id = max(
                            self.document_id_class(i) for i in table.keys()
                        ) + 1
                    else:
                        next_id = 1
                self._next_id = next_id
                doc_id = self._get_next_id()
                doc_ids.append(doc_id)
                table[doc_id] = dict(document)

        # See below for details on ``Table._update``
        self._update_table(updater, doc_ids)

        return doc_ids

    def all(self) -> list[Document]:
        """
        Get all documents stored in the table.

        :returns: a list with all documents.
        """

        # iter(self) (implemented in Table.__iter__ provides an iterator
        # that returns all documents in this table. We use it to get a list
        # of all documents by using the ``list`` constructor to perform the
        # conversion.

        return list(iter(self))

    def search(self, cond: QueryLike) -> list[Document]:
        """
        Search for all documents matching a 'where' cond.

        Note: a repeated search with the same ``cond`` is served from an
        internal cache. The returned list is a fresh copy, but its
        documents are the same objects held by the cache -- treat them as
        read-only, since mutating one (a key or a nested value) corrupts
        what a later identical search returns.

        :param cond: the condition to check against
        :returns: list of matching documents
        """

        # First, we check the query cache to see if it has results for this
        # query
        cached_results = self._query_cache.get(cond)
        if cached_results is not None:
            return cached_results[:]

        # Perform the search by applying the query to all documents.
        # Then, only if the document matches the query, convert it
        # to the document class and document ID class.
        docs = [
            self.document_class(doc, self.document_id_class(doc_id))
            for doc_id, doc in self._read_table().items()
            if cond(doc)
        ]

        # Only cache cacheable queries.
        #
        # This weird `getattr` dance is needed to make MyPy happy as
        # it doesn't know that a query might have a `is_cacheable` method
        # that is not declared in the `QueryLike` protocol due to it being
        # optional.
        # See: https://github.com/python/mypy/issues/1424
        #
        # Note also that by default we expect custom query objects to be
        # cacheable (which means they need to have a stable hash value).
        # This is to keep consistency with TinyDB's behavior before
        # `is_cacheable` was introduced which assumed that all queries
        # are cacheable.
        is_cacheable: Callable[[], bool] = getattr(cond, 'is_cacheable',
                                                   lambda: True)
        if is_cacheable():
            # Update the query cache
            self._query_cache[cond] = docs[:]

        return docs

    @overload
    def get(self) -> NoReturn: ...

    @overload
    def get(
        self, cond: QueryLike, doc_id: None = ..., doc_ids: None = ...
    ) -> Optional[Document]: ...

    @overload
    def get(
        self, *, cond: QueryLike, doc_id: None = ..., doc_ids: None = ...
    ) -> Optional[Document]: ...

    @overload
    def get(
        self, cond: Optional[QueryLike], doc_id: int, doc_ids: Optional[list] = ...
    ) -> Optional[Document]: ...

    @overload
    def get(
        self, *, cond: Optional[QueryLike] = ..., doc_id: int, doc_ids: Optional[list] = ...,
    ) -> Optional[Document]: ...

    @overload
    def get(
        self, cond: Optional[QueryLike], doc_id: None, doc_ids: list
    ) -> list[Document]: ...

    @overload
    def get(
        self, cond: Optional[QueryLike], *, doc_id: None = ..., doc_ids: list
    ) -> list[Document]: ...

    @overload
    def get(
        self, *, cond: Optional[QueryLike] = ..., doc_id: None = ..., doc_ids: list
    ) -> list[Document]: ...

    def get(
        self,
        cond: Optional[QueryLike] = None,
        doc_id: Optional[int] = None,
        doc_ids: Optional[list] = None
    ):
        """
        Get document(s) specified by a query or document ID(s).

        - With ``cond`` or ``doc_id``: returns a single document, or ``None``
          if no match exists. If multiple documents match ``cond``, the first
          match in insertion order is returned.
        - With ``doc_ids``: returns a list of matching documents (possibly
          empty).

        :param cond: the condition to check against
        :param doc_id: the document's ID
        :param doc_ids: the document's IDs (multiple)

        :returns: a document, ``None``, or a list of documents when
                  ``doc_ids`` is used
        """
        table = self._read_table()

        if doc_id is not None:
            # Retrieve a document specified by its ID
            raw_doc = table.get(str(doc_id), None)

            if raw_doc is None:
                return None

            # Convert the raw data to the document class
            return self.document_class(raw_doc, doc_id)

        elif doc_ids is not None:
            # Filter the table by extracting out all those documents which
            # have doc id specified in the doc_id list.

            # Since document IDs will be unique, we make it a set to ensure
            # constant time lookup
            doc_ids_set = set(str(doc_id) for doc_id in doc_ids)

            # Now return the filtered documents in form of list
            return [
                self.document_class(doc, self.document_id_class(doc_id))
                for doc_id, doc in table.items()
                if doc_id in doc_ids_set
            ]

        elif cond is not None:
            # Find a document specified by a query
            # The trailing underscore in doc_id_ is needed so MyPy
            # doesn't think that `doc_id_` (which is a string) needs
            # to have the same type as `doc_id` which is this function's
            # parameter and is an optional `int`.
            for doc_id_, doc in self._read_table().items():
                if cond(doc):
                    return self.document_class(
                        doc,
                        self.document_id_class(doc_id_)
                    )

            return None

        raise RuntimeError('You have to pass either cond or doc_id or doc_ids')

    def contains(
        self,
        cond: Optional[QueryLike] = None,
        doc_id: Optional[int] = None
    ) -> bool:
        """
        Check whether the database contains a document matching a query or
        an ID.

        If ``doc_id`` is set, it checks if the db contains the specified ID.

        :param cond: the condition use
        :param doc_id: the document ID to look for
        """
        if doc_id is not None:
            # Documents specified by ID
            return self.get(doc_id=doc_id) is not None

        elif cond is not None:
            # Document specified by condition
            return self.get(cond) is not None

        raise RuntimeError('You have to pass either cond or doc_id')

    def update(
        self,
        fields: Union[Mapping, Callable[[MutableMapping], None]],
        cond: Optional[QueryLike] = None,
        doc_ids: Optional[Iterable[int]] = None,
    ) -> list[int]:
        """
        Update all matching documents to have a given set of fields.

        When ``doc_ids`` is given, IDs that don't refer to an existing
        document are silently skipped, and the returned list only contains
        the IDs that were actually updated.

        :param fields: the fields that the matching documents will have
                       or a callable that mutates matching documents in place
        :param cond: which documents to update
        :param doc_ids: a list of document IDs
        :returns: a list containing the updated document's ID
        """

        # Define the function that will perform the update
        if callable(fields):
            def perform_update(table, doc_id):
                # Update documents by calling the update function provided by
                # the user
                fields(table[doc_id])
        else:
            def perform_update(table, doc_id):
                # Update documents by setting all fields from the provided data
                table[doc_id].update(fields)

        if doc_ids is not None:
            # Perform the update operation for documents specified by a list
            # of document IDs. Document IDs that don't exist in the table are
            # silently skipped, mirroring the behaviour of ``get(doc_ids=...)``
            # (see issue #591). The list of *actually* updated IDs is
            # determined inside the updater so it reflects the table state at
            # write time.
            requested_ids = list(doc_ids)
            updated_ids: list[int] = []

            def updater(table: dict):
                # Filter to IDs that exist *before* performing any updates,
                # so the operation is atomic: either every existing target is
                # updated, or none is.
                updated_ids.extend(
                    doc_id for doc_id in requested_ids if doc_id in table
                )
                for doc_id in updated_ids:
                    perform_update(table, doc_id)

            # Perform the update operation (see _update_table for details)
            self._update_table(updater, updated_ids)

            return updated_ids

        elif cond is not None:
            # Perform the update operation for documents specified by a query

            # Collect affected doc_ids
            updated_ids = []

            def updater(table: dict):
                _cond = cast(QueryLike, cond)

                # We need to convert the keys iterator to a list because
                # we may remove entries from the ``table`` dict during
                # iteration and doing this without the list conversion would
                # result in an exception (RuntimeError: dictionary changed size
                # during iteration)
                for doc_id in list(table.keys()):
                    # Pass through all documents to find documents matching the
                    # query. Call the processing callback with the document ID
                    if _cond(table[doc_id]):
                        # Add ID to list of updated documents
                        updated_ids.append(doc_id)

                        # Perform the update (see above)
                        perform_update(table, doc_id)

            # Perform the update operation (see _update_table for details)
            self._update_table(updater, updated_ids)

            return updated_ids

        else:
            # Update all documents unconditionally

            updated_ids = []

            def updater(table: dict):
                # Process all documents
                for doc_id in list(table.keys()):
                    # Add ID to list of updated documents
                    updated_ids.append(doc_id)

                    # Perform the update (see above)
                    perform_update(table, doc_id)

            # Perform the update operation (see _update_table for details)
            self._update_table(updater, updated_ids)

            return updated_ids

    def update_multiple(
        self,
        updates: Iterable[
            tuple[Union[Mapping, Callable[[MutableMapping], None]], QueryLike]
        ],
    ) -> list[int]:
        """
        Update all matching documents to have a given set of fields.

        :returns: a list containing the updated document's ID
        """

        # Define the function that will perform the update
        def perform_update(fields, table, doc_id):
            if callable(fields):
                # Update documents by calling the update function provided
                # by the user
                fields(table[doc_id])
            else:
                # Update documents by setting all fields from the provided
                # data
                table[doc_id].update(fields)

        # Perform the update operation for documents specified by a query

        # Collect affected doc_ids
        updated_ids = []

        def updater(table: dict):
            # We need to convert the keys iterator to a list because
            # we may remove entries from the ``table`` dict during
            # iteration and doing this without the list conversion would
            # result in an exception (RuntimeError: dictionary changed size
            # during iteration)
            for doc_id in list(table.keys()):
                for fields, cond in updates:
                    _cond = cast(QueryLike, cond)

                    # Pass through all documents to find documents matching the
                    # query. Call the processing callback with the document ID
                    if _cond(table[doc_id]):
                        # Add ID to list of updated documents
                        updated_ids.append(doc_id)

                        # Perform the update (see above)
                        perform_update(fields, table, doc_id)

        # Perform the update operation (see _update_table for details)
        self._update_table(updater, updated_ids)

        return updated_ids

    def upsert(self, document: Mapping, cond: Optional[QueryLike] = None) -> list[int]:
        """
        Update documents, if they exist, insert them otherwise.

        Note: This will update *all* documents matching the query.
        For example, if the query matches 3 documents, all 3 will be updated
        with the new data. If no documents match, a new one is inserted.

        Document argument can be a tinydb.table.Document object if you want
        to specify a doc_id.

        :param document: the document to insert or the fields to update
        :param cond: which document to look for, optional if you've passed a
        Document with a doc_id
        :returns: a list containing the updated documents' IDs
        """

        # Extract doc_id
        if isinstance(document, self.document_class) and hasattr(document, 'doc_id'):
            doc_ids: Optional[list[int]] = [document.doc_id]
        else:
            doc_ids = None

        # Make sure we can actually find a matching document
        if doc_ids is None and cond is None:
            raise ValueError("If you don't specify a search query, you must "
                             "specify a doc_id. Hint: use a table.Document "
                             "object.")

        # Perform the update operation. ``update`` returns an empty list
        # when no existing document matches the doc_id / query, in which
        # case we fall through to insert below.
        updated_docs = self.update(document, cond, doc_ids)

        # If documents have been updated: return their IDs
        if updated_docs:
            return updated_docs

        # There are no documents that match the specified query -> insert the
        # data as a new document
        return [self.insert(document)]

    def remove(
        self,
        cond: Optional[QueryLike] = None,
        doc_ids: Optional[Iterable[int]] = None,
    ) -> list[int]:
        """
        Remove all matching documents.

        When ``doc_ids`` is given, IDs that don't refer to an existing
        document are silently skipped, and the returned list only contains
        the IDs that were actually removed.

        :param cond: the condition to check against
        :param doc_ids: a list of document IDs
        :returns: a list containing the removed documents' ID
        """
        if doc_ids is not None:
            # This function returns the list of IDs for the documents that
            # have been removed. Document IDs that don't exist in the table
            # are silently skipped, mirroring the behaviour of
            # ``get(doc_ids=...)`` (see issue #591). The list of *actually*
            # removed IDs is determined inside the updater so it reflects the
            # table state at write time.
            requested_ids = list(doc_ids)
            removed_ids: list[int] = []

            def updater(table: dict):
                # Filter to IDs that exist *before* performing any removals,
                # so the operation is atomic: either every existing target is
                # removed, or none is.
                removed_ids.extend(
                    doc_id for doc_id in requested_ids if doc_id in table
                )
                for doc_id in removed_ids:
                    del table[doc_id]

            # Perform the remove operation
            self._update_table(updater)

            return removed_ids

        if cond is not None:
            removed_ids = []

            # This updater function will be called with the table data
            # as its first argument. See ``Table._update`` for details on this
            # operation
            def updater(table: dict):
                # We need to convince MyPy (the static type checker) that
                # the ``cond is not None`` invariant still holds true when
                # the updater function is called
                _cond = cast(QueryLike, cond)

                # We need to convert the keys iterator to a list because
                # we may remove entries from the ``table`` dict during
                # iteration and doing this without the list conversion would
                # result in an exception (RuntimeError: dictionary changed size
                # during iteration)
                for doc_id in list(table.keys()):
                    if _cond(table[doc_id]):
                        # Add document ID to list of removed document IDs
                        removed_ids.append(doc_id)

                        # Remove document from the table
                        table.pop(doc_id)

            # Perform the remove operation
            self._update_table(updater)

            return removed_ids

        raise RuntimeError('Use truncate() to remove all documents')

    def truncate(self) -> None:
        """
        Truncate the table by removing all documents.
        """

        # Update the table by resetting all data
        self._update_table(lambda table: table.clear())

        # Reset document ID counter
        self._next_id = None

    def count(self, cond: QueryLike) -> int:
        """
        Count the documents matching a query.

        :param cond: the condition use
        """

        return len(self.search(cond))

    def clear_cache(self) -> None:
        """
        Clear the query cache.
        """

        self._query_cache.clear()

    def __len__(self):
        """
        Count the total number of documents in this table.
        """

        return len(self._read_table())

    def __iter__(self) -> Iterator[Document]:
        """
        Iterate over all documents stored in the table.

        :returns: an iterator over all documents.
        """

        # Iterate all documents and their IDs
        for doc_id, doc in self._read_table().items():
            # Convert documents to the document class
            yield self.document_class(doc, self.document_id_class(doc_id))

    def _get_next_id(self):
        """
        Return the ID for a newly inserted document.
        """

        # If we already know the next ID
        if self._next_id is not None:
            next_id = self._next_id
            self._next_id = next_id + 1

            return next_id

        # Determine the next document ID by finding out the max ID value
        # of the current table documents

        # Read the table documents
        table = self._read_table()

        # If the table is empty, set the initial ID
        if not table:
            next_id = 1
            self._next_id = next_id + 1

            return next_id

        # Determine the next ID based on the maximum ID that's currently in use
        max_id = max(self.document_id_class(i) for i in table.keys())
        next_id = max_id + 1

        # The next ID we will return AFTER this call needs to be larger than
        # the current next ID we calculated
        self._next_id = next_id + 1

        return next_id

    def _read_table(self) -> dict[str, Mapping]:
        """
        Read the table data from the underlying storage.

        Documents and doc_ids are NOT yet transformed, as
        we may not want to convert *all* documents when returning
        only one document for example.
        """

        # Retrieve the tables from the storage
        tables = self._storage.read()

        if tables is None:
            # The database is empty
            return {}

        # Retrieve the current table's data
        try:
            table = tables[self.name]
        except KeyError:
            # The table does not exist yet, so it is empty
            return {}

        return table

    def _update_table(
        self,
        updater: Callable[[dict[int, Mapping]], None],
        batch_order: Optional[list] = None,
    ):
        """
        Perform a table update operation.

        The storage interface used by TinyDB only allows to read/write the
        complete database data, but not modifying only portions of it. Thus,
        to only update portions of the table data, we first perform a read
        operation, perform the update on the table data and then write
        the updated data back to the storage.

        As a further optimization, we don't convert the documents into the
        document class, as the table data will *not* be returned to the user.

        ``batch_order`` lists the document IDs touched by this operation in
        batch order. Before anything is written back, the final table state
        is validated against the declared unique constraints; a violation
        aborts the whole operation and nothing is written to the storage.
        """

        tables = self._storage.read()

        if tables is None:
            # The database is empty
            tables = {}

        try:
            raw_table = tables[self.name]
        except KeyError:
            # The table does not exist yet, so it is empty
            raw_table = {}

        # Convert the document IDs to the document ID class.
        # This is required as the rest of TinyDB expects the document IDs
        # to be an instance of ``self.document_id_class`` but the storage
        # might convert dict keys to strings.
        if self._unique_constraints:
            # Work on a deep copy of the documents: updaters may mutate
            # documents in place, and if the unique constraint validation
            # below fails, the storage must remain untouched so the whole
            # batch is really aborted (some storages, e.g. MemoryStorage,
            # hand out their internal data structures on read).
            table = {
                self.document_id_class(doc_id): deepcopy(doc)
                for doc_id, doc in raw_table.items()
            }
        else:
            table = {
                self.document_id_class(doc_id): doc
                for doc_id, doc in raw_table.items()
            }

        # Perform the table update operation
        updater(table)

        # Validate the *final* table state against the declared unique
        # constraints. Checking only the end state (instead of every
        # intermediate step) allows batches that temporarily violate a
        # constraint, e.g. when two documents swap values. If validation
        # fails, an exception is raised before anything is written back,
        # so the whole batch is aborted atomically.
        self._check_unique_constraints(table, batch_order)

        # Convert the document IDs back to strings.
        # This is required as some storages (most notably the JSON file format)
        # don't support IDs other than strings.
        tables[self.name] = {
            str(doc_id): doc
            for doc_id, doc in table.items()
        }

        # Write the newly updated data back to the storage
        self._storage.write(tables)

        # Clear the query cache, as the table contents have changed
        self.clear_cache()
