import pytest

from tinydb import TinyDB, Query, UniqueConstraintError
from tinydb.storages import MemoryStorage


@pytest.fixture
def table():
    db = TinyDB(storage=MemoryStorage)
    table = db.table('products')
    table.declare_unique_constraint('source', 'number')
    return table


def product(source, number, **extra):
    doc = {'source': source, 'number': number}
    doc.update(extra)
    return doc


def test_declare_unique_constraint(table):
    assert table.unique_constraints == (('source', 'number'),)


def test_declare_unique_constraint_validation(table):
    with pytest.raises(ValueError):
        table.declare_unique_constraint()

    with pytest.raises(ValueError):
        table.declare_unique_constraint('source', 'source')

    with pytest.raises(ValueError):
        table.declare_unique_constraint('source', 'number')

    with pytest.raises(ValueError):
        table.declare_unique_constraint(42)


def test_insert_respects_constraint(table):
    table.insert(product('erp', 'A-1'))

    with pytest.raises(UniqueConstraintError) as excinfo:
        table.insert(product('erp', 'A-1'))

    error = excinfo.value
    assert error.fields == ('source', 'number')
    assert error.key == ('erp', 'A-1')
    assert error.existing_doc_id == 1
    assert error.existing_document == product('erp', 'A-1')
    assert error.batch_position == 0

    # The failed insert must not have been written
    assert len(table) == 1


def test_composite_key_allows_partial_overlap(table):
    table.insert(product('erp', 'A-1'))
    # Same number, different source: fine
    table.insert(product('shop', 'A-1'))
    # Same source, different number: fine
    table.insert(product('erp', 'B-2'))

    assert len(table) == 3


def test_insert_multiple_conflict_with_existing_document(table):
    table.insert(product('erp', 'A-1'))

    with pytest.raises(UniqueConstraintError) as excinfo:
        table.insert_multiple([
            product('erp', 'B-2'),
            product('erp', 'C-3'),
            product('erp', 'A-1'),
        ])

    error = excinfo.value
    assert error.key == ('erp', 'A-1')
    assert error.existing_doc_id == 1
    assert error.existing_document == product('erp', 'A-1')
    assert error.batch_position == 2

    # The whole batch must have been aborted
    assert len(table) == 1
    assert table.all() == [product('erp', 'A-1')]


def test_insert_multiple_conflict_within_batch(table):
    with pytest.raises(UniqueConstraintError) as excinfo:
        table.insert_multiple([
            product('erp', 'A-1'),
            product('erp', 'B-2'),
            product('erp', 'A-1'),
        ])

    error = excinfo.value
    assert error.key == ('erp', 'A-1')
    # The first batch entry is reported as the existing document
    assert error.existing_doc_id == 1
    assert error.batch_position == 2

    assert len(table) == 0


def test_update_respects_constraint(table):
    first_id = table.insert(product('erp', 'A-1'))
    second_id = table.insert(product('erp', 'B-2'))

    with pytest.raises(UniqueConstraintError) as excinfo:
        table.update({'number': 'A-1'}, doc_ids=[second_id])

    error = excinfo.value
    assert error.key == ('erp', 'A-1')
    assert error.existing_doc_id == first_id
    assert error.existing_document == product('erp', 'A-1')
    assert error.offending_doc_id == second_id
    assert error.batch_position == 0

    # Nothing was written
    assert table.get(doc_id=second_id) == product('erp', 'B-2')


def test_update_with_cond_respects_constraint(table):
    table.insert(product('erp', 'A-1'))
    table.insert(product('erp', 'B-2'))

    Product = Query()
    with pytest.raises(UniqueConstraintError):
        table.update({'number': 'A-1'}, Product.number == 'B-2')

    assert table.count(Product.number == 'B-2') == 1


def test_update_multiple_respects_constraint(table):
    table.insert(product('erp', 'A-1'))
    table.insert(product('erp', 'B-2'))

    Product = Query()
    with pytest.raises(UniqueConstraintError):
        table.update_multiple([
            ({'number': 'A-1'}, Product.number == 'B-2'),
        ])

    assert table.count(Product.number == 'B-2') == 1


def test_batch_swap_of_numbers_is_allowed(table):
    first_id = table.insert(product('erp', 'A-1'))
    second_id = table.insert(product('erp', 'B-2'))

    # Swap the numbers of both documents in a single batch. Intermediate
    # states violate the constraint, but the final state is unique.
    swapped = {'A-1': 'B-2', 'B-2': 'A-1'}

    def swap(doc):
        doc['number'] = swapped[doc['number']]

    updated = table.update(swap, doc_ids=[first_id, second_id])

    assert updated == [first_id, second_id]
    assert table.get(doc_id=first_id) == product('erp', 'B-2')
    assert table.get(doc_id=second_id) == product('erp', 'A-1')


def test_upsert_respects_constraint(table):
    table.insert(product('erp', 'A-1'))
    table.insert(product('erp', 'B-2'))

    Product = Query()
    with pytest.raises(UniqueConstraintError):
        table.upsert(product('erp', 'A-1'), Product.number == 'B-2')

    assert len(table) == 2

    # A non-conflicting upsert still works
    table.upsert(product('erp', 'C-3'), Product.number == 'C-3')
    assert len(table) == 3


def test_documents_missing_fields_are_exempt(table):
    table.insert({'name': 'no identity'})
    table.insert({'name': 'no identity either'})
    table.insert(product('erp', 'A-1'))

    assert len(table) == 3


def test_multiple_constraints_on_one_table():
    db = TinyDB(storage=MemoryStorage)
    table = db.table('products')
    table.declare_unique_constraint('source', 'number')
    table.declare_unique_constraint('sku')

    table.insert(product('erp', 'A-1', sku='X'))

    with pytest.raises(UniqueConstraintError) as excinfo:
        table.insert(product('shop', 'A-1', sku='X'))

    assert excinfo.value.fields == ('sku',)
    assert len(table) == 1


def test_remove_and_truncate_are_not_affected(table):
    first_id = table.insert(product('erp', 'A-1'))
    table.insert(product('erp', 'B-2'))

    table.remove(doc_ids=[first_id])
    # The number is free again after the removal
    table.insert(product('erp', 'A-1'))

    table.truncate()
    table.insert(product('erp', 'A-1'))
    table.insert(product('erp', 'B-2'))

    assert len(table) == 2
