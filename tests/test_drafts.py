import pytest

from tinydb import TinyDB, where
from tinydb.drafts import DocumentDraft, MergeConflict
from tinydb.storages import MemoryStorage


@pytest.fixture
def table():
    db = TinyDB(storage=MemoryStorage)
    db.drop_tables()
    return db.table('products')


@pytest.fixture
def product(table):
    doc_id = table.insert({
        'name': 'Coffee',
        'price': 10,
        'specs': {'weight': 250, 'origin': 'BR'},
        'items': [
            {'sku': 'a', 'qty': 1},
            {'sku': 'b', 'qty': 2},
        ],
        'tags': ['x', 'y'],
    })
    return doc_id


def test_draft_isolated_from_cache_and_storage(table, product):
    cond = where('name') == 'Coffee'
    cached = table.search(cond)
    raw_before = table.storage.read()

    draft = table.draft(product)
    draft.document['price'] = 999
    draft.document['specs']['weight'] = 999

    # The query cache still serves the pre-draft results
    assert table.search(cond) == cached
    assert table.search(cond)[0]['price'] == 10
    # The cached Document objects were not mutated through the draft
    assert cached[0]['specs']['weight'] == 250
    # The raw storage is untouched
    assert table.storage.read() == raw_before
    assert table.get(doc_id=product)['price'] == 10


def test_save_commits_only_touched_columns(table, product):
    draft = table.draft(product)
    draft.document['price'] = 12

    assert set(draft.changes()) == {('price',)}
    assert draft.save() == product

    stored = table.get(doc_id=product)
    assert stored['price'] == 12
    assert stored['name'] == 'Coffee'
    assert stored['specs'] == {'weight': 250, 'origin': 'BR'}


def test_parallel_saves_merge_different_columns(table, product):
    first = table.draft(product)
    second = table.draft(product)

    first.document['price'] = 12
    second.document['name'] = 'Espresso'

    assert first.save() == product
    assert second.save() == product

    stored = table.get(doc_id=product)
    assert stored['price'] == 12
    assert stored['name'] == 'Espresso'


def test_same_column_conflict_keeps_storage_and_doc_id(table, product):
    first = table.draft(product)
    second = table.draft(product)

    first.document['price'] = 12
    second.document['price'] = 15

    first.save()
    with pytest.raises(MergeConflict) as excinfo:
        second.save()

    assert excinfo.value.doc_id == product
    assert excinfo.value.conflicts[0].path == ('price',)

    # The conflicted save wrote nothing; the original doc_id still resolves
    assert table.get(doc_id=product)['price'] == 12
    assert table.contains(doc_id=product)


def test_identical_change_on_both_sides_merges(table, product):
    first = table.draft(product)
    second = table.draft(product)

    first.document['price'] = 12
    second.document['price'] = 12

    first.save()
    second.save()
    assert table.get(doc_id=product)['price'] == 12


def test_delete_parent_vs_child_edit_conflict(table, product):
    deleter = table.draft(product)
    editor = table.draft(product)

    del deleter.document['specs']
    editor.document['specs']['weight'] = 300

    deleter.save()
    with pytest.raises(MergeConflict) as excinfo:
        editor.save()

    assert excinfo.value.conflicts[0].path == ('specs',)
    # Nothing was written by the conflicted save
    assert 'specs' not in table.get(doc_id=product)


def test_child_edit_vs_delete_parent_conflict(table, product):
    editor = table.draft(product)
    deleter = table.draft(product)

    editor.document['specs']['weight'] = 300
    del deleter.document['specs']

    editor.save()
    with pytest.raises(MergeConflict):
        deleter.save()

    assert table.get(doc_id=product)['specs']['weight'] == 300


def test_keyed_list_aligns_after_reorder(table, product):
    reordered = table.draft(product, list_keys={'items': 'sku'})
    editor = table.draft(product, list_keys={'items': 'sku'})

    reordered.document['items'] = [
        {'sku': 'b', 'qty': 2},
        {'sku': 'a', 'qty': 1},
    ]
    editor.document['items'][1]['qty'] = 5  # sku 'b'

    reordered.save()
    editor.save()

    stored = table.get(doc_id=product)
    assert [item['sku'] for item in stored['items']] == ['b', 'a']
    assert stored['items'][0] == {'sku': 'b', 'qty': 5}
    assert stored['items'][1] == {'sku': 'a', 'qty': 1}


def test_unkeyed_list_with_duplicates_refuses_to_guess(table):
    doc_id = table.insert({'tags': ['a', 'a', 'b']})

    first = table.draft(doc_id)
    second = table.draft(doc_id)
    first.document['tags'] = ['a', 'a', 'c']
    second.document['tags'] = ['a', 'a', 'd']

    first.save()
    with pytest.raises(MergeConflict) as excinfo:
        second.save()

    assert excinfo.value.conflicts[0].reason == 'ambiguous_list_alignment'
    assert table.get(doc_id=doc_id)['tags'] == ['a', 'a', 'c']


def test_unkeyed_unique_list_merges_by_value(table):
    doc_id = table.insert({'tags': ['a', 'b', 'c']})

    first = table.draft(doc_id)
    second = table.draft(doc_id)
    first.document['tags'] = ['x', 'b', 'c']
    second.document['tags'] = ['a', 'b', 'y']

    first.save()
    second.save()

    assert table.get(doc_id=doc_id)['tags'] == ['x', 'b', 'y']


def test_discard_leaves_storage_untouched(table, product):
    draft = table.draft(product)
    draft.document['price'] = 999
    draft.discard()

    assert not draft.is_active
    with pytest.raises(RuntimeError):
        draft.save()

    stored = table.get(doc_id=product)
    assert stored['price'] == 10
    assert stored.doc_id == product


def test_query_after_conflict_still_resolves_doc_id(table, product):
    cond = where('name') == 'Coffee'
    first = table.draft(product)
    second = table.draft(product)

    first.document['price'] = 12
    second.document['price'] = 15

    first.save()
    with pytest.raises(MergeConflict):
        second.save()

    # Queries with the original doc_id still map to the real stored data
    assert table.get(doc_id=product)['price'] == 12
    assert table.search(cond)[0].doc_id == product

    # A fresh draft from the new state can still be saved
    retry = table.draft(product)
    retry.document['price'] = 15
    retry.save()
    assert table.get(doc_id=product)['price'] == 15


def test_doc_id_stable_across_parallel_saves(table, product):
    drafts = [table.draft(product) for _ in range(3)]
    drafts[0].document['name'] = 'Espresso'
    drafts[1].document['price'] = 11
    drafts[2].document['specs']['origin'] = 'CO'

    for draft in drafts:
        assert draft.save() == product

    stored = table.get(doc_id=product)
    assert stored.doc_id == product
    assert stored['name'] == 'Espresso'
    assert stored['price'] == 11
    assert stored['specs']['origin'] == 'CO'
    assert len(table) == 1


def test_draft_of_missing_document_raises(table):
    with pytest.raises(ValueError):
        table.draft(42)


def test_save_after_external_document_deletion_conflicts(table, product):
    draft = table.draft(product)
    draft.document['price'] = 12

    table.remove(doc_ids=[product])

    with pytest.raises(MergeConflict) as excinfo:
        draft.save()
    assert excinfo.value.conflicts[0].reason == 'document_deleted'
    assert table.get(doc_id=product) is None


def test_draft_baseline_is_detached(table, product):
    draft = table.draft(product)
    draft.document['specs']['weight'] = 300

    assert draft.baseline['specs']['weight'] == 250
    assert table.get(doc_id=product)['specs']['weight'] == 250


def test_save_without_changes_is_noop(table, product):
    raw_before = table.storage.read()
    draft = table.draft(product)
    assert draft.save() == product
    assert table.storage.read() == raw_before


def test_document_draft_exported():
    from tinydb import DocumentDraft as Imported
    assert Imported is DocumentDraft
