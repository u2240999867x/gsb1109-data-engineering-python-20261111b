from nose2.tools.such import helper

import dpath
import dpath.exceptions


def test_delete_separator():
    dict = {
        "a": {
            "b": 0,
        },
    }

    dpath.delete(dict, ';a;b', separator=";")
    assert 'b' not in dict['a']


def test_delete_existing():
    dict = {
        "a": {
            "b": 0,
        },
    }

    dpath.delete(dict, '/a/b')
    assert 'b' not in dict['a']


def test_delete_missing():
    dict = {
        "a": {
        },
    }

    with helper.assertRaises(dpath.exceptions.PathNotFound):
        dpath.delete(dict, '/a/b')


def test_delete_filter():
    def afilter(x):
        if int(x) == 31:
            return True
        return False

    dict = {
        "a": {
            "b": 0,
            "c": 1,
            "d": 31,
        },
    }

    dpath.delete(dict, '/a/*', afilter=afilter)
    assert dict['a']['b'] == 0
    assert dict['a']['c'] == 1
    assert 'd' not in dict['a']


def test_delete_filter_respects_glob_scope():
    def afilter(x):
        return x == 'secret'

    dict = {
        "selected": {
            "token": "secret",
            "keep": 1,
        },
        "retained": {
            "token": "secret",
            "keep": 2,
        },
    }

    deleted = dpath.delete(dict, '/selected/token', afilter=afilter)
    assert deleted == 1
    assert 'token' not in dict['selected']
    assert dict['retained']['token'] == 'secret'


def test_delete_shared_object_retained_path_untouched():
    shared = {
        "token": "secret",
        "keep": 1,
    }
    dict = {
        "selected": shared,
        "retained": shared,
    }

    deleted = dpath.delete(dict, '/selected/token')
    assert deleted == 1
    assert dict['selected'] == {"keep": 1}
    assert dict['retained'] == {"token": "secret", "keep": 1}
    assert dict['retained'] is shared


def test_delete_shared_nested_object_retained_path_untouched():
    shared_sub = {
        "x": 1,
        "y": 2,
    }
    shared = {
        "sub": shared_sub,
    }
    dict = {
        "selected": shared,
        "retained": shared,
    }

    dpath.delete(dict, '/selected/sub/x')
    assert dict['selected'] == {"sub": {"y": 2}}
    assert dict['retained'] == {"sub": {"x": 1, "y": 2}}
    assert dict['retained'] is shared
    assert dict['retained']['sub'] is shared_sub


def test_delete_shared_object_selected_paths_share_removal():
    shared = {
        "token": "secret",
        "keep": 1,
    }
    dict = {
        "first": shared,
        "second": shared,
    }

    deleted = dpath.delete(dict, '/*/token')
    assert deleted == 2
    assert 'token' not in dict['first']
    assert 'token' not in dict['second']
    assert dict['first'] is dict['second']


def test_delete_unaffected_shared_columns_stay_shared():
    shared = {
        "keep": 1,
    }
    dict = {
        "a": shared,
        "b": shared,
        "c": {
            "token": "secret",
        },
    }

    dpath.delete(dict, '/c/token')
    assert dict['a'] is shared
    assert dict['b'] is shared


def test_delete_filter_observes_pre_delete_values():
    seen = []

    def afilter(x):
        seen.append(x)
        return True

    dict = {
        "a": [1, 2, 3],
    }

    dpath.delete(dict, '/a/*', afilter=afilter)
    assert sorted(seen) == [1, 2, 3]


def test_delete_filter_error_leaves_record_untouched():
    def afilter(x):
        if x == 2:
            raise ValueError("boom")
        return True

    dict = {
        "a": {
            "x": 1,
            "y": 2,
        },
    }

    with helper.assertRaises(ValueError):
        dpath.delete(dict, '/a/*', afilter=afilter)
    assert dict == {"a": {"x": 1, "y": 2}}
