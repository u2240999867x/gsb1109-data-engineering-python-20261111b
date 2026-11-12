#
# Copyright (C) 2009-2020 the sqlparse authors and contributors
# <see AUTHORS file>
#
# This module is part of python-sqlparse and is released under
# the BSD License: https://opensource.org/licenses/BSD-3-Clause

import pytest

import sqlparse
from sqlparse import ScriptCheckpoint, split_script


def test_go_splits_batches():
    script = 'SELECT 1\nGO\nSELECT 2\nGO\n'
    result = split_script(script)
    assert [b.sql for b in result.batches] == ['SELECT 1', 'SELECT 2']
    assert all(b.delimiter == 'GO' for b in result.batches)
    assert all(b.count == 1 for b in result.batches)


def test_go_is_case_insensitive():
    result = split_script('SELECT 1\ngo\nSELECT 2\nGo\n')
    assert [b.sql for b in result.batches] == ['SELECT 1', 'SELECT 2']


def test_go_repeat_count():
    result = split_script('INSERT INTO t VALUES (1)\nGO 3\n')
    assert len(result.batches) == 1
    assert result.batches[0].count == 3
    assert result.batches[0].delimiter == 'GO'


def test_go_must_be_on_own_line():
    result = split_script('SELECT 1 GO\n')
    assert len(result.batches) == 1
    assert result.batches[0].sql == 'SELECT 1 GO'
    assert result.batches[0].delimiter == ';'


def test_go_inside_string_is_sql():
    script = "SELECT 'x\nGO\ny'\nGO\n"
    result = split_script(script)
    assert len(result.batches) == 1
    assert result.batches[0].sql == "SELECT 'x\nGO\ny'"


def test_go_inside_comments_is_sql():
    script = '-- GO\nSELECT 1\nGO\n/*\nGO\n*/\nSELECT 2\nGO\n'
    result = split_script(script)
    assert [b.sql for b in result.batches] == [
        '-- GO\nSELECT 1', '/*\nGO\n*/\nSELECT 2']


def test_delimiter_switch_keeps_procedure_body_intact():
    script = (
        'DELIMITER //\n'
        'CREATE PROCEDURE p()\n'
        'BEGIN\n'
        '  SELECT 1;\n'
        '  SELECT 2;\n'
        'END//\n'
        'DELIMITER ;\n'
        'SELECT 3;\n'
    )
    result = split_script(script)
    assert len(result.batches) == 2
    proc, select = result.batches
    assert proc.delimiter == '//'
    assert proc.sql.startswith('CREATE PROCEDURE p()')
    assert proc.sql.endswith('END')
    assert 'SELECT 1;' in proc.sql
    assert select.delimiter == ';'
    assert select.sql == 'SELECT 3'
    assert result.delimiter == ';'


def test_delimiter_inside_comment_is_ignored():
    result = split_script('/* DELIMITER // */\nSELECT 1;\n')
    assert len(result.batches) == 1
    assert result.batches[0].delimiter == ';'


def test_delimiter_word_inside_string_is_sql():
    script = "SELECT 'DELIMITER //';\nSELECT 2;\n"
    result = split_script(script)
    assert [b.sql for b in result.batches] == [
        "SELECT 'DELIMITER //'", 'SELECT 2']


def test_unknown_client_commands_are_listed_separately():
    script = ':setvar X 1\nSELECT 1;\n\\connect db\nSELECT 2;\n'
    result = split_script(script)
    assert [b.sql for b in result.batches] == ['SELECT 1', 'SELECT 2']
    assert [c.text for c in result.client_commands] == [
        ':setvar X 1', '\\connect db']
    for batch in result.batches:
        assert ':setvar' not in batch.sql
        assert '\\connect' not in batch.sql


def test_client_command_inside_string_is_sql():
    result = split_script("SELECT ':setvar';\n")
    assert len(result.batches) == 1
    assert result.client_commands == ()


def test_client_command_with_quoted_argument():
    result = split_script(":setvar Region 'eu'\nSELECT 1;\n")
    assert [b.sql for b in result.batches] == ['SELECT 1']
    assert [c.text for c in result.client_commands] == [":setvar Region 'eu'"]


def test_directive_like_lines_inside_multiline_string_are_sql():
    script = "SELECT '\nGO\nDELIMITER //\n:setvar x\n';\n"
    result = split_script(script)
    assert len(result.batches) == 1
    assert result.batches[0].delimiter == ';'
    assert result.client_commands == ()


def test_identical_sql_from_different_ranges_is_not_merged():
    script = 'SELECT 1;\nSELECT 1;\n'
    result = split_script(script)
    assert len(result.batches) == 2
    first, second = result.batches
    assert first.sql == second.sql
    assert (first.start, first.end) != (second.start, second.end)


def test_source_ranges_map_back_to_script():
    script = (
        'INSERT INTO a VALUES (1);\n'
        'GO 2\n'
        'DELIMITER //\n'
        'UPDATE b SET x = 1//\n'
    )
    result = split_script(script)
    assert len(result.batches) == 2
    for batch in result.batches:
        assert script[batch.start:batch.end] == batch.sql
    assert result.batches[0].count == 2
    assert result.batches[1].delimiter == '//'


def test_multiple_statements_per_line():
    result = split_script('SELECT 1; SELECT 2;\n')
    assert [b.sql for b in result.batches] == ['SELECT 1', 'SELECT 2']


def test_unterminated_tail_uses_current_delimiter():
    result = split_script('SELECT 1')
    assert len(result.batches) == 1
    assert result.batches[0].delimiter == ';'
    assert result.batches[0].count == 1


def test_empty_script():
    result = split_script('')
    assert result.batches == ()
    assert result.client_commands == ()
    assert list(result.resume(ScriptCheckpoint(batch=0))) == []


def test_bytes_input_with_encoding():
    result = split_script(b'SELECT 1;\n')
    assert [b.sql for b in result.batches] == ['SELECT 1']


def test_resume_mid_count_skips_acknowledged_executions():
    result = split_script('INSERT INTO t VALUES (1)\nGO 3\nSELECT 2;\n')
    remaining = list(result.resume(ScriptCheckpoint(batch=0, executed=2)))
    assert [b.sql for b in remaining] == [
        'INSERT INTO t VALUES (1)', 'SELECT 2']
    assert remaining[0].count == 1
    assert remaining[1].count == 1


def test_resume_after_completed_batch():
    result = split_script('INSERT INTO t VALUES (1)\nGO 3\nSELECT 2;\n')
    remaining = list(result.resume(ScriptCheckpoint(batch=0, executed=3)))
    assert [b.sql for b in remaining] == ['SELECT 2']


def test_resume_preserves_delimiter_policy():
    script = (
        'DELIMITER //\n'
        'CREATE PROCEDURE p() BEGIN SELECT 1; END//\n'
        'UPDATE a SET x = 1//\n'
    )
    result = split_script(script)
    remaining = list(result.resume(ScriptCheckpoint(batch=0, executed=1)))
    assert [b.sql for b in remaining] == ['UPDATE a SET x = 1']
    assert remaining[0].delimiter == '//'


def test_resume_at_end_of_script():
    result = split_script('SELECT 1;\n')
    assert list(result.resume(ScriptCheckpoint(batch=1))) == []


def test_checkpoint_helper():
    result = split_script('SELECT 1\nGO 2\n')
    cp = result.checkpoint(0, executed=1)
    assert [b.count for b in result.resume(cp)] == [1]


@pytest.mark.parametrize('checkpoint', [
    ScriptCheckpoint(batch=-1),
    ScriptCheckpoint(batch=2),
    ScriptCheckpoint(batch=0, executed=-1),
    ScriptCheckpoint(batch=0, executed=2),
    ScriptCheckpoint(batch=1, executed=1),
])
def test_resume_rejects_invalid_checkpoints(checkpoint):
    result = split_script('SELECT 1;\n')
    with pytest.raises(ValueError):
        list(result.resume(checkpoint))


def test_public_api_is_exported():
    assert sqlparse.split_script is split_script
    assert 'split_script' in sqlparse.__all__
