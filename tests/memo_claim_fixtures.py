"""
Deterministic in-memory fixtures for the memo claim preview tests.

The binaries are generated into pytest's ``tmp_path`` instead of being
committed, which keeps every byte under test explicit and reproducible.
The shared table is a copy of the shipped ``memotest.dbf`` (Visual FoxPro,
dbversion 0x30) whose active records reference memo blocks 1 and 2.
"""
import os
import shutil
import struct

VFP_TEXT = 0x01
DBT_BLOCK_SIZE = 512

CANONICAL_NOTES = {
    1: b'Alice memo',
    2: b'Bob memo',
}
VARIANT_NOTES = {
    1: b'Alice memo',
    2: b'Bob CHANGED',
}
WRONG_FINGERPRINT_NOTES = {
    1: b'Alice DIFFERENT',
    2: b'Bob memo',
}


def copy_table(tmp_path):
    """Copy the Visual FoxPro table into ``tmp_path`` and return its path."""
    target = os.path.join(str(tmp_path), 'memotest.dbf')
    source = os.path.join(os.path.dirname(__file__), 'cases', 'memotest.dbf')
    shutil.copyfile(source, target)
    return target


def write_fpt(path, blocksize, blocks, size=None, override_length=None):
    """Write a Visual FoxPro FPT.

    ``blocks`` maps block index to ``(type, data)``.  When ``override_length``
    is given for an index the memo header declares that many bytes regardless
    of how much data is actually present, which simulates a corrupt length.
    """
    with open(path, 'wb') as handle:
        handle.write(struct.pack('>LHH', 0, 0, blocksize))
        handle.write(b'\x00' * (512 - 8))
        for index, (memo_type, data) in blocks.items():
            handle.seek(index * blocksize)
            declared = override_length.get(index, len(data)) \
                if override_length else len(data)
            handle.write(struct.pack('>LL', memo_type, declared))
            handle.write(data)
        if size is not None:
            handle.truncate(size)


def write_db4_dbt(path, blocks, size=None, override_length=None):
    """Write a length-prefixed dBase IV style .DBT file."""
    with open(path, 'wb') as handle:
        handle.write(b'\x00' * DBT_BLOCK_SIZE)
        for index, data in blocks.items():
            handle.seek(index * DBT_BLOCK_SIZE)
            declared = override_length.get(index, len(data)) \
                if override_length else len(data)
            handle.write(struct.pack('<LL', 0x0808FFFF, declared))
            handle.write(data)
        if size is not None:
            handle.truncate(size)


def write_db3_dbt(path, blocks, size=None):
    """Write a marker-terminated dBase III style .DBT file."""
    with open(path, 'wb') as handle:
        handle.write(b'\x00' * DBT_BLOCK_SIZE)
        for index, data in blocks.items():
            handle.seek(index * DBT_BLOCK_SIZE)
            handle.write(data + b'\x1a')
        if size is not None:
            handle.truncate(size)


def build_fpt_candidates(tmp_path, blocksize=512):
    """Create the full set of FPT candidates plus a wrong-format .DBT."""
    directory = str(tmp_path)
    paths = {}

    paths['canonical'] = os.path.join(directory, 'memotest_canonical.FPT')
    write_fpt(paths['canonical'], blocksize,
              {index: (VFP_TEXT, data)
               for index, data in CANONICAL_NOTES.items()})

    # Byte-identical content under a different (lower-case) file name: it must
    # land in the same equivalence class purely on content.
    paths['copy'] = os.path.join(directory, 'memotest_canonical_copy.fpt')
    shutil.copyfile(paths['canonical'], paths['copy'])

    paths['variant'] = os.path.join(directory, 'memotest_variant.FPT')
    write_fpt(paths['variant'], blocksize,
              {index: (VFP_TEXT, data)
               for index, data in VARIANT_NOTES.items()})

    paths['wrong_fingerprint'] = os.path.join(
        directory, 'memotest_wrongfp.FPT')
    write_fpt(paths['wrong_fingerprint'], blocksize,
              {index: (VFP_TEXT, data)
               for index, data in WRONG_FINGERPRINT_NOTES.items()})

    paths['truncated'] = os.path.join(directory, 'memotest_truncated.FPT')
    write_fpt(paths['truncated'], blocksize,
              {1: (VFP_TEXT, CANONICAL_NOTES[1])},
              size=blocksize + 8 + len(CANONICAL_NOTES[1]))

    paths['bad_length'] = os.path.join(directory, 'memotest_badlength.FPT')
    write_fpt(paths['bad_length'], blocksize,
              {index: (VFP_TEXT, data)
               for index, data in CANONICAL_NOTES.items()},
              override_length={2: 10 ** 6})

    paths['wrong_format'] = os.path.join(directory, 'memotest_wrong.dbt')
    write_db4_dbt(paths['wrong_format'], CANONICAL_NOTES)

    return paths
