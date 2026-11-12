#!/usr/bin/env python3
"""Unittest class"""

import hashlib
import os
import sys
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

this_dir = Path(__file__).parent
sys.path.insert(0, str(this_dir))
sys.path.insert(0, str(this_dir.parent))

from probables import CountingBloomFilter  # noqa: E402
from probables.constants import UINT32_T_MAX  # noqa: E402
from probables.exceptions import InitializationError, SimilarityError  # noqa: E402
from tests.utilities import calc_file_md5, different_hash  # noqa: E402

DELETE_TEMP_FILES = True


class TestCountingBloomFilter(unittest.TestCase):
    """Test the default bloom filter implementation"""

    def test_cbf_init(self):
        """test version information"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        self.assertEqual(blm.false_positive_rate, 0.05000000074505806)
        self.assertEqual(blm.estimated_elements, 10)
        self.assertEqual(blm.number_hashes, 4)
        self.assertEqual(blm.number_bits, 63)
        self.assertEqual(blm.elements_added, 0)
        self.assertEqual(blm.is_on_disk, False)
        self.assertEqual(blm.bloom_length, 63)

    def test_cbf_ea(self):
        """test elements added is correct"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        self.assertEqual(blm.elements_added, 0)
        blm.add("this is a test")
        self.assertEqual(blm.elements_added, 1)

    def test_cbf_ea_diff_hash(self):
        """test elements added is correct"""
        blm1 = CountingBloomFilter(est_elements=10, false_positive_rate=0.05, hash_function=different_hash)
        hsh1 = blm1.hashes("this is a test")
        blm2 = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        hsh2 = blm2.hashes("this is a test")
        self.assertNotEqual(hsh1, hsh2)

    def test_cbf_check(self):
        """ensure that checking the bloom filter works"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        blm.add("this is a test")
        blm.add("this is another test")
        self.assertEqual(blm.check("this is a test"), True)
        self.assertEqual(blm.check("this is another test"), True)
        self.assertEqual(blm.check("this is yet another test"), False)
        self.assertEqual(blm.check("this is not another test"), False)

    def test_cbf_in_check(self):
        """check that the in construct works"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        blm.add("this is a test")
        blm.add("this is another test")
        self.assertEqual("this is a test" in blm, True)
        self.assertEqual("this is another test" in blm, True)
        self.assertEqual("this is yet another test" in blm, False)
        self.assertEqual("this is not another test" in blm, False)

    def test_cbf_stats(self):
        """test that the information in the stats is correct"""
        msg = (
            "CountingBloom:\n"
            "\tbits: 63\n"
            "\testimated elements: 10\n"
            "\tnumber hashes: 4\n"
            "\tmax false positive rate: 0.050000\n"
            "\telements added: 10\n"
            "\tcurrent false positive rate: 0.048806\n"
            "\tis on disk: no\n"
            "\tindex fullness: 0.603175\n"
            "\tmax index usage: 3\n"
            "\tmax index id: 56\n"
            "\tcalculated elements: 9\n"
        )
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        for i in range(0, 10):
            blm.add(f"this is a test {i}")
        stats = str(blm)
        self.assertEqual(stats, msg)

    def test_cbf_clear(self):
        """test clearing out the bloom filter"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        self.assertEqual(blm.elements_added, 0)
        for i in range(0, 10):
            blm.add(f"this is a test {i}")
        self.assertEqual(blm.elements_added, 10)

        blm.clear()
        self.assertEqual(blm.elements_added, 0)
        for idx in range(blm.bloom_length):
            self.assertEqual(blm._get_element(idx), 0)

    def test_cbf_export_file(self):
        """test exporting bloom filter to file"""
        md5_val = "bdb045d625b52cbb442f76e41588471a"
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".cbm", delete=DELETE_TEMP_FILES) as fobj:
            blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
            blm.add("test")
            blm.add("out")
            blm.add("the")
            blm.add("counting")
            blm.add("bloom")
            blm.add("filter")

            blm.add("test")
            blm.add("Test")
            blm.add("out")
            blm.add("test")
            blm.export(fobj.name)

            md5_out = calc_file_md5(fobj.name)
            self.assertEqual(md5_out, md5_val)

    def test_cbf_bytes(self):
        """test exporting counting bloom filter to bytes"""
        md5_val = "bdb045d625b52cbb442f76e41588471a"
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        blm.add("test")
        blm.add("out")
        blm.add("the")
        blm.add("counting")
        blm.add("bloom")
        blm.add("filter")

        blm.add("test")
        blm.add("Test")
        blm.add("out")
        blm.add("test")

        md5_out = hashlib.md5(bytes(blm)).hexdigest()
        self.assertEqual(md5_out, md5_val)

    def test_cbf_frombytes(self):
        """test loading counting bloom filter from bytes"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        blm.add("test")
        blm.add("out")
        blm.add("the")
        blm.add("counting")
        blm.add("bloom")
        blm.add("filter")

        blm.add("test")
        blm.add("Test")
        blm.add("out")
        blm.add("test")
        bytes_out = bytes(blm)

        blm2 = CountingBloomFilter.frombytes(bytes_out)
        self.assertEqual(blm.false_positive_rate, 0.05000000074505806)
        self.assertEqual(blm.estimated_elements, 10)
        self.assertEqual(blm.number_hashes, 4)
        self.assertEqual(blm.number_bits, 63)
        self.assertEqual(blm.elements_added, 10)
        self.assertEqual(blm.is_on_disk, False)
        self.assertEqual(blm.bloom_length, 63)

        self.assertEqual(bytes(blm2), bytes(blm))
        self.assertTrue(blm2.check("test"))
        self.assertFalse(blm2.check("something"))

    def test_cbf_load_file(self):
        """test loading bloom filter from file"""
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".cbm", delete=DELETE_TEMP_FILES) as fobj:
            blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
            blm.add("this is a test")
            blm.export(fobj.name)

            blm2 = CountingBloomFilter(filepath=fobj.name)
            self.assertEqual("this is a test" in blm2, True)
            self.assertEqual("this is not a test" in blm2, False)

    def test_cbf_load_invalid_file(self):
        """test importing a bloom filter from an invalid filepath"""
        filename = "invalid.cbm"
        self.assertRaises(InitializationError, lambda: CountingBloomFilter(filepath=filename))

    def test_cbf_invalid_params_msg(self):
        """test importing a bloom filter from an invalid filepath msg"""
        filename = "invalid.cbm"
        msg = "Insufecient parameters to set up the Counting Bloom Filter"
        try:
            CountingBloomFilter(filepath=filename)
        except InitializationError as ex:
            self.assertEqual(str(ex), msg)

    def test_cbf_export_hex(self):
        """test the exporting of the bloom filter to a hex string"""
        h_val = (
            "43424c4d010000000000000000000000"
            "00000000000000000000000000000000"
            "00000000000000000000000000000000"
            "00000000000000000000000000000000"
            "00000000010000000000000001000000"
            "02000000000000000100000001000000"
            "00000000000000000000000001000000"
            "00000000000000000200000000000000"
            "01000000020000000000000000000000"
            "00000000010000000000000000000000"
            "02000000000000000100000001000000"
            "00000000000000000000000001000000"
            "00000000000000000100000000000000"
            "01000000020000000000000000000000"
            "00000000010000000100000000000000"
            "01000000000000000100000001000000"
            "00000000000000000100000000000000"
            "01000000010000000100000000000000"
            "01000000030000000000000001000000"
            "01000000000000000000000001000000"
            "000000000000000a000000000000000a"
            "3d4ccccd"
        )

        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        for i in range(0, 10):
            tmp = f"this is a test {i}"
            blm.add(tmp)
        hex_out = blm.export_hex()

        self.assertEqual(hex_out, h_val)

    def test_cbf_load_hex(self):
        """test importing a bloom filter from hex value"""
        h_val = (
            "01000000000000000100000002000000000000000100000001000000"
            "00000000000000000000000001000000000000000000000002000000"
            "00000000010000000200000000000000000000000000000001000000"
            "00000000000000000200000000000000010000000200000000000000"
            "00000000000000000100000000000000000000000100000000000000"
            "01000000020000000000000000000000000000000100000001000000"
            "00000000010000000000000001000000020000000000000000000000"
            "01000000000000000100000001000000010000000000000001000000"
            "03000000000000000100000001000000000000000000000001000000"
            "000000000000000a000000000000000a3d4ccccd"
        )
        blm = CountingBloomFilter(hex_string=h_val)
        self.assertEqual("this is a test 0" in blm, True)
        self.assertEqual("this is a test 1" in blm, True)
        self.assertEqual("this is a test 2" in blm, True)
        self.assertEqual("this is a test 3" in blm, True)
        self.assertEqual("this is a test 4" in blm, True)
        self.assertEqual("this is a test 5" in blm, True)
        self.assertEqual("this is a test 6" in blm, True)
        self.assertEqual("this is a test 7" in blm, True)
        self.assertEqual("this is a test 8" in blm, True)
        self.assertEqual("this is a test 9" in blm, True)

        self.assertEqual("this is a test 10" in blm, False)
        self.assertEqual("this is a test 11" in blm, False)
        # self.assertEqual("this is a test 12" in blm, False)  # This is a false positive!
        self.assertEqual("this is a test 15" in blm, False)

    def test_cbf_load_invalid_hex(self):
        """test importing a bloom filter from an invalid hex value"""
        h_val = (
            "01000300000000010002000002010102000000000000010000010000000"
            "10100020100010101000002000101000201000000020001000001010000"
            "01020002000000000000000a000000000000000a3d4ccccQ"
        )
        self.assertRaises(InitializationError, lambda: CountingBloomFilter(hex_string=h_val))

    def test_cbf_export_c_header(self):
        """test exporting a c header"""
        hex_val = (
            "43424c4d0100000000000000"
            "000000000000000000000000"
            "000000000000000000000000"
            "000000000000000000000000"
            "000000000000000000000000"
            "000000000000000001000000"
            "000000000100000002000000"
            "000000000100000001000000"
            "000000000000000000000000"
            "010000000000000000000000"
            "020000000000000001000000"
            "020000000000000000000000"
            "000000000100000000000000"
            "000000000200000000000000"
            "010000000100000000000000"
            "000000000000000001000000"
            "000000000000000001000000"
            "000000000100000002000000"
            "000000000000000000000000"
            "010000000100000000000000"
            "010000000000000001000000"
            "010000000000000000000000"
            "010000000000000001000000"
            "010000000100000000000000"
            "010000000300000000000000"
            "010000000100000000000000"
            "000000000100000000000000"
            "0000000a000000000000000a"
            "3d4ccccd"
        )
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        for i in range(0, 10):
            tmp = f"this is a test {i}"
            blm.add(tmp)

        with NamedTemporaryFile(dir=os.getcwd(), suffix=".blm", delete=DELETE_TEMP_FILES) as fobj:
            blm.export_c_header(fobj.name)

            # now load the file, parse it and do some tests!
            with open(fobj.name) as fobj:
                data = fobj.readlines()

        data = [x.strip() for x in data]

        self.assertEqual("/* BloomFilter Export of a CountingBloomFilter */", data[0])
        self.assertEqual("#include <inttypes.h>", data[1])
        self.assertEqual(f"const uint64_t estimated_elements = {blm.estimated_elements};", data[2])
        self.assertEqual(f"const uint64_t elements_added = {blm.elements_added};", data[3])
        self.assertEqual(f"const float false_positive_rate = {blm.false_positive_rate};", data[4])
        self.assertEqual(f"const uint64_t number_bits = {blm.number_bits};", data[5])
        self.assertEqual(f"const unsigned int number_hashes = {blm.number_hashes};", data[6])
        self.assertEqual("const unsigned char bloom[] = {", data[7])
        self.assertEqual("};", data[-1])

        # rebuild the hex version!
        new_hex = "".join([x.strip().replace("0x", "") for x in " ".join(data[8:-1]).split(",")])
        self.assertEqual(hex_val, new_hex)

    def test_cbf_export_size(self):
        """test the size of the exported file"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        self.assertEqual(505, blm.export_size())

    def test_cbf_jaccard_ident(self):
        """test jaccard of two identical counting bloom filters"""
        blm1 = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        blm1.add("this is a test", 10)
        blm2 = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        blm2.add("this is a test", 10)
        self.assertEqual(blm1.jaccard_index(blm2), 1.0)

    def test_cbf_jaccard_ident_2(self):
        """test jaccard of two mostly identical counting bloom filters"""
        blm1 = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        blm1.add("this is a test", 10)
        blm2 = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        blm2.add("this is a test", 15)
        self.assertEqual(blm1.jaccard_index(blm2), 1.0)

    def test_cbf_jaccard_similar(self):
        """test jaccard of two similar counting bloom filters"""
        blm1 = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        blm1.add("this is a test", 10)
        blm1.add("this is a different test", 10)
        blm2 = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        blm2.add("this is a test", 10)
        blm2.add("this is also a test", 10)
        res = blm1.jaccard_index(blm2)
        self.assertGreater(res, 0.50)
        self.assertLessEqual(res, 0.60)

    def test_cbf_jaccard_similar_2(self):
        """test jaccard of two similar counting bloom filters - again"""
        blm1 = CountingBloomFilter(est_elements=100, false_positive_rate=0.01)
        blm1.add("this is a test", 10)
        blm1.add("this is a different test", 10)
        blm2 = CountingBloomFilter(est_elements=100, false_positive_rate=0.01)
        blm2.add("this is a test", 10)
        res = blm1.jaccard_index(blm2)
        self.assertLessEqual(res, 0.50)

    def test_cbf_jaccard_different(self):
        """test jaccard of two completly different counting bloom filters"""
        blm1 = CountingBloomFilter(est_elements=20, false_positive_rate=0.05)
        blm1.add("this is a test", 10)
        blm2 = CountingBloomFilter(est_elements=20, false_positive_rate=0.05)
        blm2.add("this is also a test", 10)
        self.assertEqual(blm1.jaccard_index(blm2), 0.0)

    def test_cbf_jaccard_empty(self):
        """test jaccard of an empty counting bloom filters"""
        blm1 = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        blm1.add("this is a test", 10)
        blm2 = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        self.assertEqual(blm1.jaccard_index(blm2), 0.0)

    def test_cbf_jaccard_empty_both(self):
        """test jaccard of an empty counting bloom filters"""
        blm1 = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        blm2 = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        self.assertEqual(blm1.jaccard_index(blm2), 1.0)

    def test_cbf_jaccard_different_2(self):
        """test jaccard of an mismath of counting bloom filters"""
        blm1 = CountingBloomFilter(est_elements=101, false_positive_rate=0.01)
        blm2 = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        self.assertRaises(SimilarityError, lambda: blm1.jaccard_index(blm2))

    def test_cbf_jaccard_invalid(self):
        """use an invalid type in a jaccard index"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        blm.add("this is a test")
        self.assertRaises(TypeError, lambda: blm.jaccard_index(1))

    def test_cbf_jaccard_invalid_msg(self):
        """check invalid type in a jaccard index message"""
        msg = "The parameter second must be of type CountingBloomFilter"
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        blm.add("this is another test")
        try:
            blm.jaccard_index(15)
        except TypeError as ex:
            self.assertEqual(str(ex), msg)
        else:
            self.assertEqual(True, False)

    def test_cbf_estimate_easy(self):
        """check estimate elements"""
        blm = CountingBloomFilter(est_elements=20, false_positive_rate=0.05)
        blm.add("this is a test", 10)
        blm.add("this is also a test", 5)
        self.assertEqual(blm.estimate_elements(), 2)

    def test_cbf_estimate_2(self):
        """check estimate elements - different"""
        blm = CountingBloomFilter(est_elements=20, false_positive_rate=0.05)
        blm.add("this is a test", 10)
        blm.add("this is a different test", 5)
        self.assertEqual(blm.estimate_elements(), 2)

    def test_cbf_remove(self):
        """test to see if the remove functionality works correctly"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        self.assertEqual(blm.elements_added, 0)
        for i in range(0, 5):
            tmp = f"this is a test {i}"
            blm.add(tmp)
        self.assertEqual(blm.elements_added, 5)
        res = blm.remove("this is a test 0")
        self.assertEqual(blm.elements_added, 4)
        self.assertEqual(res, 0)
        blm.remove("this is a test 0")
        self.assertEqual(blm.elements_added, 4)
        self.assertEqual(res, 0)

    def test_cbf_remove_mult(self):
        """test to see if the remove multiples functionality works correctly"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        self.assertEqual(blm.elements_added, 0)
        blm.add("this is a test 0", 15)
        self.assertEqual(blm.elements_added, 15)
        res = blm.remove("this is a test 0", 11)
        self.assertEqual(blm.elements_added, 4)
        self.assertEqual(res, 4)
        res = blm.remove("this is a test 0", 10)
        self.assertEqual(blm.elements_added, 0)
        self.assertEqual(res, 0)

    def test_cbf_very_large_add(self):
        """test adding a very large number of elements"""
        large = 2**32
        very_large = 2**64
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        self.assertEqual(blm.elements_added, 0)
        res = blm.add("this is a test 0", large)
        self.assertEqual(blm.elements_added, large)
        self.assertEqual(res, large - 1)
        res = blm.add("this is a test 0", very_large)
        self.assertEqual(blm.elements_added, very_large - 1)
        self.assertEqual(res, large - 1)

    def test_cbf_remove_from_large(self):
        """test adding a very large number of elements"""
        large = 2**32
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        self.assertEqual(blm.elements_added, 0)
        res = blm.add("this is a test 0", large)
        self.assertEqual(blm.elements_added, large)
        self.assertEqual(res, large - 1)

        res = blm.remove("this is a test 0", large)
        self.assertEqual(blm.elements_added, large)
        self.assertEqual(res, large - 1)

    def test_cbf_union(self):
        """test calculating the union between two counting bloom filters"""
        blm1 = CountingBloomFilter(est_elements=100, false_positive_rate=0.05)
        blm1.add("this is a test", 10)
        blm1.add("this is a different test", 10)
        blm2 = CountingBloomFilter(est_elements=100, false_positive_rate=0.05)
        blm2.add("this is a test", 10)
        res = blm1.union(blm2)

        self.assertEqual(res.check("this is a test"), 20)
        self.assertEqual(res.check("this is a different test"), 10)
        self.assertEqual(res.check("this is not a test"), 0)
        self.assertEqual(res.elements_added, 2)

    def test_cbf_union_error(self):
        """test union of two counting bloom filters type error"""
        blm1 = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        self.assertRaises(TypeError, lambda: blm1.union(1))

    def test_cbf_union_error_msg(self):
        """test union of two counting bloom filters type error msg"""
        blm1 = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        msg = "The parameter second must be of type CountingBloomFilter"
        try:
            blm1.union(1)
        except TypeError as ex:
            self.assertEqual(str(ex), msg)
        else:
            self.assertEqual(True, False)

    def test_cbf_union_diff(self):
        """test union of an mismath of counting bloom filters"""
        blm1 = CountingBloomFilter(est_elements=101, false_positive_rate=0.01)
        blm2 = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        self.assertRaises(SimilarityError, lambda: blm1.union(blm2))

    def test_cbf_inter(self):
        """test calculating the intersection between two
        counting bloom filters"""
        blm1 = CountingBloomFilter(est_elements=100, false_positive_rate=0.05)
        blm1.add("this is a test", 10)
        blm1.add("this is a different test", 10)
        blm2 = CountingBloomFilter(est_elements=100, false_positive_rate=0.05)
        blm2.add("this is a test", 10)
        res = blm1.intersection(blm2)

        self.assertEqual(res.check("this is a test"), 20)
        self.assertEqual(res.check("this is a different test"), 0)
        self.assertEqual(res.check("this is not a test"), 0)
        self.assertEqual(res.elements_added, 1)

    def test_cbf_inter_error(self):
        """test intersection of two counting bloom filters type error"""
        blm1 = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        self.assertRaises(TypeError, lambda: blm1.intersection(1))

    def test_cbf_inter_error_msg(self):
        """test intersection of two counting bloom filters type error msg"""
        blm1 = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        msg = "The parameter second must be of type CountingBloomFilter"
        try:
            blm1.intersection(1)
        except TypeError as ex:
            self.assertEqual(str(ex), msg)
        else:
            self.assertEqual(True, False)

    def test_cbf_inter_diff(self):
        """test intersection of an mismath of counting bloom filters"""
        blm1 = CountingBloomFilter(est_elements=101, false_positive_rate=0.01)
        blm2 = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        self.assertRaises(SimilarityError, lambda: blm1.intersection(blm2))

    def test_cbf_all_bits_set(self):
        """test inserting too many elements so that the all bits are set"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        for i in range(100):
            blm.add(str(i))
        # NOTE: this causes an exception when all bits are set
        self.assertEqual(-1, blm.estimate_elements())


class TestCountingBloomFilterSaturation(unittest.TestCase):
    """Test saturated bucket handling and exact/lower-bound distinction"""

    def test_saturated_bucket_keeps_accepting_events(self):
        """a saturated bucket stays clamped but still records new insertions"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        self.assertFalse(blm.has_saturated_buckets)
        blm.add("hot", UINT32_T_MAX)
        self.assertTrue(blm.has_saturated_buckets)
        self.assertEqual(blm.check("hot"), UINT32_T_MAX)

        blm.add("hot", 1000)
        self.assertEqual(blm.check("hot"), UINT32_T_MAX)
        self.assertTrue(blm.is_saturated("hot"))
        self.assertEqual(blm.saturated_buckets, tuple(sorted(blm.saturated_buckets)))

    def test_check_with_status_distinguishes_exact_from_lower_bound(self):
        """exact counters and lower-bound counters are distinguishable"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        blm.add("normal", 3)
        count, is_exact = blm.check_with_status("normal")
        self.assertEqual((count, is_exact), (3, True))

        blm.add("hot", UINT32_T_MAX)
        count, is_exact = blm.check_with_status("hot")
        self.assertEqual(count, UINT32_T_MAX)
        self.assertFalse(is_exact)

    def test_remove_does_not_erase_present_key_via_saturated_bucket(self):
        """removing from a saturated bucket must not drop a still-present key"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        blm.add("hot", UINT32_T_MAX)
        blm.remove("hot", 1000)
        self.assertEqual(blm.check("hot"), UINT32_T_MAX)
        self.assertTrue(blm.is_saturated("hot"))
        self.assertTrue("hot" in blm)

    def test_remove_mixed_saturated_and_exact_buckets(self):
        """only exact buckets are decremented when one bucket is saturated"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        saturated_key = [7, 11, 13, 17]
        blm.add_alt(saturated_key, UINT32_T_MAX)
        # shares bucket 7 with the saturated key; other buckets stay exact
        mixed_key = [7, 21, 25, 29]
        blm.add_alt(mixed_key, 4)
        blm.remove_alt(mixed_key, 2)
        self.assertEqual(blm.bloom[21], 2)
        self.assertEqual(blm.bloom[25], 2)
        self.assertEqual(blm.bloom[29], 2)
        self.assertEqual(blm.bloom[7], UINT32_T_MAX)

    def test_duplicate_hash_positions_count_once(self):
        """repeated hash positions within a key update each bucket once"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        key = [5, 5, 9, 9]
        blm.add_alt(key)
        self.assertEqual(blm.bloom[5], 1)
        self.assertEqual(blm.bloom[9], 1)
        self.assertEqual(blm.check_alt(key), 1)
        blm.remove_alt(key)
        self.assertEqual(blm.bloom[5], 0)
        self.assertEqual(blm.bloom[9], 0)

    def test_clear_resets_saturation(self):
        """clearing also resets the saturation flags"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        blm.add("hot", UINT32_T_MAX)
        self.assertTrue(blm.has_saturated_buckets)
        blm.clear()
        self.assertFalse(blm.has_saturated_buckets)
        self.assertEqual(blm.saturated_buckets, ())


class TestCountingBloomFilterExport(unittest.TestCase):
    """Test versioned export round trips and legacy format compatibility"""

    def test_bytes_round_trip_preserves_saturation(self):
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        blm.add("normal", 2)
        blm.add("hot", UINT32_T_MAX)
        saturated = blm.saturated_buckets

        blm2 = CountingBloomFilter.frombytes(bytes(blm))
        self.assertEqual(bytes(blm2), bytes(blm))
        self.assertEqual(blm2.saturated_buckets, saturated)
        self.assertEqual(blm2.check_with_status("normal"), (2, True))
        self.assertEqual(blm2.check_with_status("hot"), (UINT32_T_MAX, False))

    def test_legacy_bytes_load_without_saturation_flags(self):
        """legacy binary exports (header-less) still load as exact counters"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        blm.add("this is a test")
        data = bytes(blm)
        header = len(CountingBloomFilter._EXPORT_HEADER)
        legacy = data[header + blm.bloom_length:]

        blm2 = CountingBloomFilter.frombytes(legacy)
        self.assertTrue(blm2.check("this is a test"))
        self.assertFalse(blm2.has_saturated_buckets)

    def test_legacy_hex_still_loads(self):
        """legacy hex exports without the versioned header remain readable"""
        h_val = (
            "01000000000000000100000002000000000000000100000001000000"
            "00000000000000000000000001000000000000000000000002000000"
            "00000000010000000200000000000000000000000000000001000000"
            "00000000000000000200000000000000010000000200000000000000"
            "00000000000000000100000000000000000000000100000000000000"
            "01000000020000000000000000000000000000000100000001000000"
            "00000000010000000000000001000000020000000000000000000000"
            "01000000000000000100000001000000010000000000000001000000"
            "03000000000000000100000001000000000000000000000001000000"
            "000000000000000a000000000000000a3d4ccccd"
        )
        blm = CountingBloomFilter(hex_string=h_val)
        self.assertTrue("this is a test 0" in blm)
        self.assertFalse(blm.has_saturated_buckets)

    def test_hex_round_trip_preserves_saturation(self):
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        blm.add("hot", UINT32_T_MAX)
        blm2 = CountingBloomFilter(hex_string=blm.export_hex())
        self.assertEqual(blm2.saturated_buckets, blm.saturated_buckets)
        self.assertEqual(blm2.check_with_status("hot"), (UINT32_T_MAX, False))

    def test_file_round_trip_preserves_saturation(self):
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".cbm", delete=DELETE_TEMP_FILES) as fobj:
            blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
            blm.add("hot", UINT32_T_MAX)
            blm.export(fobj.name)

            blm2 = CountingBloomFilter(filepath=fobj.name)
            self.assertEqual(blm2.saturated_buckets, blm.saturated_buckets)
            self.assertTrue(blm2.is_saturated("hot"))

    def test_export_size_matches_bytes(self):
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.01)
        blm.add("this is a test")
        self.assertEqual(blm.export_size(), len(bytes(blm)))

    def test_union_and_intersection_propagate_saturation(self):
        blm1 = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        blm2 = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        blm1.add("hot", UINT32_T_MAX)
        blm2.add("other")

        union = blm1.union(blm2)
        self.assertTrue(all(union._saturated[i] for i in blm1.saturated_buckets))

        intersection = blm1.intersection(blm2)
        for i in range(blm1.bloom_length):
            expected_flag = (
                blm1._saturated[i] | blm2._saturated[i]
                if blm1.bloom[i] > 0 and blm2.bloom[i] > 0
                else 0
            )
            self.assertEqual(intersection._saturated[i], expected_flag)


class TestCountingBloomFilterReconcile(unittest.TestCase):
    """Test sealed-segment reconciliation against the complete key stream"""

    def test_reconcile_recovers_later_batch_only(self):
        """recomputing a sealed segment leaves exactly the later batch ledger"""
        segment_one = [f"seg1-{i}" for i in range(8)]
        segment_two = [f"seg2-{i}" for i in range(8)]

        blm = CountingBloomFilter(est_elements=30, false_positive_rate=0.05)
        for key in segment_one + segment_two:
            blm.add(key)
        blm.reconcile(segment_one)

        expected = CountingBloomFilter(est_elements=30, false_positive_rate=0.05)
        for key in segment_two:
            expected.add(key)
        self.assertEqual(list(blm.bloom), list(expected.bloom))
        self.assertEqual(blm.elements_added, len(segment_two))
        self.assertTrue(all(key not in blm for key in segment_one))
        self.assertTrue(all(key in blm for key in segment_two))

    def test_reconcile_keeps_later_batch_share_of_shared_bucket(self):
        """a shared bucket keeps the still-present key's contribution"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        sealed = [5, 10, 15, 20]
        present = [5, 25, 30, 35]  # shares bucket 5 with the sealed key
        blm.add_alt(sealed)
        blm.add_alt(present)
        self.assertEqual(blm.bloom[5], 2)

        touched = blm.reconcile_alt([sealed])
        self.assertEqual(touched, 4)
        self.assertEqual(blm.bloom[5], 1)
        self.assertEqual(blm.check_alt(present), 1)
        self.assertEqual(blm.check_alt(sealed), 0)

    def test_reconcile_is_incremental(self):
        """splitting the segment stream into pieces reaches the same result"""
        segment_one = [f"seg1-{i}" for i in range(8)]
        segment_two = [f"seg2-{i}" for i in range(8)]

        blm = CountingBloomFilter(est_elements=30, false_positive_rate=0.05)
        for key in segment_one + segment_two:
            blm.add(key)
        split = 3
        blm.reconcile(segment_one[:split])
        blm.reconcile(segment_one[split:])

        expected = CountingBloomFilter(est_elements=30, false_positive_rate=0.05)
        for key in segment_two:
            expected.add(key)
        self.assertEqual(list(blm.bloom), list(expected.bloom))

    def test_reconcile_honors_multiplicity_and_duplicate_positions(self):
        """shared buckets accumulate the real update multiplicity"""
        blm = CountingBloomFilter(est_elements=30, false_positive_rate=0.05)
        segment_one = [f"seg1-{i}" for i in range(6)]
        segment_two = [f"seg2-{i}" for i in range(6)]
        for key in segment_one:
            blm.add(key, 3)
        for key in segment_two:
            blm.add(key)
        blm.reconcile(segment_one, num_els=3)

        expected = CountingBloomFilter(est_elements=30, false_positive_rate=0.05)
        for key in segment_two:
            expected.add(key)
        self.assertEqual(list(blm.bloom), list(expected.bloom))
        self.assertEqual(blm.elements_added, len(segment_two))

    def test_reconcile_retains_saturation_flags(self):
        """a clamped bucket stays a lower bound after subtracting a segment"""
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        blm.add("hot", UINT32_T_MAX)
        blm.add("later", 2)
        saturated = set(blm.saturated_buckets)
        blm.reconcile(["hot"], num_els=UINT32_T_MAX)
        self.assertEqual(set(blm.saturated_buckets), saturated)
        self.assertTrue(blm.is_saturated("hot"))

    def test_reconcile_invalid_num_els(self):
        blm = CountingBloomFilter(est_elements=10, false_positive_rate=0.05)
        blm.add("key")
        self.assertRaises(InitializationError, lambda: blm.reconcile(["key"], num_els=0))


if __name__ == "__main__":
    unittest.main()
