#!/usr/bin/env python3
"""Unittest class"""

import hashlib
import os
import random
import sys
import unittest
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile

this_dir = Path(__file__).parent
sys.path.insert(0, str(this_dir))
sys.path.insert(0, str(this_dir.parent))

from probables import (  # noqa: E402
    UPDATE_POLICY_CONSERVATIVE,
    UPDATE_POLICY_STANDARD,
    CountMeanMinSketch,
    CountMeanSketch,
    CountMinSketch,
    HeavyHitters,
    StreamThreshold,
)
from probables.constants import INT32_T_MAX, INT32_T_MIN, INT64_T_MAX, INT64_T_MIN  # noqa: E402
from probables.exceptions import CountMinSketchError, InitializationError, NotSupportedError  # noqa: E402
from tests.utilities import calc_file_md5, different_hash  # noqa: E402

DELETE_TEMP_FILES = True


def controlled_hash(key, depth):
    """deterministic hash mapping a tiny vocabulary onto width-2 columns:
    x -> (0, 0), y -> (0, 1), z -> (1, 0), anything else -> (1, 1)"""
    mapping = {"x": (0, 0), "y": (0, 1), "z": (1, 0)}
    return list(mapping.get(key, (1, 1)))[:depth]


class TestCountMinSketch(unittest.TestCase):
    """Test the default count-min sketch implementation"""

    def test_cms_init_wd(self):
        """Test count-min sketch initialization using depth and width"""
        cms = CountMinSketch(width=1000, depth=5)
        self.assertEqual(cms.width, 1000)
        self.assertEqual(cms.depth, 5)
        self.assertEqual(cms.confidence, 0.96875)
        self.assertEqual(cms.error_rate, 0.002)
        self.assertEqual(cms.elements_added, 0)

    def test_cms_init_ce(self):
        """Test count-min sketch initialization using confidence and error
        rate"""
        cms = CountMinSketch(confidence=0.96875, error_rate=0.002)
        self.assertEqual(cms.width, 1000)
        self.assertEqual(cms.depth, 5)
        self.assertEqual(cms.confidence, 0.96875)
        self.assertEqual(cms.error_rate, 0.002)
        self.assertEqual(cms.elements_added, 0)

    def test_cms_init_error(self):
        """Test count-min sketch initialization without enough params"""
        self.assertRaises(InitializationError, lambda: CountMinSketch(width=1000))

    def test_cms_init_error_msg(self):
        """Test count-min sketch initialization without enough params"""
        try:
            CountMinSketch(width=1000)
        except InitializationError as ex:
            msg = (
                "Must provide one of the following to initialize the "
                "Count-Min Sketch:\n"
                "    A file to load,\n"
                "    The width and depth,\n"
                "    OR confidence and error rate"
            )
            self.assertEqual(str(ex), msg)
        else:
            self.assertEqual(True, False)

    def test_cms_set_query_type(self):
        """test setting different query types"""
        cms = CountMinSketch(width=1000, depth=5)
        self.assertEqual(cms.query_type, "min")
        cms.query_type = "mean-min"
        self.assertEqual(cms.query_type, "mean-min")
        cms.query_type = "mean"
        self.assertEqual(cms.query_type, "mean")
        cms.query_type = "unknown"
        self.assertEqual(cms.query_type, "min")

    def test_cms_add_single(self):
        """test the insertion of a single element at a time"""
        cms = CountMinSketch(width=1000, depth=5)
        self.assertEqual(cms.add("this is a test"), 1)
        self.assertEqual(cms.add("this is a test"), 2)
        self.assertEqual(cms.add("this is a test"), 3)
        self.assertEqual(cms.add("this is a test"), 4)
        self.assertEqual(cms.elements_added, 4)

    def test_cms_add_mult(self):
        """test the insertion of multiple elements at a time"""
        cms = CountMinSketch(width=1000, depth=5)
        self.assertEqual(cms.add("this is a test", 4), 4)
        self.assertEqual(cms.add("this is a test", 4), 8)
        self.assertEqual(cms.add("this is a test", 4), 12)
        self.assertEqual(cms.add("this is a test", 4), 16)
        self.assertEqual(cms.elements_added, 16)

    def test_cms_remove_single(self):
        """test the removal of a single element at a time"""
        cms = CountMinSketch(width=1000, depth=5)
        self.assertEqual(cms.add("this is a test", 4), 4)
        self.assertEqual(cms.elements_added, 4)
        self.assertEqual(cms.remove("this is a test"), 3)
        self.assertEqual(cms.remove("this is a test"), 2)
        self.assertEqual(cms.elements_added, 2)

    def test_cms_remove_mult(self):
        """test the removal of multiple elements at a time"""
        cms = CountMinSketch(width=1000, depth=5)
        self.assertEqual(cms.add("this is a test", 16), 16)
        self.assertEqual(cms.elements_added, 16)
        self.assertEqual(cms.remove("this is a test", 4), 12)
        self.assertEqual(cms.elements_added, 12)

    def test_cms_check_min(self):
        """test checking number elements using min algorithm"""
        cms = CountMinSketch(width=1000, depth=5)
        self.assertEqual(cms.add("this is a test", 255), 255)
        self.assertEqual(cms.add("this is another test", 189), 189)
        self.assertEqual(cms.add("this is also a test", 16), 16)
        self.assertEqual(cms.add("this is something to test", 5), 5)

        self.assertEqual(cms.check("this is something to test"), 5)
        self.assertEqual(cms.check("this is also a test"), 16)
        self.assertEqual(cms.check("this is another test"), 189)
        self.assertEqual(cms.check("this is a test"), 255)
        self.assertEqual(cms.elements_added, 5 + 16 + 189 + 255)

    def test_cms_check_min_called(self):
        """test checking number elements using min algorithm called out"""
        cms = CountMinSketch(width=1000, depth=5)
        cms.query_type = None  # type: ignore
        self.assertEqual(cms.add("this is a test", 255), 255)
        self.assertEqual(cms.add("this is another test", 189), 189)
        self.assertEqual(cms.add("this is also a test", 16), 16)
        self.assertEqual(cms.add("this is something to test", 5), 5)

        self.assertEqual(cms.check("this is something to test"), 5)
        self.assertEqual(cms.check("this is also a test"), 16)
        self.assertEqual(cms.check("this is another test"), 189)
        self.assertEqual(cms.check("this is a test"), 255)
        self.assertEqual(cms.elements_added, 5 + 16 + 189 + 255)

    def test_cms_check_mean_called(self):
        """test checking number elements using mean algorithm called out"""
        cms = CountMinSketch(width=1000, depth=5)
        cms.query_type = "mean"
        self.assertEqual(cms.add("this is a test", 255), 255)
        self.assertEqual(cms.add("this is another test", 189), 189)
        self.assertEqual(cms.add("this is also a test", 16), 16)
        self.assertEqual(cms.add("this is something to test", 5), 5)

        self.assertEqual(cms.check("this is something to test"), 5)
        self.assertEqual(cms.check("this is also a test"), 16)
        self.assertEqual(cms.check("this is another test"), 189)
        self.assertEqual(cms.check("this is a test"), 255)
        self.assertEqual(cms.elements_added, 5 + 16 + 189 + 255)

    def test_cms_check_mean_min_called(self):
        """test checking number elements using mean-min algorithm called
        out"""
        cms = CountMinSketch(width=1000, depth=5)
        cms.query_type = "mean-min"
        self.assertEqual(cms.add("this is a test", 255), 255)
        self.assertEqual(cms.add("this is another test", 189), 189)
        self.assertEqual(cms.add("this is also a test", 16), 16)
        self.assertEqual(cms.add("this is something to test", 5), 5)

        self.assertEqual(cms.check("this is something to test"), 5)
        self.assertEqual(cms.check("this is also a test"), 16)
        self.assertEqual(cms.check("this is another test"), 189)
        self.assertEqual(cms.check("this is a test"), 255)
        self.assertEqual(cms.elements_added, 5 + 16 + 189 + 255)

    def test_cms_check_mean_called_even(self):
        """test checking number elements using mean algorithm called out when
        the depth is an even number..."""
        cms = CountMinSketch(width=1000, depth=6)
        cms.query_type = "mean-min"
        self.assertEqual(cms.add("this is a test", 255), 255)
        self.assertEqual(cms.add("this is another test", 189), 189)
        self.assertEqual(cms.add("this is also a test", 16), 16)
        self.assertEqual(cms.add("this is something to test", 5), 5)

        self.assertEqual(cms.check("this is something to test"), 5)
        self.assertEqual(cms.check("this is also a test"), 16)
        self.assertEqual(cms.check("this is another test"), 189)
        self.assertEqual(cms.check("this is a test"), 255)
        self.assertEqual(cms.elements_added, 5 + 16 + 189 + 255)

    def test_cms_export(self):
        """test exporting a count-min sketch"""
        md5_val = "fb1c39dd1a73f1ef0d7fc79f60fc028e"
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".cms", delete=DELETE_TEMP_FILES) as fobj:
            cms = CountMinSketch(width=1000, depth=5)
            cms.add("this is a test", 100)
            cms.export(fobj.name)
            md5_out = calc_file_md5(fobj.name)
        self.assertEqual(md5_out, md5_val)

    def test_cms_bytes(self):
        """test exporting a count-min sketch as bytes"""
        md5_val = "fb1c39dd1a73f1ef0d7fc79f60fc028e"
        cms = CountMinSketch(width=1000, depth=5)
        cms.add("this is a test", 100)
        md5_out = hashlib.md5(bytes(cms)).hexdigest()
        self.assertEqual(md5_out, md5_val)

    def test_cms_frombytes(self):
        """test loading a count-min sketch from bytes"""
        cms = CountMinSketch(width=1000, depth=5)
        cms.add("this is a test", 100)
        bytes_out = bytes(cms)

        cms2 = CountMinSketch.frombytes(bytes_out)
        self.assertEqual(bytes(cms2), bytes(cms))
        self.assertEqual(cms2.width, 1000)
        self.assertEqual(cms2.depth, 5)
        self.assertEqual(cms2.check("this is a test"), 100)

    def test_cms_load(self):
        """test loading a count-min sketch from file"""
        md5_val = "fb1c39dd1a73f1ef0d7fc79f60fc028e"
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".cms", delete=DELETE_TEMP_FILES) as fobj:
            cms = CountMinSketch(width=1000, depth=5)
            self.assertEqual(cms.add("this is a test", 100), 100)
            cms.export(fobj.name)
            md5_out = calc_file_md5(fobj.name)
            self.assertEqual(md5_out, md5_val)

            # try loading directly to file!
            cms2 = CountMinSketch(filepath=fobj.name)
            self.assertEqual(cms2.elements_added, 100)
            self.assertEqual(cms2.check("this is a test"), 100)

    def test_cms_load_diff_hash(self):
        """test loading a count-min sketch from file"""
        md5_val = "fb1c39dd1a73f1ef0d7fc79f60fc028e"
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".cms", delete=DELETE_TEMP_FILES) as fobj:
            cms = CountMinSketch(width=1000, depth=5)
            self.assertEqual(cms.add("this is a test", 100), 100)
            cms.export(fobj.name)
            md5_out = calc_file_md5(fobj.name)
            self.assertEqual(md5_out, md5_val)

            cms2 = CountMinSketch(filepath=fobj.name, hash_function=different_hash)
            self.assertEqual(cms2.elements_added, 100)
            # should not work since it is a different hash
            self.assertNotEqual(cms.check("this is a test"), True)
            self.assertNotEqual(cms.hashes("this is a test"), cms2.hashes("this is a test"))

    def test_cms_load_invalid_file(self):
        """test loading a count-min sketch from invalid file"""
        filename = "invalid.cms"
        self.assertRaises(InitializationError, lambda: CountMinSketch(filepath=filename))

    def test_cms_different_hash(self):
        """test using a different hash function"""
        cms = CountMinSketch(width=1000, depth=5)
        hashes1 = cms.hashes("this is a test")

        cms2 = CountMinSketch(width=1000, depth=5, hash_function=different_hash)
        hashes2 = cms2.hashes("this is a test")
        self.assertNotEqual(hashes1, hashes2)

    def test_cms_min_val(self):
        """test when we come to the bottom of the 32 bit int
        (stop overflow)"""
        too_large = INT64_T_MAX + 5
        cms = CountMinSketch(width=1000, depth=5)
        cms.remove("this is a test", too_large)
        self.assertEqual(cms.check("this is a test"), INT32_T_MIN)
        self.assertEqual(cms.elements_added, INT64_T_MIN)

    def test_cms_max_val(self):
        """test when we come to the top of the 32 bit int
        (stop overflow)"""
        too_large = INT64_T_MAX + 5
        cms = CountMinSketch(width=1000, depth=5)
        cms.add("this is a test", too_large)
        self.assertEqual(cms.check("this is a test"), INT32_T_MAX)
        self.assertEqual(cms.elements_added, INT64_T_MAX)

    def test_cms_clear(self):
        """test the clear functionality"""
        cms = CountMinSketch(width=1000, depth=5)
        self.assertEqual(cms.add("this is a test", 100), 100)
        self.assertEqual(cms.elements_added, 100)

        cms.clear()
        self.assertEqual(cms.elements_added, 0)
        self.assertEqual(cms.check("this is a test"), 0)

    def test_cms_str(self):
        """test the string representation of the count-min sketch"""
        cms = CountMinSketch(width=1000, depth=5)
        self.assertEqual(cms.add("this is a test", 100), 100)
        msg = (
            "Count-Min Sketch:\n"
            "\tWidth: 1000\n"
            "\tDepth: 5\n"
            "\tConfidence: 0.96875\n"
            "\tError Rate: 0.002\n"
            "\tElements Added: 100"
        )
        self.assertEqual(str(cms), msg)

    def test_cms_join(self):
        """test standard count-min sketch join"""
        cms1 = CountMinSketch(width=1000, depth=5)
        cms2 = CountMinSketch(width=1000, depth=5)

        self.assertEqual(255, cms1.add("this is a test", 255))
        self.assertEqual(189, cms1.add("this is another test", 189))
        self.assertEqual(16, cms1.add("this is also a test", 16))
        self.assertEqual(5, cms1.add("this is something to test", 5))

        self.assertEqual(255, cms2.add("this is a test", 255))
        self.assertEqual(189, cms2.add("this is another test", 189))
        self.assertEqual(16, cms2.add("this is also a test", 16))
        self.assertEqual(5, cms2.add("this is something to test", 5))

        cms1.join(cms2)
        self.assertEqual(255 * 2, cms1.check("this is a test"))
        self.assertEqual(189 * 2, cms1.check("this is another test"))
        self.assertEqual(16 * 2, cms1.check("this is also a test"))
        self.assertEqual(5 * 2, cms1.check("this is something to test"))
        self.assertEqual(cms1.elements_added, (255 + 189 + 16 + 5) * 2)

    def test_cms_join_overflow(self):
        """test count-min sketch overflow"""
        too_large = INT32_T_MAX + 5
        cms = CountMinSketch(width=1000, depth=5)
        cms.add("this is a test", too_large // 2)
        cms.join(cms)
        self.assertEqual(INT32_T_MAX, cms.check("this is a test"))
        self.assertEqual(cms.elements_added, too_large)

        cms.add("this is a test 2 ", INT64_T_MAX // 2)
        cms.join(cms)
        self.assertEqual(cms.elements_added, INT64_T_MAX)

    def test_cms_join_underflow(self):
        """test count-min sketch underflow"""
        too_large = INT32_T_MAX + 5
        cms = CountMinSketch(width=1000, depth=5)
        cms.remove("this is a test", too_large // 2)
        cms.join(cms)
        self.assertEqual(INT32_T_MIN, cms.check("this is a test"))
        self.assertEqual(cms.elements_added, -too_large)

        cms.remove("this is a test 2 ", INT64_T_MAX // 2)
        cms.join(cms)
        self.assertEqual(cms.elements_added, INT64_T_MIN)

    def test_cms_join_mixed_types(self):
        """test count-min, count-mean, and count-meanmin joining"""
        cms = CountMinSketch(width=1000, depth=5)
        cmeans = CountMeanSketch(width=1000, depth=5)
        cmms = CountMeanMinSketch(width=1000, depth=5)

        cms.add("this is a test", 500)
        cmeans.add("this is another test", 500)
        cmms.add("this is yet another test", 500)

        cms.join(cmeans)
        self.assertTrue("this is a test" in cms)
        self.assertTrue("this is another test" in cms)
        self.assertFalse("this is yet another test" in cms)

        cmeans.join(cmms)
        self.assertFalse("this is a test" in cmeans)
        self.assertTrue("this is another test" in cmeans)
        self.assertTrue("this is yet another test" in cmeans)
        self.assertFalse("foobar" in cmeans)

        cmms.join(cms)
        self.assertTrue("this is a test" in cmms)
        self.assertTrue("this is another test" in cmms)
        self.assertTrue("this is yet another test" in cmms)
        self.assertFalse("this is yet another test!" in cmms)

    def test_cms_join_mismatch_width(self):
        """test joining cms with mismatch width"""
        cms1 = CountMinSketch(width=1000, depth=5)
        cms2 = CountMinSketch(width=1001, depth=5)

        try:
            cms1.join(cms2)
        except CountMinSketchError as ex:
            msg = "Unable to merge as the count-min sketches are mismatched"
            self.assertEqual(ex.message, msg)
        else:
            self.assertEqual(True, False)

    def test_cms_join_mismatch_depth(self):
        """test joining cms with mismatch depth"""
        cms1 = CountMinSketch(width=1000, depth=5)
        cms2 = CountMinSketch(width=1000, depth=4)

        try:
            cms1.join(cms2)
        except CountMinSketchError as ex:
            msg = "Unable to merge as the count-min sketches are mismatched"
            self.assertEqual(ex.message, msg)
        else:
            self.assertEqual(True, False)

    def test_cms_mismatch_hash_function(self):
        """test joining when hash functions do not match"""
        cms1 = CountMinSketch(width=1000, depth=5)
        cms2 = CountMinSketch(width=1000, depth=5, hash_function=different_hash)

        def runner():
            """runner"""
            cms1.join(cms2)

        self.assertRaises(CountMinSketchError, runner)
        try:
            cms1.join(cms2)
        except CountMinSketchError as ex:
            msg = "Unable to merge as the count-min sketches are mismatched"
            self.assertEqual(ex.message, msg)
        else:
            self.assertEqual(True, False)

    def test_cms_join_invalid(self):
        """test joing a cms with an invalid type"""
        cms = CountMinSketch(width=1000, depth=5)

        try:
            cms.join(1)  # type: ignore
        except TypeError as ex:
            msg = "Unable to merge a count-min sketch with {}".format("<class 'int'>")
            self.assertEqual(str(ex), msg)
        else:
            self.assertEqual(True, False)

    def test_cms_invalid_width(self):
        """test invalid width"""

        def runner():
            """runner"""
            CountMinSketch(width=0, depth=5)

        self.assertRaises(InitializationError, runner)
        msg = "CountMinSketch: width and depth must be greater than 0"
        try:
            runner()
        except InitializationError as ex:
            self.assertEqual(str(ex), msg)
        else:
            self.assertEqual(True, False)

    def test_cms_invalid_depth(self):
        """test invalid width"""

        def runner():
            """runner"""
            CountMinSketch(width=1000, depth=-5)

        self.assertRaises(InitializationError, runner)
        msg = "CountMinSketch: width and depth must be greater than 0"
        try:
            runner()
        except InitializationError as ex:
            self.assertEqual(str(ex), msg)
        else:
            self.assertEqual(True, False)

    def test_cms_invalid_width_2(self):
        """test invalid width invalid type"""

        def runner():
            """runner"""
            CountMinSketch(width="0.0", depth=5)  # type: ignore

        self.assertRaises(InitializationError, runner)
        msg = "CountMinSketch: width and depth must be greater than 0"
        try:
            runner()
        except InitializationError as ex:
            self.assertEqual(str(ex), msg)
        else:
            self.assertEqual(True, False)

    def test_cms_invalid_depth_2(self):
        """test invalid depth type"""

        def runner():
            """runner"""
            CountMinSketch(width=1000, depth=[])  # type: ignore

        self.assertRaises(InitializationError, runner)
        msg = "CountMinSketch: width and depth must be greater than 0"
        try:
            runner()
        except InitializationError as ex:
            self.assertEqual(str(ex), msg)
        else:
            self.assertEqual(True, False)

    def test_cms_invalid_conf(self):
        """test invalid width"""

        def runner():
            """runner"""
            CountMinSketch(confidence=-3.0, error_rate=0.99)

        self.assertRaises(InitializationError, runner)
        msg = "CountMinSketch: width and depth must be greater than 0"
        try:
            runner()
        except InitializationError as ex:
            self.assertEqual(str(ex), msg)
        else:
            self.assertEqual(True, False)

    def test_cms_invalid_err_rate(self):
        """test invalid width"""

        def runner():
            """runner"""
            CountMinSketch(confidence=3.0, error_rate=0)

        self.assertRaises(InitializationError, runner)
        msg = "CountMinSketch: width and depth must be greater than 0"
        try:
            runner()
        except InitializationError as ex:
            self.assertEqual(str(ex), msg)
        else:
            self.assertEqual(True, False)

    def test_cms_invalid_conf_2(self):
        """test invalid width invalid type"""

        def runner():
            """runner"""
            CountMinSketch(confidence=3.0, error_rate="0.99")  # type: ignore

        self.assertRaises(InitializationError, runner)
        msg = "CountMinSketch: width and depth must be greater than 0"
        try:
            runner()
        except InitializationError as ex:
            self.assertEqual(str(ex), msg)
        else:
            self.assertEqual(True, False)

    def test_cms_invalid_err_rate_2(self):
        """test invalid error rate invalid type"""

        def runner():
            """runner"""
            CountMinSketch(width=1000, depth=[])  # type: ignore

        self.assertRaises(InitializationError, runner)
        msg = "CountMinSketch: width and depth must be greater than 0"
        try:
            runner()
        except InitializationError as ex:
            self.assertEqual(str(ex), msg)
        else:
            self.assertEqual(True, False)


class TestHeavyHitters(unittest.TestCase):
    """Test the default heavy hitters implementation"""

    def test_heavyhitters_init_wd(self):
        """test initializing heavy hitters"""
        hh1 = HeavyHitters(num_hitters=1000, width=1000, depth=5)
        self.assertEqual(hh1.width, 1000)
        self.assertEqual(hh1.depth, 5)
        self.assertEqual(hh1.confidence, 0.96875)
        self.assertEqual(hh1.error_rate, 0.002)
        self.assertEqual(hh1.elements_added, 0)
        self.assertEqual(hh1.heavy_hitters, dict())
        self.assertEqual(hh1.number_heavy_hitters, 1000)

    def test_heavyhitters_init_ce(self):
        """test initializing heavy hitters"""
        hh1 = HeavyHitters(num_hitters=1000, confidence=0.96875, error_rate=0.002)
        self.assertEqual(hh1.width, 1000)
        self.assertEqual(hh1.depth, 5)
        self.assertEqual(hh1.confidence, 0.96875)
        self.assertEqual(hh1.error_rate, 0.002)
        self.assertEqual(hh1.elements_added, 0)
        self.assertEqual(hh1.heavy_hitters, dict())
        self.assertEqual(hh1.number_heavy_hitters, 1000)

    def test_heavyhitters_add(self):
        """test adding things (singular) to the heavy hitters"""
        hh1 = HeavyHitters(num_hitters=2, width=1000, depth=5)
        self.assertEqual(hh1.add("this is a test"), 1)
        self.assertEqual(hh1.add("this is a test"), 2)
        self.assertEqual(hh1.add("this is a test"), 3)
        self.assertEqual(hh1.add("this is also a test"), 1)
        self.assertEqual(hh1.add("this is not a test"), 1)
        self.assertEqual(hh1.add("this is not a test"), 2)
        self.assertEqual(hh1.heavy_hitters, {"this is a test": 3, "this is not a test": 2})
        self.assertEqual(hh1.add("this is also a test"), 2)
        self.assertEqual(hh1.add("this is also a test"), 3)
        self.assertEqual(hh1.add("this is also a test"), 4)
        self.assertEqual(hh1.heavy_hitters, {"this is a test": 3, "this is also a test": 4})

    def test_heavyhitters_add_mult(self):
        """test adding things (multiple) to the heavy hitters"""
        hh1 = HeavyHitters(num_hitters=2, width=1000, depth=5)
        self.assertEqual(hh1.add("this is a test", 3), 3)
        self.assertEqual(hh1.add("this is also a test"), 1)
        self.assertEqual(hh1.add("this is not a test", 2), 2)
        self.assertEqual(hh1.heavy_hitters, {"this is a test": 3, "this is not a test": 2})
        self.assertEqual(hh1.add("this is also a test", 3), 4)
        self.assertEqual(hh1.heavy_hitters, {"this is a test": 3, "this is also a test": 4})
        self.assertEqual(hh1.add("this is not a test", 2), 4)
        self.assertEqual(hh1.add("this is not a test", 2), 6)
        self.assertEqual(hh1.add("this is not a test", 2), 8)
        self.assertEqual(hh1.add("this is not a test", 2), 10)
        self.assertEqual(hh1.heavy_hitters, {"this is not a test": 10, "this is also a test": 4})

    def test_hh_remove(self):
        """test remove from heavy hitters exception"""
        hh1 = HeavyHitters(num_hitters=2, width=1000, depth=5)
        self.assertEqual(hh1.add("this is a test", 3), 3)
        self.assertRaises(NotSupportedError, lambda: hh1.remove("this is a test"))

    def test_hh_remove_msg(self):
        """test remove from heavy hitters exception message"""
        hh1 = HeavyHitters(num_hitters=2, width=1000, depth=5)
        self.assertEqual(hh1.add("this is a test", 3), 3)
        try:
            hh1.remove("this is a test")
        except NotSupportedError as ex:
            msg = (
                "Unable to remove elements in the HeavyHitters "
                "class as it is an un supported action (and does not"
                "make sense)!"
            )
            self.assertEqual(str(ex), msg)
        else:
            self.assertEqual(True, False)

    def test_hh_clear(self):
        """test clearing out the heavy hitters object"""
        hh1 = HeavyHitters(num_hitters=1000, width=1000, depth=5)
        self.assertEqual(hh1.width, 1000)
        self.assertEqual(hh1.depth, 5)
        self.assertEqual(hh1.confidence, 0.96875)
        self.assertEqual(hh1.error_rate, 0.002)
        self.assertEqual(hh1.elements_added, 0)
        self.assertEqual(hh1.heavy_hitters, dict())
        self.assertEqual(hh1.number_heavy_hitters, 1000)

        self.assertEqual(hh1.add("this is a test", 3), 3)
        self.assertEqual(hh1.elements_added, 3)
        self.assertEqual(hh1.heavy_hitters, {"this is a test": 3})

        hh1.clear()
        self.assertEqual(hh1.elements_added, 0)
        self.assertEqual(hh1.heavy_hitters, dict())

    def test_hh_export(self):
        """test exporting a heavy hitters sketch"""
        md5_val = "fb1c39dd1a73f1ef0d7fc79f60fc028e"
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".cms", delete=DELETE_TEMP_FILES) as fobj:
            hh1 = HeavyHitters(num_hitters=1000, width=1000, depth=5)
            hh1.add("this is a test", 100)
            hh1.export(fobj.name)
            md5_out = calc_file_md5(fobj.name)
        self.assertEqual(md5_out, md5_val)

    def test_hh_load(self):
        """test loading a heavy hitters from file"""
        md5_val = "fb1c39dd1a73f1ef0d7fc79f60fc028e"
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".cms", delete=DELETE_TEMP_FILES) as fobj:
            hh1 = HeavyHitters(num_hitters=1000, width=1000, depth=5)
            self.assertEqual(hh1.add("this is a test", 100), 100)
            self.assertEqual(hh1.elements_added, 100)
            self.assertEqual(hh1.heavy_hitters, {"this is a test": 100})
            hh1.export(fobj.name)
            md5_out = calc_file_md5(fobj.name)
            self.assertEqual(md5_out, md5_val)

            # try loading directly to file!
            hh2 = HeavyHitters(num_hitters=1000, filepath=fobj.name)
            self.assertEqual(hh2.width, 1000)
            self.assertEqual(hh2.depth, 5)
            self.assertEqual(hh2.elements_added, 100)
            self.assertEqual(hh2.check("this is a test"), 100)
            # show on load that the tracking of heavy hitters is gone
            self.assertEqual(hh2.heavy_hitters, dict())
            self.assertEqual(hh2.add("this is a test", 1), 101)
            self.assertEqual(hh2.heavy_hitters, {"this is a test": 101})

    def test_hh_bytes(self):
        """test exporting a heavy hitters sketch as bytes"""
        md5_val = "fb1c39dd1a73f1ef0d7fc79f60fc028e"

        hh1 = HeavyHitters(num_hitters=1000, width=1000, depth=5)
        hh1.add("this is a test", 100)
        self.assertEqual(hashlib.md5(bytes(hh1)).hexdigest(), md5_val)

    def test_hh_frombytes(self):
        """testinitializing a heavy hitters from bytes"""
        hh1 = HeavyHitters(num_hitters=1000, width=1000, depth=5)
        hh1.add("this is a test", 100)
        bytes_out = bytes(hh1)

        hh2 = HeavyHitters.frombytes(bytes_out, num_hitters=500)
        self.assertEqual(hh2.width, 1000)
        self.assertEqual(hh2.depth, 5)
        self.assertEqual(hh2.number_heavy_hitters, 500)
        self.assertEqual(hh2.elements_added, 100)
        self.assertEqual(bytes(hh2), bytes(hh1))
        self.assertEqual(hh2.check("this is a test"), 100)

    def test_hh_str(self):
        """test the string representation of the heavy hitters sketch"""
        hh1 = HeavyHitters(num_hitters=2, width=1000, depth=5)
        self.assertEqual(hh1.add("this is a test", 100), 100)
        msg = (
            "Heavy Hitters Count-Min Sketch:\n"
            "\tWidth: 1000\n"
            "\tDepth: 5\n"
            "\tConfidence: 0.96875\n"
            "\tError Rate: 0.002\n"
            "\tElements Added: 100\n"
            "\tNumber Hitters: 2\n"
            "\tNumber Recorded: 1"
        )
        self.assertEqual(str(hh1), msg)

    def test_hh_join(self):
        """test that stream threshold raises exception"""
        hh1 = HeavyHitters(num_hitters=2, width=1000, depth=5)
        hh2 = HeavyHitters(num_hitters=2, width=1000, depth=5)
        self.assertRaises(NotSupportedError, lambda: hh1.join(hh2))


class TestCountMeanSketch(unittest.TestCase):
    """test the basic count-mean sketch"""

    def test_default_count_mean_query(self):
        """test the default query of the count-mean sketch"""
        cms = CountMeanSketch(width=1000, depth=5)
        self.assertEqual(cms.query_type, "mean")


class TestCountMeanMinSketch(unittest.TestCase):
    """test the basic count-mean-min sketch"""

    def test_def_count_mean_min_query(self):
        """test the default query of the count-mean-min sketch"""
        cms = CountMeanMinSketch(width=1000, depth=5)
        self.assertEqual(cms.query_type, "mean-min")


class TestStreamThreshold(unittest.TestCase):
    """Test the default stream threshold implementation"""

    def test_streamthreshold_init_wd(self):
        """test initializing the stream threshold using width and depth"""
        st1 = StreamThreshold(threshold=1000, width=1000, depth=5)
        self.assertEqual(st1.width, 1000)
        self.assertEqual(st1.depth, 5)
        self.assertEqual(st1.confidence, 0.96875)
        self.assertEqual(st1.error_rate, 0.002)
        self.assertEqual(st1.elements_added, 0)
        self.assertEqual(st1.meets_threshold, dict())
        self.assertEqual(st1.threshold, 1000)

    def test_streamthreshold_init_ec(self):
        """test initializing the stream threshold using error rate and
        confidence"""
        st1 = StreamThreshold(threshold=1000, confidence=0.96875, error_rate=0.002)
        self.assertEqual(st1.width, 1000)
        self.assertEqual(st1.depth, 5)
        self.assertEqual(st1.confidence, 0.96875)
        self.assertEqual(st1.error_rate, 0.002)
        self.assertEqual(st1.elements_added, 0)
        self.assertEqual(st1.meets_threshold, dict())
        self.assertEqual(st1.threshold, 1000)

    def test_streamthreshold_add(self):
        """test adding elements to the stream threshold in singular"""
        st1 = StreamThreshold(threshold=2, width=1000, depth=5)
        self.assertEqual(st1.add("this is a test"), 1)
        self.assertEqual(st1.meets_threshold, dict())
        self.assertEqual(st1.add("this is a test"), 2)
        self.assertEqual(st1.meets_threshold, {"this is a test": 2})
        self.assertEqual(st1.add("this is not a test"), 1)
        self.assertEqual(st1.meets_threshold, {"this is a test": 2})
        self.assertEqual(st1.add("this is a test"), 3)
        self.assertEqual(st1.meets_threshold, {"this is a test": 3})
        self.assertEqual(st1.add("this is not a test"), 2)
        self.assertEqual(st1.add("this is still not a test"), 1)
        self.assertEqual(st1.meets_threshold, {"this is a test": 3, "this is not a test": 2})
        self.assertEqual(st1.elements_added, 6)

    def test_streamthreshold_add_mult(self):
        """test adding elements to the stream threshold in multiple"""
        st1 = StreamThreshold(threshold=10, width=1000, depth=5)
        self.assertEqual(st1.add("this is a test", 5), 5)
        self.assertEqual(st1.meets_threshold, dict())
        self.assertEqual(st1.add("this is a test", 5), 10)
        self.assertEqual(st1.meets_threshold, {"this is a test": 10})
        self.assertEqual(st1.add("this is not a test", 9), 9)
        self.assertEqual(st1.meets_threshold, {"this is a test": 10})
        self.assertEqual(st1.add("this is a test", 20), 30)
        self.assertEqual(st1.meets_threshold, {"this is a test": 30})
        self.assertEqual(st1.add("this is not a test", 2), 11)
        self.assertEqual(st1.meets_threshold, {"this is a test": 30, "this is not a test": 11})
        self.assertEqual(st1.elements_added, 41)

    def test_streamthreshold_clear(self):
        """test clearing the stream threshold"""
        st1 = StreamThreshold(threshold=10, width=1000, depth=5)
        self.assertEqual(st1.add("this is a test", 5), 5)
        self.assertEqual(st1.meets_threshold, dict())
        self.assertEqual(st1.add("this is a test", 5), 10)
        self.assertEqual(st1.meets_threshold, {"this is a test": 10})
        self.assertEqual(st1.add("this is not a test", 9), 9)
        self.assertEqual(st1.meets_threshold, {"this is a test": 10})
        self.assertEqual(st1.add("this is a test", 20), 30)
        self.assertEqual(st1.meets_threshold, {"this is a test": 30})
        self.assertEqual(st1.add("this is not a test", 2), 11)
        self.assertEqual(st1.meets_threshold, {"this is a test": 30, "this is not a test": 11})
        self.assertEqual(st1.elements_added, 41)

        st1.clear()
        self.assertEqual(st1.meets_threshold, dict())
        self.assertEqual(st1.elements_added, 0)

    def test_streamthreshold_remove(self):
        """test removing elements from the stream threshold singular"""
        st1 = StreamThreshold(threshold=10, width=1000, depth=5)
        self.assertEqual(st1.add("this is a test", 5), 5)
        self.assertEqual(st1.meets_threshold, dict())
        self.assertEqual(st1.add("this is a test", 5), 10)
        self.assertEqual(st1.meets_threshold, {"this is a test": 10})
        self.assertEqual(st1.add("this is not a test", 9), 9)
        self.assertEqual(st1.meets_threshold, {"this is a test": 10})
        self.assertEqual(st1.add("this is a test", 20), 30)
        self.assertEqual(st1.meets_threshold, {"this is a test": 30})
        self.assertEqual(st1.add("this is not a test", 2), 11)
        self.assertEqual(st1.meets_threshold, {"this is a test": 30, "this is not a test": 11})
        self.assertEqual(st1.remove("this is a test"), 29)
        self.assertEqual(st1.meets_threshold, {"this is a test": 29, "this is not a test": 11})
        self.assertEqual(st1.remove("this is not a test"), 10)
        self.assertEqual(st1.remove("this is not a test"), 9)
        self.assertEqual(st1.meets_threshold, {"this is a test": 29})

        self.assertEqual(st1.elements_added, 38)

    def test_streamthreshold_rem_mult(self):
        """test adding elements to the stream threshold in multiple"""
        st1 = StreamThreshold(threshold=10, width=1000, depth=5)
        self.assertEqual(st1.add("this is a test", 30), 30)
        self.assertEqual(st1.add("this is not a test", 11), 11)
        self.assertEqual(st1.meets_threshold, {"this is a test": 30, "this is not a test": 11})
        self.assertEqual(st1.elements_added, 41)
        self.assertEqual(st1.remove("this is not a test", 2), 9)
        self.assertEqual(st1.meets_threshold, {"this is a test": 30})
        self.assertEqual(st1.elements_added, 39)

    def test_streamthreshold_export(self):
        """test exporting a stream threshold sketch"""
        md5_val = "fb1c39dd1a73f1ef0d7fc79f60fc028e"
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".cms", delete=DELETE_TEMP_FILES) as fobj:
            st1 = StreamThreshold(threshold=10, width=1000, depth=5)
            st1.add("this is a test", 100)
            st1.export(fobj.name)
            md5_out = calc_file_md5(fobj.name)
        self.assertEqual(md5_out, md5_val)

    def test_streamthreshold_load(self):
        """test loading a stream threshold sketch from file"""
        md5_val = "fb1c39dd1a73f1ef0d7fc79f60fc028e"
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".cms", delete=DELETE_TEMP_FILES) as fobj:
            st1 = StreamThreshold(threshold=10, width=1000, depth=5)
            self.assertEqual(st1.add("this is a test", 100), 100)
            self.assertEqual(st1.elements_added, 100)
            self.assertEqual(st1.meets_threshold, {"this is a test": 100})
            st1.export(fobj.name)
            md5_out = calc_file_md5(fobj.name)
            self.assertEqual(md5_out, md5_val)

            # try loading directly to file!
            st2 = StreamThreshold(threshold=10, filepath=fobj.name)
            self.assertEqual(st2.width, 1000)
            self.assertEqual(st2.depth, 5)
            self.assertEqual(st2.elements_added, 100)
            self.assertEqual(st2.check("this is a test"), 100)
            # show on load that the tracking of stream threshold is gone
            self.assertEqual(st2.meets_threshold, dict())
            self.assertEqual(st2.add("this is a test", 1), 101)
            self.assertEqual(st2.meets_threshold, {"this is a test": 101})

    def test_streamthreshold_bytes(self):
        """test exporting a streaming threshold as bytes"""
        md5_val = "fb1c39dd1a73f1ef0d7fc79f60fc028e"
        st1 = StreamThreshold(threshold=10, width=1000, depth=5)
        st1.add("this is a test", 100)
        bytes_out = bytes(st1)
        self.assertEqual(hashlib.md5(bytes_out).hexdigest(), md5_val)

    def test_streamthreshold_frombytes(self):
        """testinitializing a heavy hitters from bytes"""
        st1 = StreamThreshold(threshold=10, width=1000, depth=5)
        st1.add("this is a test", 100)
        bytes_out = bytes(st1)

        st2 = StreamThreshold.frombytes(bytes_out, threshold=10)
        self.assertEqual(st2.width, 1000)
        self.assertEqual(st2.depth, 5)
        self.assertEqual(st2.threshold, 10)
        self.assertEqual(st2.elements_added, 100)
        self.assertEqual(bytes(st2), bytes(st1))
        self.assertEqual(st2.check("this is a test"), 100)
        # show on load that the tracking of stream threshold is gone
        self.assertEqual(st2.meets_threshold, dict())
        self.assertEqual(st2.add("this is a test", 1), 101)
        self.assertEqual(st2.meets_threshold, {"this is a test": 101})

    def test_streamthreshold_str(self):
        """test the string representation of the stream threshold sketch"""
        st1 = StreamThreshold(threshold=10, width=1000, depth=5)
        self.assertEqual(st1.add("this is a test", 100), 100)
        msg = (
            "Stream Threshold Count-Min Sketch:\n"
            "\tWidth: 1000\n"
            "\tDepth: 5\n"
            "\tConfidence: 0.96875\n"
            "\tError Rate: 0.002\n"
            "\tElements Added: 100\n"
            "\tThreshold: 10\n"
            "\tNumber Meeting Threshold: 1"
        )
        self.assertEqual(str(st1), msg)

    def test_streamthreshold_join(self):
        """test that stream threshold raises exception"""
        st1 = StreamThreshold(threshold=1000, width=1000, depth=5)
        st2 = StreamThreshold(threshold=1000, width=1000, depth=5)
        self.assertRaises(NotSupportedError, lambda: st1.join(st2))


class TestConservativeUpdate(unittest.TestCase):
    """Test the conservative update policy"""

    def test_default_policy_is_standard(self):
        """the default update policy remains standard"""
        cms = CountMinSketch(width=1000, depth=5)
        self.assertEqual(cms.update_policy, UPDATE_POLICY_STANDARD)

    def test_invalid_policy(self):
        """unknown update policies are rejected"""
        self.assertRaises(InitializationError, lambda: CountMinSketch(width=1000, depth=5, update_policy="nope"))
        cms = CountMinSketch(width=1000, depth=5)
        self.assertRaises(InitializationError, setattr, cms, "update_policy", "nope")

    def test_policy_setter(self):
        """the policy can be switched after construction"""
        cms = CountMinSketch(width=1000, depth=5)
        cms.update_policy = UPDATE_POLICY_CONSERVATIVE
        self.assertEqual(cms.update_policy, UPDATE_POLICY_CONSERVATIVE)
        cms.update_policy = UPDATE_POLICY_STANDARD
        self.assertEqual(cms.update_policy, UPDATE_POLICY_STANDARD)

    def test_batched_matches_sequential(self):
        """a compressed adjacent run must match one-by-one inserts exactly"""
        rng = random.Random(42)
        for _ in range(25):
            width = rng.randint(4, 40)
            depth = rng.randint(2, 7)
            sequential = CountMinSketch(width=width, depth=depth, update_policy=UPDATE_POLICY_CONSERVATIVE)
            batched = CountMinSketch(width=width, depth=depth, update_policy=UPDATE_POLICY_CONSERVATIVE)
            words = [f"w{rng.randint(0, 15)}" for _ in range(80)]

            runs = []
            for word in words:
                if runs and runs[-1][0] == word:
                    runs[-1] = (word, runs[-1][1] + 1)
                else:
                    runs.append((word, 1))

            for word in words:
                sequential.add(word)
            for word, count in runs:
                batched.add(word, count)

            self.assertEqual(sequential._bins, batched._bins)
            self.assertEqual(sequential.elements_added, batched.elements_added)
            for word, _ in runs:
                self.assertEqual(sequential.check(word), batched.check(word))

    def test_large_compressed_count(self):
        """the cost and result of a huge compressed count do not depend on n"""
        batched = CountMinSketch(width=1000, depth=5, update_policy=UPDATE_POLICY_CONSERVATIVE)
        self.assertEqual(batched.add("hot-word", 1_000_000), 1_000_000)

        sequential = CountMinSketch(width=1000, depth=5, update_policy=UPDATE_POLICY_CONSERVATIVE)
        for _ in range(1000):
            sequential.add("hot-word")
        sequential.add("hot-word", 999_000)

        self.assertEqual(batched._bins, sequential._bins)
        self.assertEqual(batched.check("hot-word"), 1_000_000)

    def test_conservative_never_exceeds_standard(self):
        """conservative bins stay at or below the standard bins and avoid
        inflating a bin already owned by a heavier key"""
        standard = CountMinSketch(width=2, depth=2, hash_function=controlled_hash)
        conservative = CountMinSketch(
            width=2, depth=2, hash_function=controlled_hash, update_policy=UPDATE_POLICY_CONSERVATIVE
        )
        # x owns bins {0, 2}; the lighter y shares bin 0 but also owns bin 3
        for sketch in (standard, conservative):
            sketch.add("x", 100)
            sketch.add("y", 5)

        for std_bin, con_bin in zip(standard._bins, conservative._bins, strict=True):
            self.assertLessEqual(con_bin, std_bin)
        # the shared bin is spared the extra +5 under conservative update
        self.assertEqual(standard._bins[0], 105)
        self.assertEqual(conservative._bins[0], 100)
        self.assertEqual(conservative.check("y"), 5)

    def test_overflow_saturates(self):
        """saturation still applies to conservative updates"""
        cms = CountMinSketch(width=1000, depth=5, update_policy=UPDATE_POLICY_CONSERVATIVE)
        cms.add("this is a test", INT64_T_MAX + 5)
        self.assertEqual(cms.check("this is a test"), INT32_T_MAX)
        self.assertEqual(cms.elements_added, INT64_T_MAX)

    def test_bytes_round_trip_preserves_policy(self):
        """saving and restoring keeps the update policy"""
        conservative = CountMinSketch(width=16, depth=4, update_policy=UPDATE_POLICY_CONSERVATIVE)
        conservative.add("this is a test", 100)
        restored = CountMinSketch.frombytes(bytes(conservative))
        self.assertEqual(restored.update_policy, UPDATE_POLICY_CONSERVATIVE)
        self.assertEqual(restored._bins, conservative._bins)
        self.assertEqual(bytes(restored), bytes(conservative))

    def test_standard_keeps_legacy_byte_layout(self):
        """standard sketches remain byte-for-byte compatible with old files"""
        md5_val = "fb1c39dd1a73f1ef0d7fc79f60fc028e"
        cms = CountMinSketch(width=1000, depth=5)
        cms.add("this is a test", 100)
        self.assertEqual(hashlib.md5(bytes(cms)).hexdigest(), md5_val)
        restored = CountMinSketch.frombytes(bytes(cms))
        self.assertEqual(restored.update_policy, UPDATE_POLICY_STANDARD)

    def test_file_round_trip_preserves_policy(self):
        """file export and load keep the update policy"""
        with NamedTemporaryFile(dir=os.getcwd(), suffix=".cms", delete=DELETE_TEMP_FILES) as fobj:
            cms = CountMinSketch(width=1000, depth=5, update_policy=UPDATE_POLICY_CONSERVATIVE)
            cms.add("this is a test", 100)
            cms.export(fobj.name)

            loaded = CountMinSketch(filepath=fobj.name)
            self.assertEqual(loaded.update_policy, UPDATE_POLICY_CONSERVATIVE)
            self.assertEqual(loaded.check("this is a test"), 100)

    def test_subclass_round_trip_preserves_policy(self):
        """heavy hitters and stream threshold keep the policy on load"""
        hh = HeavyHitters(num_hitters=5, width=16, depth=4, update_policy=UPDATE_POLICY_CONSERVATIVE)
        hh.add("this is a test", 100)
        self.assertEqual(HeavyHitters.frombytes(bytes(hh)).update_policy, UPDATE_POLICY_CONSERVATIVE)

        st = StreamThreshold(threshold=10, width=16, depth=4, update_policy=UPDATE_POLICY_CONSERVATIVE)
        st.add("this is a test", 100)
        self.assertEqual(StreamThreshold.frombytes(bytes(st), threshold=10).update_policy, UPDATE_POLICY_CONSERVATIVE)


class TestShuffleSafety(unittest.TestCase):
    """Test the two-batch interleaving shuffle safety analysis"""

    @staticmethod
    def _all_interleavings(first, second):
        """brute force every order preserving each source's ordering"""
        n_first = len(first)
        orders = []
        for picks in combinations(range(n_first + len(second)), n_first):
            picks = set(picks)
            order = []
            i = j = 0
            for pos in range(n_first + len(second)):
                if pos in picks:
                    order.append(first[i])
                    i += 1
                else:
                    order.append(second[j])
                    j += 1
            orders.append(order)
        return orders

    @staticmethod
    def _replay(policy, order, width=2, depth=2):
        cms = CountMinSketch(width=width, depth=depth, hash_function=controlled_hash, update_policy=policy)
        for key in order:
            cms.add(key)
        return tuple(cms._bins)

    def test_identical_batches_are_consistent(self):
        """the same ordered batch from both sources is always safe"""
        cms = CountMinSketch(width=8, depth=3)
        report = cms.check_shuffle_safety(["a", "b", "c"], ["a", "b", "c"])
        self.assertTrue(report.consistent)
        self.assertTrue(bool(report))
        self.assertIsNone(report.first_witness)
        self.assertEqual(report.bin_differences, {})

    def test_standard_policy_is_always_consistent(self):
        """plain count-min addition is commutative across the bins"""
        cms = CountMinSketch(width=4, depth=2)
        report = cms.check_shuffle_safety(["a", "b", "a"], ["b", "a", "b"])
        self.assertTrue(report.consistent)

    def test_conservative_detects_inconsistency(self):
        """an order-sensitive conservative stream is flagged with witnesses"""
        cms = CountMinSketch(width=2, depth=2, hash_function=controlled_hash, update_policy=UPDATE_POLICY_CONSERVATIVE)
        report = cms.check_shuffle_safety(["y", "z"], ["x"], query_keys=["x", "y", "z", "q"])

        self.assertFalse(report.consistent)
        self.assertFalse(bool(report))
        self.assertNotEqual(report.first_witness, report.second_witness)
        self.assertIn(report.first_affected_bin, report.bin_differences)

        first_state = self._replay(UPDATE_POLICY_CONSERVATIVE, report.first_witness)
        second_state = self._replay(UPDATE_POLICY_CONSERVATIVE, report.second_witness)
        self.assertNotEqual(first_state, second_state)
        self.assertEqual(
            report.bin_differences,
            {idx: (a, b) for idx, (a, b) in enumerate(zip(first_state, second_state, strict=True)) if a != b},
        )
        self.assertEqual(report.first_affected_bin, min(report.bin_differences))
        self.assertIn("x", report.changed_queries)
        self.assertNotIn("q", report.changed_queries)

    def test_witnesses_are_reachable_interleavings(self):
        """both reported orders preserve each source's ordering"""
        cms = CountMinSketch(width=2, depth=2, hash_function=controlled_hash, update_policy=UPDATE_POLICY_CONSERVATIVE)
        first = ["y", "z"]
        second = ["x"]
        report = cms.check_shuffle_safety(first, second)
        valid = {tuple(order) for order in self._all_interleavings(first, second)}
        self.assertIn(tuple(report.first_witness), valid)
        self.assertIn(tuple(report.second_witness), valid)

    def test_bin_differences_retained_without_query_hits(self):
        """bucket-level divergence is reported even with no query impact"""
        cms = CountMinSketch(width=2, depth=2, hash_function=controlled_hash, update_policy=UPDATE_POLICY_CONSERVATIVE)
        report = cms.check_shuffle_safety(["y", "z"], ["x"], query_keys=["q"])
        self.assertFalse(report.consistent)
        self.assertEqual(report.changed_queries, {})
        self.assertGreater(len(report.bin_differences), 0)

    def test_matches_brute_force(self):
        """the analyzer verdict matches exhaustive interleaving enumeration"""
        rng = random.Random(7)
        for policy in (UPDATE_POLICY_STANDARD, UPDATE_POLICY_CONSERVATIVE):
            for _ in range(30):
                first = [rng.choice("xyz") for _ in range(rng.randint(0, 4))]
                second = [rng.choice("xyz") for _ in range(rng.randint(0, 4))]
                cms = CountMinSketch(width=2, depth=2, hash_function=controlled_hash, update_policy=policy)
                report = cms.check_shuffle_safety(first, second)
                states = {self._replay(policy, order) for order in self._all_interleavings(first, second)}
                self.assertEqual(report.consistent, len(states) == 1)

    def test_starting_sketch_is_used_and_untouched(self):
        """replay starts from the current bins and never mutates the sketch"""
        cms = CountMinSketch(width=2, depth=2, hash_function=controlled_hash, update_policy=UPDATE_POLICY_CONSERVATIVE)
        cms.add("x", 3)
        starting_bins = tuple(cms._bins)

        report = cms.check_shuffle_safety(["y", "z"], ["y"])
        self.assertEqual(tuple(cms._bins), starting_bins)
        self.assertEqual(cms.elements_added, 3)

        orders = self._all_interleavings(["y", "z"], ["y"])
        states = set()
        for order in orders:
            seeded = CountMinSketch(
                width=2, depth=2, hash_function=controlled_hash, update_policy=UPDATE_POLICY_CONSERVATIVE
            )
            seeded.add("x", 3)
            for key in order:
                seeded.add(key)
            states.add(tuple(seeded._bins))
        self.assertEqual(report.consistent, len(states) == 1)

    def test_window_large_enough_matches_full_analysis(self):
        """a window covering both batches is equivalent to no window"""
        cms = CountMinSketch(width=2, depth=2, hash_function=controlled_hash, update_policy=UPDATE_POLICY_CONSERVATIVE)
        full = cms.check_shuffle_safety(["y", "z", "y"], ["x", "x"])
        windowed = cms.check_shuffle_safety(["y", "z", "y"], ["x", "x"], window=10)
        self.assertEqual(windowed.consistent, full.consistent)

    def test_small_window_drains_exhausted_source(self):
        """a window of one still explores every boundary choice and drains a
        source once it runs out, rather than collapsing to a single order"""
        first = ["z", "y", "z", "x", "x"]
        second = ["x", "y", "y"]

        def tagged_interleavings():
            n_first = len(first)
            for picks in combinations(range(n_first + len(second)), n_first):
                picks = set(picks)
                tagged = []
                i = j = 0
                for pos in range(n_first + len(second)):
                    if pos in picks:
                        tagged.append((0, first[i]))
                        i += 1
                    else:
                        tagged.append((1, second[j]))
                        j += 1
                yield tagged

        def within_window(tagged, window):
            i = j = 0
            for source, _ in tagged:
                both = i < len(first) and j < len(second)
                if both and source == 1 and min(i, j) >= window:
                    return False
                if source == 0:
                    i += 1
                else:
                    j += 1
            return True

        cms = CountMinSketch(width=2, depth=2, hash_function=controlled_hash, update_policy=UPDATE_POLICY_CONSERVATIVE)
        report = cms.check_shuffle_safety(first, second, window=1)

        in_window_states = {
            self._replay(UPDATE_POLICY_CONSERVATIVE, [key for _, key in tagged])
            for tagged in tagged_interleavings()
            if within_window(tagged, 1)
        }
        self.assertGreater(len(in_window_states), 1)
        self.assertEqual(report.consistent, len(in_window_states) == 1)
        # the full analysis must remain order-sensitive for this stream
        self.assertFalse(cms.check_shuffle_safety(first, second).consistent)

    def test_invalid_window(self):
        """non-positive windows are rejected"""
        cms = CountMinSketch(width=8, depth=3)
        self.assertRaises(InitializationError, lambda: cms.check_shuffle_safety(["a"], ["b"], window=0))
        self.assertRaises(InitializationError, lambda: cms.check_shuffle_safety(["a"], ["b"], window=-2))

    def test_empty_batches(self):
        """empty sources are trivially consistent"""
        cms = CountMinSketch(width=2, depth=2, hash_function=controlled_hash, update_policy=UPDATE_POLICY_CONSERVATIVE)
        self.assertTrue(cms.check_shuffle_safety([], ["x", "y"]).consistent)
        self.assertTrue(cms.check_shuffle_safety(["x"], []).consistent)
        self.assertTrue(cms.check_shuffle_safety([], []).consistent)

    def test_replay_does_not_touch_subclass_trackers(self):
        """heavy hitters and stream threshold state survive an analysis"""
        hh = HeavyHitters(
            num_hitters=5,
            width=2,
            depth=2,
            hash_function=controlled_hash,
            update_policy=UPDATE_POLICY_CONSERVATIVE,
        )
        hh.add("x", 3)
        hh.check_shuffle_safety(["y", "z"], ["x"])
        self.assertEqual(hh.heavy_hitters, {"x": 3})
        self.assertEqual(hh.elements_added, 3)

        st = StreamThreshold(
            threshold=2,
            width=2,
            depth=2,
            hash_function=controlled_hash,
            update_policy=UPDATE_POLICY_CONSERVATIVE,
        )
        st.add("y", 2)
        st.check_shuffle_safety(["y", "z"], ["x"])
        self.assertEqual(st.meets_threshold, {"y": 2})
        self.assertEqual(st.elements_added, 2)

    def test_query_estimates_honor_query_method(self):
        """changed estimates follow the sketch's configured query method"""
        cms = CountMinSketch(width=2, depth=2, hash_function=controlled_hash, update_policy=UPDATE_POLICY_CONSERVATIVE)
        cms.query_type = "mean"
        report = cms.check_shuffle_safety(["y", "z"], ["x"], query_keys=["x", "y", "z"])
        self.assertFalse(report.consistent)
        for key, (first_estimate, second_estimate) in report.changed_queries.items():
            first_sketch = CountMinSketch(
                width=2, depth=2, hash_function=controlled_hash, update_policy=UPDATE_POLICY_CONSERVATIVE
            )
            second_sketch = CountMinSketch(
                width=2, depth=2, hash_function=controlled_hash, update_policy=UPDATE_POLICY_CONSERVATIVE
            )
            first_sketch.query_type = "mean"
            second_sketch.query_type = "mean"
            for word in report.first_witness:
                first_sketch.add(word)
            for word in report.second_witness:
                second_sketch.add(word)
            self.assertEqual(first_estimate, first_sketch.check(key))
            self.assertEqual(second_estimate, second_sketch.check(key))

    def test_estimates_honor_mean_min_query(self):
        """mean-min estimates use the post-merge element total"""
        cms = CountMinSketch(width=2, depth=2, hash_function=controlled_hash, update_policy=UPDATE_POLICY_CONSERVATIVE)
        cms.query_type = "mean-min"
        report = cms.check_shuffle_safety(["y", "z"], ["x"], query_keys=["x", "y", "z"])
        self.assertFalse(report.consistent)
        for key, (first_estimate, second_estimate) in report.changed_queries.items():
            first_sketch = CountMinSketch(
                width=2, depth=2, hash_function=controlled_hash, update_policy=UPDATE_POLICY_CONSERVATIVE
            )
            second_sketch = CountMinSketch(
                width=2, depth=2, hash_function=controlled_hash, update_policy=UPDATE_POLICY_CONSERVATIVE
            )
            first_sketch.query_type = "mean-min"
            second_sketch.query_type = "mean-min"
            for word in report.first_witness:
                first_sketch.add(word)
            for word in report.second_witness:
                second_sketch.add(word)
            self.assertEqual(first_estimate, first_sketch.check(key))
            self.assertEqual(second_estimate, second_sketch.check(key))
        # the analyzer did not leak the hypothetical element total
        self.assertEqual(cms.elements_added, 0)


if __name__ == "__main__":
    unittest.main()
