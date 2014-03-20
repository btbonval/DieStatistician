'''
Bryan Bonvallet
2013

This file tests functionality of series.py (and PMF.py).
'''

from testbase import FinitePMF
from testbase import InfinitePMF
from testbase import TestSeriesPMF

class TestSeries(TestSeriesPMF):
    # Test functions in series.py

    def test_finite_str(self):
        # Test that a string is returned.
        obj = self._build_finite_obj()
        test = str(obj)
        self.assertTrue(isinstance(test, str) or isinstance(test, unicode))

    def test_infinite_str(self):
        # Test that a string is returned.
        obj = self._build_infinite_obj()
        test = str(obj)
        self.assertTrue(isinstance(test, str) or isinstance(test, unicode))

    def test_infinite_str_maxterms(self):
        # Test that maxterms is respected
        obj = self._build_infinite_obj()
        self.assertTrue(hasattr(obj, 'maxterms'))
        terms = obj.maxterms

        # Triple the number of terms to guarantee it won't display them all.
        obj = self._build_infinite_obj(terms*3)
        test = str(obj)

        # This test makes strong assumptions about the string representation.
        # Assume terms are listed in rows
        # Grab the first row of terms between square brackets
        nums = test.split('[')[-1].split(']')[0]
        # Assume numbers are white space separated
        nums = nums.split()
        # There should be #terms not #terms*3
        self.assertEqual(len(nums), terms)
