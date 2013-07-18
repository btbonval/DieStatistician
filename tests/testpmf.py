'''
Bryan Bonvallet
2013

This file tests functionality of PMF.py.
'''

import numpy

from testbase import FinitePMF
from testbase import InfinitePMF
from testbase import TestSeriesPMF

class TestPMF(TestSeriesPMF):
    # Test functions in PMF.py
    repeat = 10

    def test_bad_construction_dist(self):
        # Test a bad distribution passed into the constructor
        for dist in (
                      # No distribution.
                      None,
                      # Strings
                      numpy.array([['a','b'],['0.5','0.5']]),
                      # One dimensional
                      numpy.ones( (1,10) ) / 10.,
                      # Does not add up to 1.
                      numpy.zeros( (2,10) ),
                    ):
            self.assertRaises(TypeError, FinitePMF, dist)
            self.assertRaises(TypeError, InfinitePMF, dist)
