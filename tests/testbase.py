'''
Bryan Bonvallet
2013

This file contains test functions for series.py and PMF.py.
'''

import numpy
import unittest

from PMF import PMF
from series import FiniteSequence, InfiniteSequence

class FinitePMF(FiniteSequence, PMF): pass

class InfinitePMF(InfiniteSequence, PMF): pass

class TestSeriesPMF(unittest.TestCase):
    # Test functions in series.py

    def _build_obj(self, cls, length = None):
        if length is None:
            length = numpy.random.randint(10,100)
        # Generate valid uniform distribution 
        dist = numpy.zeros( (2,length) )
        dist[0,:] = numpy.arange(1, 1+length)
        dist[1,:] = numpy.ones( (1,length) ) / float(length)
        return cls(dist)

    def _build_finite_obj(self, length=None):
        return self._build_obj(FinitePMF, length)
    def _build_infinite_obj(self, length=None):
        return self._build_obj(InfinitePMF, length)

    def test_build_funcs(self):
        # Test that build functions return expected objects.
        obj = self._build_finite_obj()
        self.assertIsInstance(obj, PMF)
        self.assertIsInstance(obj, FiniteSequence)
        obj = self._build_infinite_obj()
        self.assertIsInstance(obj, PMF)
        self.assertIsInstance(obj, InfiniteSequence)
