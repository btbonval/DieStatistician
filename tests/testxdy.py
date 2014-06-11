'''
Bryan Bonvallet
2014

This contains test functions for XdY
'''

import numpy
import unittest

from XdY import XdY

class testxdy(unittest.TestCase):
    # Runs through some test cases to check expected behavior.

    def _get_error(self):
        return XdY.error

    def _equals(self, lhs, rhs):
        return self.assertAlmostEquals(lhs, rhs, delta=self._get_error())

    def _build_xdy(self, x, y):
        return XdY( (x,y) )

    def _build_2d6_dist(self):
        dist = numpy.array( ((2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                             (1, 2, 3, 4, 5, 6, 5, 4,  3,  2,  1)) )
        dist = dist.astype(numpy.float)
        dist[1,:] /= 36.
        return dist

    def testgeterror(self):
        # It is assumed that error is a certain value.
        self._equals(self._get_error(), XdY( (1,2) ).getError())

    def testaddition(self):
        # Build 2d6 in two ways and ensure the results are correct.
        # 2d6 directly
        a2d6 = self._build_xdy(2,6)

        # 2d6 via addition of 1d6s
        b1d6 = self._build_xdy(1,6)
        b2d6 = b1d6 + b1d6

        # actual distribution for 2d6
        dist = self._build_2d6_dist()

        error = self._get_error()
        for i in range(0,2):
            for j in range(0,len(dist)):
                # check 2d6 = 1d6 + 1d6
                self._equals(a2d6[i,j], b2d6[i,j])
                # check 2d6 is correct
                self._equals(a2d6[i,j], dist[i,j])

    def testexpectedvalue(self):
        # Build some XdY cases with known expected value and test the result.
        self._equals(self._build_xdy(1,6).EV(), 3.5)
        self._equals(self._build_xdy(1,12).EV(), 6.5)
        self._equals(self._build_xdy(4,4).EV(), 10)

    def testscalarcomparison(self):
        # Compare distribution against scalar values.

        a1d20 = self._build_xdy(1,20)

        # 1d20 is 50/50 less than 11, greater than 10, etc.
        self._equals(a1d20 < 11, 0.50)
        self._equals(11 > a1d20, 0.50)
        self._equals(a1d20 > 10, 0.50)
        self._equals(10 < a1d20, 0.50)

        self._equals(a1d20 <= 10, 0.50)
        self._equals(10 >= a1d20, 0.50)
        self._equals(a1d20 >= 11, 0.50)
        self._equals(11 <= a1d20, 0.50)

        self._equals(1 <= a1d20 <= 10, 0.50)
        self._equals(10 >= a1d20 >= 1, 0.50) # fail
        self._equals(11 <= a1d20 <= 20, 0.50) # fail
        self._equals(20 >= a1d20 >= 11, 0.50)

        # 1d20 has 1/20 probability of any particular value
        self._equals(a1d20 == 10, 0.05)
        self._equals(9 == a1d20, 0.05)

        self._equals(9 < a1d20 < 11, 0.05) # fail
        self._equals(1 <= a1d20 <= 1, 0.05)
        self._equals(11 > a1d20 > 9, 0.05) # fail
        self._equals(1 >= a1d20 >= 1, 0.05) # fail
