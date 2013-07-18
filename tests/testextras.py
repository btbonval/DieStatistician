'''
Bryan Bonvallet
2013

This file tests the functions in extramath.py.
'''

import unittest
import extramath

class TestExtras(unittest.TestCase):
    def test_factorial(self):
        # Compare against a sampling of known factorial values.
        known = ( (0, 1),
                  (1, 1),
                  (2, 2),
                  (3, 6), 
                  (8, 40320),
                 (13, 6227020800), )
        for x,y in known:
            self.assertEqual(extramath.factorial(x), y)

    def test_permutation(self):
        # Compare against a sampling of known permuation values.
        known = ( (16, 3, 3360),
                  (10, 2, 90),
                  (10, 3, 720),
                  ( 8, 3, 336), )
        for x,y,z in known:
            self.assertEqual(extramath.permutation(x,y), z)

    def test_combination(self):
        # Compare against a sampling of known combination values.
        known = ( (16,  3, 560),
                  (16, 13, 560),
                  (10,  3, 120),
                  (10,  7, 120),
                  ( 8,  3, 56),
                  ( 8,  5, 56), )
        for x,y,z in known:
            self.assertEqual(extramath.combination(x,y), z)
