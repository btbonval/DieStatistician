'''
Bryan Bonvallet
2009

This file contains extra math functions needed for Die Statistician.
'''

def factorial(x):
   ''' Find the value of x!  Might operate strangely on non-integers. '''
   acc = 1
   while x > 0:
      acc = acc * x
      x -= 1
   return acc

def permutation(n,r):
   ''' Calculate the permutation of n P r. '''
   return factorial(n)*1.0 / factorial(n-r)

def combination(n,k):
   ''' Calculate the combination of n C k. '''
   return permutation(n,k) / factorial(k)
