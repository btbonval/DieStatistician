'''
Bryan Bonvallet
2009

XeY is a novel notation.  'e' represents exploding dice.  The term
exploding dice is borrowed from the game Shadowrun.  Roll X dice
with Y sides.  Any single die showing its maximum face results in
an additional die roll.  Accumulate all values shown.
'''

from XdY import *
from series import InfiniteSequence

class XeY(XdY,InfiniteSequence):
   ''' Represents a discrete probability mass function
       of X dice with Y faces, with the rule that rolling
       the max face value on any die allows an additional roll
       of that die with a running sum.

       Since it is theoretically possible to roll the maximum value
       an infinite number of times, this probability mass function
       represents a discrete, infinite sequence.  In practice, it is
       finite up to the precision specified by the error attribute.

       Allows direct sampling from the distribution, calculation of
       statistics, and some more advanced probability distribution
       arithmetic. '''

   def genDistribution(self, X, Y):
      ''' Generate the distribution for XeY using PMF intermediates. '''
      # Must generate the base function of 1eY with uniform distribution.
      values = numpy.array(range(1,Y+1))
      probs = numpy.ones(Y) * 1.0/Y
      probs[Y-1] = 0.0

      # Temporarily make the error large enough to encompass
      # enough explosions to be valid after convolving.
      baseerror = self.error
      self.error = self.error / (10 ** (X-1))

      # The distribution runs out to infinity. Use the error value to
      # truncate the distribution.  Calculate the distribution until
      # it satisfies the error criteria.
      count = 0
      basevalues = values
      baseprobs = probs
      while not self.validateDistribution(numpy.array([values,probs])):
         count += 1
         values = numpy.concatenate((values, basevalues + count*Y),1)
         probs = numpy.concatenate((probs, baseprobs ** (count+1)),1)

      # Set error back to its original value.
      self.error = baseerror
      # Add the dice distributions together X times.
      basepmf = self.__class__(numpy.array([values,probs]),self.error)
      pmf = basepmf
      for i in range(1,X):
         pmf = pmf + basepmf

      return pmf
