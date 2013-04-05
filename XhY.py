'''
Bryan Bonvallet
2009

XhY is novel nomenclature.  'h' represents highest die.  Throw
X dice of Y sides, and then find the highest value shown.  The
idea is similar to that used in Risk, but only in that there is
no addition of die values.  Mathematically it is quite different.
The more six sided dice that are thrown, the more likely the
outcome is to be 6.
'''

from PMF import *
from extramath import *

class XhY(PMF,FiniteSequence):
   ''' Represents a discrete probability mass function
       of taking the highest face of X dice with Y faces.

       Allows direct sampling from the distribution, calculation of
       statistics, and some more advanced probability distribution
       arithmetic. '''

   def __init__(self, description, error=None):
      ''' Instantiate a discrete PMF.  Description is either [X, Y] or a
          distribution.
          X and Y are integers.  X represents the number of dice, and Y
          represents the number of faces on each die. '''
      if error is not None:
         self.error = error
      self.description = description
      self.setDistribution()

   def setDistribution(self):
      ''' Updates the internal distribution using the internal error and
          internal description. '''
      description = self.description
      try:
         # Assume [X, Y] is provided:
         if numpy.matrix(description).size == 2:
            distribution = self.genDistribution(description[0],description[1])
         else:
            distribution = description
      except:
         # [X, Y] is not provided.  Assume it is a distribution.
         distribution = description
      
      if not self.validateDistribution(distribution):
          raise TypeError('Invalid distribution: %s.  Input: %s' %(self.validationError, distribution))

      self.distribution = self.castToDistribution(distribution)

   def genDistribution(self, X, Y):
      ''' Generate the distribution for XhY using PMF intermediates. '''
      # Must generate the base function of 1hY with uniform distribution.
      values = range(1,Y+1)
      probs = numpy.zeros(Y)
      # Run a summation for each output value to determine its probability.
      for Z in values:
         acc = 0
         for i in range(1,X+1):
            acc += combination(X,i) * (1.0/Y)**i * ((Z-1.0)/Y)**(X-i)
         probs[Z-1] = acc
      pmf = self.__class__([values,probs],self.error)
      return pmf

   def setError(self, error):
      ''' Sets the internal maximal error value as a singleton
          real number specified by the argument error.
          Then recalculates the distribution. '''
      self.error = error
      self.setDistribution()

   def __ror__(self,other):
      ''' This is the same as or, but implies other does not support or. '''
      return self | other

   def __or__(self, other):
      ''' The probability distribution of the take highest operation
          over two independent random variables is an O(nm) comparison. '''
      # First, make sure other can be compared properly.
      if not self.validateDistribution(other):
         raise TypeError('Invalid distribution for take-highest: %s' %(self.validation))

      # Find appropriate error value.  Choose maximum if possible.
      try:
         if self.error > other.error:
            error = self.error
         else:
            error = other.error
      except:
         error = self.error

      # "cast" into friendly format.
      other = self.__class__(other,error)

      # Establish output domain.
      minmin = int(min([min(self.distribution[0,:]), min(other.distribution[0,:])]))
      maxmax = int(max([max(self.distribution[0,:]), max(other.distribution[0,:])]))
      odomain = numpy.array(range(minmin,maxmax+1))

      # Find output range by accumulating probabilities.
      orange = numpy.zeros(maxmax-minmin+1)
      for i in range(0,len(self)):
         for j in range(0,len(other)):
            # Determine probability of this event occuring
            prob = self.distribution[1,i] * other.distribution[1,j]

            # Determine largest of showing value
            value = max([self.distribution[0,i], other.distribution[0,j]])
            idx = value - minmin

            # Update probability for selected value
            orange[idx] += prob

      return self.__class__(numpy.array([[odomain],[orange]]),error)
