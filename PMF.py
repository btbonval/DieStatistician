'''
Bryan Bonvallet
2009

Creates a class to build and store a discrete probability mass function.
See help on the PMF class for more information.
'''

import numpy
from series import *

class PMF:
   ''' Represents a discrete probability mass function.

       Allows direct sampling from the distribution, calculation of
       statistics, and some more advanced probability distribution
       arithmetic.  See dir(thisobject) for a list of functions. '''

   # Error estimate due to calculation or measurement.
   # Think of this as a minimum precision requirement.
   error = 1e-5

   def __init__(self, distribution):
      '''
      Instantiate a discrete PMF.  Distribution is expected to be
      a matrix/array compatible with numpy arrays, with a size of
      2xN, and probability values along the 2nd row that sum to 1.
      The first row should be values with probability corresponding to
      that given in the first row of the same column.
      '''
      if not self.validateDistribution(distribution):
          raise TypeError('Invalid distribution: %s.  Input: %s' %(self.validationError, distribution))

      self.distribution = self.castToDistribution(distribution)

   def moment(self,k):
      ''' Calculate sample moment of order k; useful in
          estimating distribution parameters. '''
      return numpy.dot( self.getDistribution(0)**k, self.getDistribution(1) )

   def EV(self):
      ''' Calculates the Expected Value by finding the first moment. '''
      return self.moment(1)

   def getError(self):
      ''' Return the internal maximal error value as a singleton
          real number. '''
      return self.error

   def setError(self,error):
      ''' Sets the internal maximal error value as a singleton
          real number specified by the argument error. '''
      self.error = error

   def getDistribution(self, row=None, col=None):
      ''' Safer way to return a copy of the distribution, specifically
          in the case of infinite distributions. '''
      if row is None and col is None:
         return self.distribution
      if col is None:
         # first if check precludes row from being None
         return self.distribution[row,:]
      if row is None:
         # first if check precludes col from being None
         return self.distribution[:,col]
      return self.distribution[row,col]

   def castToDistribution(self, dist):
      ''' Method to convert input into internal array format. '''
      # Check to see if the distribution can be extracted.
      try:
         dist = dist.getDistribution()
      except:
         # No worries if it can't.
         pass

      # "Cast" to a numpy matrix, as it is more flexible with inputs.
      # Then "cast" to numpy array, as it is better for calculations.
      dist = numpy.array(numpy.matrix(dist)).squeeze()

      # Check for singleton value.  Cast this to a PMF with a single
      # value with unity chance of occurence.
      if dist.size == 1:
         value = dist.flatten()[0]
         dist = numpy.array([[value],[1.0]])

      return dist

   def validateDistribution(self, dist=None):
      '''
      Validates a supplied distribution dist for use in this code base.
      dist is a distribution that could be passed to the constructor.
      If dist has an error attribute, that error will be used to determine
      the precision for this validation.  If dist does not have an error
      attribute, this object's error attribute will be used.
      Returns True or False to indicate valid or invalid.
      '''
      # If dist is not supplied, use internal distribution.
      try:
         if dist is None:
            dist = self.getDistribution()
      except:
         self.validationError = 'Unable to process null distribution'
         return False

      # Use the proper error if applicable
      try:
         error = dist.error
      except:
         error = self.error

      # Attempt to "cast" the distribution successfully.
      try:
         dist = self.castToDistribution(dist)
      except:
         self.validationError = 'Could not cast using castToDistribution()'
         return False

      # Check that first and second row have values.
      try:
         if len(dist[0,:]) == 0:
            raise Exception('')
         if len(dist[1,:]) == 0:
            raise Exception('')
      except:
         self.validationError = 'Input does not appear to be 2xN in dimension'
         return False

      # Check that second row sums to 1 within some level of precision.
      try:
         if numpy.abs(1.0-numpy.sum(dist[1,:])) > error:
            self.validationError = 'Probabilities do not sum to 1 within error of %d' %(self.error)
            return False
      except:
         self.validationError = 'Unable to sum probabilities'
         return False

      # All other objects have been tested and rejected.
      # Accept the distribution.
      return True

   def __str__(self):
      ''' Convert to string by printing the contained distribution. '''
      return str(self.__class__) + '(\n' + str(self.getDistribution()) + '\n)'

   def __len__(self):
      ''' Return length as a number of values N.  The distribution is
          of size 2xN. '''
      return len(self[1,:])

   def __getitem__(self,key):
      ''' Pass along slice to the distribution and return that. '''
      return self.getDistribution().__getitem__(key)

   def __lt__(self, other):
      ''' Return the probability that a random sample from this
          distribution is less than a random sample from the other
          distribution. '''
      # First, make sure other can be compared properly.  
      if not self.validateDistribution(other):
          raise TypeError('Invalid distribution for comparison: %s' %(self.validationError))

      # "cast" into a friendly format
      other = self.__class__(other)

      # Use slow O(NM) ~ O(N^2) algorithm to compare and calculate.
      cumsum = 0.0
      for i in range(0,len(self)):
         for j in range(0,len(other)):
            # For each element, compare if the value is less in this
            # distribution than the other.  If it is, accumulate the
            # probability of those two events occurring together.
            if self[0,i] < other[0,j]:
               cumsum += (self[1,i]*other[1,j])

      # The probability that this distribution is less than the other
      # has been calculated.  Return it.
      return cumsum

   def __eq__(self, other):
      ''' Return the probability that a random sample from this
          distribution is equal to a random sample from the other
          distribution. '''
      # First, make sure other can be compared properly.  
      if not self.validateDistribution(other):
          raise TypeError('Invalid distribution for comparison: %s' %(self.validationError))

      # "cast" into a friendly format
      other = self.__class__(other)

      # Use slow O(NM) ~ O(N^2) algorithm to compare and calculate.
      cumsum = 0.0
      for i in range(0,len(self)):
         for j in range(0,len(other)):
            # For each element, compare if the value is equal in this
            # distribution to the other.  If it is, accumulate the
            # probability of those two events occurring together.
            if numpy.abs(self[0,i] - other[0,j]) <= self.error:
               cumsum += (self[1,i]*other[1,j])

      # The probability that this distribution is less than the other
      # has been calculated.  Return it.
      return cumsum

   def __le__(self, other):
      ''' Return the probability that a random sample from this
          distribution is less than or equal to a random sample from
          the other distribution. '''
      # Since the two sets are independent, we may add the probability
      # of less than to the probability of equal to obtain the probability
      # of less than or equal to.
      return ( (self < other) + (self == other) )

   def __ne__(self, other):
      ''' Return the probability that a random sample from this
          distribution is not equal to a random sample from
          the other distribution. '''
      # Find the complement of the equal probability.
      return ( 1.0 - (self == other) )

   def __gt__(self, other):
      ''' Return the probability that a random sample from this
          distribution is greater than a random sample
          from the other distribution. '''
      # P(X > Y) == P(Y < X).
      # less than is already implemented, so use it instead.

      # First, have to make sure other will use the proper < function.
      if not self.validateDistribution(other):
          raise TypeError('Invalid distribution for comparison: %s' %(self.validationError))
      other = self.__class__(other)

      return (other < self)

   def __ge__(self, other):
      ''' Return the probability that a random sample from this
          distribution is greater than or equal to a random sample
          from the other distribution. '''
      # Since the two sets are independent, we may add the probability
      # of less than to the probability of equal to obtain the probability
      # of less than or equal to.
      return ( (self > other) + (self == other) )

   def __hash__(self):
      ''' Returns hash code value for this object. (Cannot perform, raises error)'''
      raise TypeError('PMF objects are unhashable')

   def getSample(self):
      ''' Returns a random sample from the distribution. '''
      # Current method:
      # Sample from a denser uniform distribution and map it to the CDF of
      # this PMF with a 1:1 outcome.

      # Determine how many values are required of the uniform distribution
      # to be more dense.
      # Find the minimum difference between probabilities and square it.
      absmin = numpy.min(self[1,:])
      relmin = numpy.min(numpy.diff(self[1,:]))
      # For uniform distributions, relmin will be 0.  In this case, use
      # the absmin.  Otherwise use the minimum of the two.
      if relmin == 0:
         realmin = absmin
      else:
         realmin = min([absmin, relmin])
      elements = numpy.ceil((1.0/realmin)**2)

      # Sample from uniform distribution in integer range of (0,elements]
      # where each integer has a probability of 1/elements.
      unisample = numpy.random.randint(0,elements)+1
      # Convert sample to the CDF probability of that sample being drawn.
      unisample = unisample / elements
      # unisample now lies on (0,1].
      # Now consider a CDF for this distribution.  unisample represents
      # the output, map it back to the input.
      myCDF = numpy.cumsum(self[1,:])
      minidx = numpy.min(numpy.nonzero(myCDF >= unisample))
      # minidx is the index of the smallest probability that is greater than
      # the unisample.  The corresponding value is the desired value.
      return self[0,minidx]


# Example usage
if __name__ == "__main__":
   # Establish a uniform distribution for values 1, 2, 3, and 4

   # Make the first try a bad one: does not sum to 1.
   dist = numpy.matrix([ [1, 2, 3, 4], [0.25, 0.25, 0.25, 0.26] ])
   try:
      test1 = PMF(dist)
   except TypeError:
      print "Testing invalid distribution successful."

   # Make the second try a good one.
   dist = numpy.matrix([ [1, 2, 3, 4], [0.25, 0.25, 0.25, 0.25] ])
   try:
      test2 = PMF(dist)
   except:
      print "Testing valid distribution failed."
      import sys
      sys.exit(1)

   # Test EV
   if test2.EV() == 2.5:
      print "Expected value successfully calculated."
   else:
      print "Expected value test failed."

   # Draw from the distribution and tally results.
   trials = 500
   tally = numpy.array([0,0,0,0])
   for i in range(0,trials):
      drawvalue = test2.getSample()
      # drawvalue is 1 to 4 inclusive
      # corresponding index in tally is 0 to 3 inclusive
      index = int(drawvalue - 1)
      # increment the appropriate index
      tally[index] = tally[index] + 1
   # Calculate percentage for each draw
   tally = tally / float(trials)
   print "tally should be approximately 0.25 for each value: "
   for i in range(0,len(test2)):
      print str(test2[0,i]) + ": " + str(tally[i])

   # Create an uneven distribution with odd values
   dist = numpy.matrix([ [10, 25, 50, 99], [0.25, 0.125, 0.325, 0.3] ])
   try:
      test3 = PMF(dist)
   except:
      print "Testing valid distribution failed."
      import sys
      sys.exit(1)

   # Draw from the distribution and tally results.
   trials = 1000
   index = {10.0: 0, 25.0: 1, 50.0: 2, 99.0: 3}
   tally = numpy.array([0,0,0,0])
   for i in range(0,trials):
      drawvalue = test3.getSample()
      tally[index[drawvalue]] = tally[index[drawvalue]] + 1
   # Calculate percentage for each draw
   tally = tally / float(trials)
   print "tally should be (10, 0.25), (25, 0.125), (50, 0.325), (99, 0.3): "
   for i in range(0,len(test3)):
      print str(test3[0,i]) + ": " + str(tally[i])
