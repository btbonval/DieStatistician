'''
Bryan Bonvallet
2009

XdY is standard nomenclature for taking X dice with Y faces,
rolling them all, and adding the result.  'd' represents dice.

The class XdY is defined to create a probability mass function
for the result of rolling and adding X dice with Y faces.
'''

from PMF import *

class XdY(PMF,FiniteSequence):
   ''' Represents a discrete probability mass function
       of adding up X dice with Y faces.

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
      ''' Generate the distribution for XdY using PMF intermediates. '''

      # Must generate the base function of 1dY with uniform distribution.
      values = range(1,Y+1)
      probs = numpy.ones(Y) * 1.0/Y
      basepmf = self.__class__([values,probs],self.error)

      # Add the dice distributions together X times.
      pmf = basepmf
      for i in range(1,X):
         pmf = pmf + basepmf

      return pmf

   def setError(self, error):
      ''' Sets the internal maximal error value as a singleton
          real number specified by the argument error.
          Then recalculates the distribution. '''
      self.error = error
      self.setDistribution()

   def __radd__(self, other):
      ''' Reverse add acts just as normal add, but implies other
          does not support adding. '''
      return self + other

   def __add__(self, other):
      ''' The probability distribution of the addition of two
          independent random variables is the convolution of the
          probability distribution functions of the random variables. '''
      # First, make sure other can be compared properly.  
      if not self.validateDistribution(other):
          raise TypeError('Invalid distribution for addition: %s' %(self.validationError))

      # Find appropriate error value.  Choose maximum if possible.
      try:
         if self.error > other.error:
            error = self.error
         else: 
            error = other.error
      except:
         error = self.error

      # "cast" into a friendly format.
      other = self.__class__(other,error)

      # Setup values and probabilities to Convolve the PMFs.
      inputAvalue = self.getDistribution(0)
      inputBvalue = other.getDistribution(0)
      inputAprob = self.getDistribution(1)
      inputBprob = other.getDistribution(1)

      leftside = numpy.min(inputBvalue) - numpy.min(inputAvalue)
      if leftside > 0:
         # B begins further "right" than A.  Leftpad B with zeros.
         inputBprob = numpy.concatenate((numpy.zeros(leftside),inputBprob),1)
      if leftside < 0:
         # B begins further "left" than A.  Leftpad A with zeros.
         inputAprob = numpy.concatenate((numpy.zeros(-1*leftside),inputAprob),1)

      # Convolve the distributions.
      outputprob = numpy.convolve(inputAprob, inputBprob)
      # Either A or B may be left padded.  The number of zeros padded
      # to the input of convolution will be the number of zeros padded
      # to the output.  Skip the padding, but keep the rest.
      outputprob = outputprob[numpy.abs(leftside):]

      # Find the values for the associated convolution.
      minoutputvalue = numpy.min(inputAvalue) + numpy.min(inputBvalue)
      maxoutputvalue = minoutputvalue + len(self) + len(other) - 2
      outputvalue = range(int(minoutputvalue),int(maxoutputvalue)+1)

      return self.__class__(numpy.array([outputvalue, outputprob]),error)


# Some example code
if __name__ == "__main__":
   print "Take one six-sided die and roll it.  The distribution would be: "
   six1 = XdY( (1,6) )
   print str(six1)

   print "Add two six-sided dice together: "
   six2a = six1 + six1
   print str(six2a)

   print "Also add two six-sided dice together: "
   six2b = XdY( (2,6) )
   print str(six2b)
  
   print "If heads is one and tails is two, sum of three coin flips: "
   coin3 = XdY( (3,2) )
   print str(coin3)

   print "Three coin flips plus two six sided dice: "
   mix = six2b + coin3
   print str(mix)

   print "Expected value from the above distribution: "
   print mix.EV()

   print "Take three samples from the above distribution: "
   for i in range(0,3):
      print mix.getSample()
