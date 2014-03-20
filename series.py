'''
Bryan Bonvallet
2009

This module stores classes that can contain an ordered
series of values.

FiniteSequence stores a discrete, finite series of values.
InfiniteSequence stores a discrete, pseudo-infinite or
maintains a calculation to represent a discrete, infinite sequence.

These classes expect to be subclassed with PMF classes.
'''

class FiniteSequence():
   ''' Contains features common to finite sequences.
       Currently, this controls how to print out the sequence. '''

   def __str__(self):
      ''' Convert to string by printing the contained distribution. '''
      return str(self.getDistribution())

class InfiniteSequence():
   ''' Contains features common to infinite sequences.
       Currently, this controls how to print out the sequence. '''

   # Max terms determines the max number of terms to print out.
   # This is merely an aesthetic variable.
   maxterms = 40

   def __str__(self):
      ''' Convert to string by printing the contained distribution, but
          limit number of displayed values. '''
      if (len(self) <= self.maxterms):
         return str(self.distribution)
      else:
         return str(self.distribution[:,0:self.maxterms])
