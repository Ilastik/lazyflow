import numpy
from roi import roiToSlice

class SlotType( object ):
    def allocateDestination( self, roi ):
      pass

    def returnDestination(self, destination):
      pass

    def writeIntoDestination( self, destination, value, roi ):
      pass

    def isCompatible(self, value):
      """
      Slot types must implement this method.

      this method should check wether the supplied value
      is compatible with this type trait and should return true
      if this is the case.
      """
      pass

    def setupMetaForValue(self, value):
      """
      Slot types must implement this method.

      this method should extract valuable meta information
      from the provied value and set up the self.slot.meta
      MetaDict accordingly
      """
      pass

class ArrayLike( SlotType ):
    def __init__( self, slot):
        self.slot = slot

    def allocateDestination( self, roi ):
        shape = roi.stop - roi.start if roi else self.slot.meta.shape
        storage = numpy.ndarray(shape, dtype=self.slot.meta.dtype)
        # if axistags is True:
        #     storage = vigra.VigraArray(storage, storage.dtype, axistags = copy.copy(s))elf.axistags))
        #     #storage = storage.view(vigra.VigraArray)
        #     #storage.axistags = copy.copy(self.axistags)
        return storage

    def returnDestination(self, destination):
        return destination

    def writeIntoDestination( self, destination, value, roi ):
        sl = roiToSlice(roi.start, roi.stop)
        destination[:] = value[sl]

    def isCompatible(self, value):
      return True


    def setupMetaForValue(self, value):
      if hasattr(value,"__getitem__")  and hasattr(value,"shape") and hasattr(value,"dtype"):
        self.slot.meta.shape = value.shape
        self.slot.meta.dtype = value.dtype
        if hasattr(value,"axistags"):
          self.slot.meta.axistags = value.axistags
      else:
        self.slot.meta.shape = (1,)
        self.slot.meta.dtype = object


class Default( SlotType ):
    def __init__( self, slot):
        self.slot = slot

    def allocateDestination( self, roi ):
        return [None]
   
    def returnDestination(self, destination):
        return destination[0]

    def writeIntoDestination( self, destination, value,roi ):
        destination[0] = value
    

    def isCompatible(self, value):
      return True
   
    def setupMetaForValue(self, value):
      self.slot.meta.shape = (1,)
      self.slot.meta.dtype = object
      self.slot.meta.axistags = vigra.defaultAxistags(1)
