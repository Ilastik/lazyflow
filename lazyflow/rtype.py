from roi import sliceToRoi, roiToSlice
import vigra,numpy,copy
from lazyflow.roi import TinyVector

class Roi(object):
    def __init__(self, slot):
        self.slot = slot
        pass
    pass

class SubRegion(Roi):
    def __init__(self, slot, start = None, stop = None, pslice = None):
        super(SubRegion,self).__init__(slot)
        if pslice != None or start is not None and stop is None and pslice is None:
            if pslice is None:
                pslice = start
            assert self.slot.meta.shape is not None
            self.start, self.stop = sliceToRoi(pslice,self.slot.meta.shape)
        elif start is None and pslice is None:
            self.start, self.stop = sliceToRoi(slice(None,None,None),self.slot.meta.shape)
        else:
            self.start = start
            self.stop = stop
        self.axistags = None
        self.dim = len(self.start)
    
    def __str__( self ):
        return "".join(("Subregion: start '", str(self.start), "' stop '", str(self.stop), "'"))
    
    def setAxistags(self,axistags):
        assert type(axistags) == vigra.vigranumpycore.AxisTags
        self.axistags = copy.copy(axistags)
    
    def expandByShape(self,shape):
        """
        extend a roi by a given in shape
        """
        #TODO: make sure its bounded
        if type(shape == int):
            tmp = shape
            shape = numpy.zeros(self.dim).astype(int)
            shape[:] = tmp
            shape[self.axistags.channelIndex] = 0
        self.start = TinyVector([x-s for x,s in zip(self.start,shape)])
        self.stop = TinyVector([x+s for x,s in zip(self.stop,shape)])
    
    def decreaseByShape(self,shape):
        """
        extend a roi by a given in shape
        """
        #TODO: make sure its bounded
        if type(shape == int):
            tmp = shape
            shape = numpy.zeros(self.dim).astype(int)
            shape[:] = tmp
            shape[self.axistags.channelIndex] = 0
        self.start = TinyVector([x+s for x,s in zip(self.start,shape)])
        self.stop = TinyVector([x-s for x,s in zip(self.stop,shape)])
    
    def popAxis(self,axis):
        popKey = self.axistags.index(axis)
        self.start.pop(popKey)
        self.stop.pop(popKey)
    
    def centerIn(self,shape):
        difference = [int(((shape-(stop-start))/2.0)) for (shape,start),stop in zip(zip(shape,self.start),self.stop)]  
        dimension = [int(stop-start) for start,stop in zip(self.start,self.stop)]
        self.start = TinyVector(difference)
        self.stop = TinyVector([diff+dim for diff,dim in zip(difference,dimension)])
    
    def isEqualTo(self,shape):
        channelKey = self.axistags.index('c')
        dim = [stop-start for start,stop in zip(self.start,self.stop)]
        shape = list(shape)
        dim.pop(channelKey)
        shape.pop(channelKey)
        if dim == list(shape):
            return True
        else:
            return False
    
    def setStopAtAxisTo(self,axis,length):
        axisKey = self.axistags.index(axis)
        self.stop[axisKey]=length
        
                
    def toSlice(self, hardBind = False):
        return roiToSlice(self.start,self.stop, hardBind)
