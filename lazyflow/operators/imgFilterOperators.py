from lazyflow.graph import Operator,InputSlot,OutputSlot
from lazyflow.operators import OpArrayPiper
from lazyflow.helpers import AxisIterator,generateRandomRoi
import numpy,vigra,copy
from lazyflow.roi import TinyVector

class OpBaseVigraFilter(OpArrayPiper):
    
    inputSlots = [InputSlot("input"),InputSlot("sigma")]
    outputSlots = [OutputSlot("output")]
    
    name = 'OpBaseVigraFilter'
    category = 'Image Filter'
    
    vigraFilter = None
    windowSize = 4.0
    
    def __init__(self, parent):
        OpArrayPiper.__init__(self, parent)
    
    def setupOutputs(self):
        
        inputSlot = self.inputs["input"]
        outputSlot = self.outputs["output"]
        
        outputSlot._dtype = inputSlot.dtype
        outputSlot._shape = inputSlot.shape
        outputSlot._axistags = copy.copy(inputSlot.axistags)
        
        ################################################
#        numChannels  = 1
#        inputSlot = self.inputs["Input"]
#        if inputSlot.axistags.axisTypeCount(vigra.AxisType.Channels) > 0:
#            channelIndex = self.inputs["Input"].axistags.channelIndex
#            numChannels = self.inputs["Input"].shape[channelIndex]
#            inShapeWithoutChannels = popFlagsFromTheKey( self.inputs["Input"].shape,self.inputs["Input"].axistags,'c')
#        else:
#            inShapeWithoutChannels = inputSlot.shape
#            channelIndex = len(inputSlot.shape)
#                
#        self.outputs["Output"]._dtype = self.outputDtype
#        p = self.inputs["Input"].partner
#        at = copy.copy(inputSlot.axistags)
#
#        if at.axisTypeCount(vigra.AxisType.Channels) == 0:
#            at.insertChannelAxis()
#            
#        self.outputs["Output"]._axistags = at 
#        
#        channelsPerChannel = self.resultingChannels()
#        inShapeWithoutChannels = list(inShapeWithoutChannels)
#        inShapeWithoutChannels.insert(channelIndex,numChannels * channelsPerChannel)        
#        self.outputs["Output"]._shape = tuple(inShapeWithoutChannels)
#        
#        if self.outputs["Output"]._axistags.axisTypeCount(vigra.AxisType.Channels) == 0:
#            self.outputs["Output"]._axistags.insertChannelAxis()

    
    def execute(self,slot,roi,result):
        
        axistags = self.inputs["input"].axistags
        sigma = self.inputs["sigma"].value
        
        roi.setAxistags(axistags)
        roi.expandByShape(sigma*self.windowSize)
        
        source = self.inputs["input"](roi.start,roi.stop).wait()
        source = vigra.VigraArray(source,axistags=axistags)
        
        roi.decreaseByShape(sigma*self.windowSize)
        roi.centerIn(source.shape)
        roi.popAxis('c')
        
        spaceIterator = AxisIterator(source,'spatialc',result,'spatialc')

        for srckey,trgtkey in spaceIterator:
            self.vigraFilter(source[srckey],sigma,out=result[trgtkey],roi=(roi.start,roi.stop))
            
    
class OpGaussianSmoothing(OpBaseVigraFilter):
    name = "GaussianSmoothing"
    
    vigraFilter = staticmethod(vigra.filters.gaussianSmoothing)
    outputDtype = numpy.float32 

    def resultingChannels(self):
        return 1
    
    def setupOutputs(self):
        OpBaseVigraFilter.setupOutputs(self)
        
class OpHessianOfGaussian(OpBaseVigraFilter):
    name = "HessianOfGaussian"
    vigraFilter = staticmethod(vigra.filters.hessianOfGaussian)
    outputDtype = numpy.float32 
    
    def resultingChannels(self):
        temp = self.inputs["Input"].axistags.axisTypeCount(vigra.AxisType.Space)*(self.inputs["Input"].axistags.axisTypeCount(vigra.AxisType.Space) + 1) / 2
        return temp
        
from lazyflow.graph import Graph        

if __name__ == "__main__":
    g = Graph()
    op = OpHessianOfGaussian(g)
    vol = vigra.VigraArray((200,200,1))
    vol[20:180,20:180,0] = 1
    vol[:]  = 1 - vol[:]
    vigra.impex.writeImage(vol,'davor.png')
    result = vigra.filters.hessianOfGaussian(vol, 5.0)
    vigra.impex.writeImage(result,'danach.png')
    op.inputs["input"].setValue(vol)
    op.inputs["sigma"].setValue(2.0)
    roi = [TinyVector([60,60,0]),TinyVector([140,140,1])]
    op.outputs["output"](roi[0], roi[1]).wait()