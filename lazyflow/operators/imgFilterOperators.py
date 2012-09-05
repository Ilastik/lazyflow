from lazyflow.graph import Operator,InputSlot,OutputSlot
from lazyflow.helpers import AxisIterator, newIterator
from lazyflow.operators.obsolete.generic import OpMultiArrayStacker
from lazyflow.operators.obsolete.vigraOperators import Op50ToMulti
import numpy
import vigra
from math import sqrt
from functools import partial
from lazyflow.roi import roiToSlice

class OpBaseVigraFilter(Operator):
    
    inputSlots = []
    outputSlots = [OutputSlot("Output")]
    
    name = 'OpBaseVigraFilter'
    category = 'Image Filter'
    
    vigraFilter = None
    windowSize = 4.0
    
    def __init__(self, parent):
        Operator.__init__(self, parent)
        self.iterator = None
        
    def resultingChannels(self):
        pass
    
    def setupFilter(self):
        pass
    
    def setupIterator(self,source,result):
        self.iterator = AxisIterator(source,'spatialc',result,'spatialc',[(),(1,1,1,1,self.resultingChannels())])
    
    def setupOutputs(self):
        
        inputSlot = self.inputs["Input"]
        outputSlot = self.outputs["Output"]
        channelNum = self.resultingChannels()
        outputSlot.meta.assignFrom(inputSlot.meta)
        outputSlot.setShapeAtAxisTo('c', channelNum)
        
    def execute(self,slot,roi,result):
        axistags = self.inputs["Input"].axistags
        inputShape  = self.inputs["Input"].shape
        channelIndex = axistags.index('c')
        channelsPerC = self.channelsPerChannel()
        #vigra returns a value even if time is not part of the axistags
        timeIndex = axistags.index('t')
        if timeIndex >= roi.dim:
            timeIndex = None
        origRoi = roi.copy()
        #Set up roi
        roi.setInputShape(inputShape)
        #setup filter ONLY WHEN SIGMAS ARE SET and get maxSigma  
        sigma = self.setupFilter()
        #set up roi to retrieve necessary source data
        roi.expandByShape(sigma*self.windowSize,channelIndex).adjustChannel(channelsPerC,channelIndex)
        source = self.inputs["Input"](roi.start,roi.stop).wait()
        source = vigra.VigraArray(source,axistags=axistags)
        srcGrid = [source.shape[i] if i!= channelIndex else 1 for i in range(len(source.shape))]
        trgtGrid = [inputShape[i]  if i != channelIndex else self.channelsPerChannel() for i in range(len(source.shape))]
        #vigra cant handle time (easily) 
        if timeIndex is not None:
            srcGrid[timeIndex] = 1
            trgtGrid[timeIndex] = 1
        nIt = newIterator(origRoi,srcGrid,trgtGrid,timeIndex=timeIndex,channelIndex = channelIndex,halo = sigma,style = 'lazyflow')
        for src,trgt,mask in nIt:
            result[trgt] = self.vigraFilter(source = source[src])[mask]
        return result
    
class OpGaussianSmoothing(OpBaseVigraFilter):
    inputSlots = [InputSlot("Input"),InputSlot("Sigma")]
    name = "GaussianSmoothing"
    
    def __init__(self,parent):
        OpBaseVigraFilter.__init__(self,parent)
        self.vigraFilter = None
        
    def setupIterator(self,source,result):
        self.iterator = AxisIterator(source,'spatialc',result,'spatialc',[(),({'c':self.channelsPerChannel()})])   
    
    def setupFilter(self):
        sigma = self.inputs["Sigma"].value
        
        def tmpFilter(source,sigma):
            tmpfilter = vigra.filters.gaussianSmoothing
            return tmpfilter(array=source,sigma=sigma)
    
        self.vigraFilter = partial(tmpFilter,sigma=sigma)
        return sigma
        
    def resultingChannels(self):
        return self.inputs["Input"].meta.shape[self.inputs["Input"].meta.axistags.index('c')]
    
    def channelsPerChannel(self):
        return 1
    
class OpDifferenceOfGaussians(OpBaseVigraFilter):
    inputSlots = [InputSlot("Input"), InputSlot("Sigma", stype = "float"), InputSlot("Sigma2", stype = "float")]
    name = "DifferenceOfGaussians"
    
    def __init__(self,parent):
        OpBaseVigraFilter.__init__(self,parent)
        self.vigraFilter = None
        
    def setupFilter(self):
        sigma0 = self.inputs["Sigma"].value
        sigma1 = self.inputs["Sigma2"].value
        
        def tmpFilter(s0,s1,source):
            tmpfilter = vigra.filters.gaussianSmoothing
            return tmpfilter(source,s0)-tmpfilter(source,s1)
        
        self.vigraFilter = partial(tmpFilter,s0=sigma0,s1=sigma1)
        return max(sigma0,sigma1)
    
    def resultingChannels(self):
        return self.inputs["Input"].meta.shape[self.inputs["Input"].meta.axistags.index('c')]
    
    def channelsPerChannel(self):
        return 1

        
class OpHessianOfGaussian(OpBaseVigraFilter):
    inputSlots = [InputSlot("Input"),InputSlot("Sigma")]
    name = "OpHessianOfGaussian"
    
    def __init__(self,parent):
        OpBaseVigraFilter.__init__(self,parent)
        self.vigraFilter = None
        
    def setupIterator(self,source,result):
        self.iterator = AxisIterator(source,'spatial',result,'spatial',[(),({'c':self.resultingChannels()})])   
    
    def setupFilter(self):
        sigma = self.inputs["Sigma"].value
        
        def tmpFilter(source,sigma):
            tmpfilter = vigra.filters.hessianOfGaussian
            if source.axistags.axisTypeCount(vigra.AxisType.Space) == 2:
                return tmpfilter(image=source,sigma=sigma)
            elif source.axistags.axisTypeCount(vigra.AxisType.Space) == 3:
                return tmpfilter(volume=source,sigma=sigma)
            
        self.vigraFilter = partial(tmpFilter,sigma=sigma)
        return sigma
        
    def resultingChannels(self):
        return self.inputs["Input"].axistags.axisTypeCount(vigra.AxisType.Space)*(self.inputs["Input"].axistags.axisTypeCount(vigra.AxisType.Space) + 1) / 2
    
    def channelsPerChannel(self):
        return self.inputs["Input"].axistags.axisTypeCount(vigra.AxisType.Space)*(self.inputs["Input"].axistags.axisTypeCount(vigra.AxisType.Space) + 1) / 2
    
class OpLaplacianOfGaussian(OpBaseVigraFilter):
    inputSlots = [InputSlot("Input"), InputSlot("Sigma", stype = "float")]
    name = "LaplacianOfGaussian"
    
    def __init__(self,parent):
        OpBaseVigraFilter.__init__(self,parent)
        self.vigraFilter = None
        
    def setupFilter(self):
        scale = self.inputs["Sigma"].value
        
        def tmpFilter(source,scale):
            tmpfilter = vigra.filters.laplacianOfGaussian
            return tmpfilter(array=source,scale=scale)

        self.vigraFilter = partial(tmpFilter,scale=scale)

        return scale
    
    def resultingChannels(self):
        return self.inputs["Input"].meta.shape[self.inputs["Input"].meta.axistags.index('c')]
    
    def channelsPerChannel(self):
        return 1

class OpStructureTensorEigenvalues(OpBaseVigraFilter):
    inputSlots = [InputSlot("Input"), InputSlot("Sigma", stype = "float"),InputSlot("Sigma2", stype = "float")]
    name = "StructureTensorEigenvalues"
    
    def __init__(self,parent):
        OpBaseVigraFilter.__init__(self,parent)
        self.vigraFilter = None
        
    def setupFilter(self):
        innerScale = self.inputs["Sigma2"].value
        outerScale = self.inputs["Sigma"].value
        
        def tmpFilter(source,innerScale,outerScale):
            tmpfilter = vigra.filters.structureTensorEigenvalues
            return tmpfilter(image=source,innerScale=innerScale,outerScale=outerScale)

        self.vigraFilter = partial(tmpFilter,innerScale=innerScale,outerScale=outerScale)

        return max(innerScale,outerScale)
    
    def setupIterator(self, source, result):
        self.iterator = AxisIterator(source,'spatial',result,'spatial',[(),({'c':self.channelsPerChannel()})])   
        
    def resultingChannels(self):
        return self.inputs["Input"].axistags.axisTypeCount(vigra.AxisType.Space)*self.inputs["Input"].shape[self.inputs["Input"].axistags.channelIndex]
    
    def channelsPerChannel(self):
        return self.inputs["Input"].axistags.axisTypeCount(vigra.AxisType.Space)
    
class OpHessianOfGaussianEigenvalues(OpBaseVigraFilter):
    inputSlots = [InputSlot("Input"), InputSlot("Sigma", stype = "float")]
    name = "HessianOfGaussianEigenvalues"
    
    def __init__(self,parent):
        OpBaseVigraFilter.__init__(self,parent)
        self.vigraFilter = None
        
    def setupFilter(self):
        scale = self.inputs["Sigma"].value
        
        def tmpFilter(source,scale):
            tmpfilter = vigra.filters.hessianOfGaussianEigenvalues
            return tmpfilter(image=source,scale=scale)

        self.vigraFilter = partial(tmpFilter,scale=scale)
        
        return scale
    
    def setupIterator(self, source, result):
        self.iterator = AxisIterator(source,'spatial',result,'spatial',[(),({'c':self.channelsPerChannel()})])   
  
    def resultingChannels(self):
        return self.inputs["Input"].axistags.axisTypeCount(vigra.AxisType.Space)*self.inputs["Input"].shape[self.inputs["Input"].axistags.channelIndex]
    
    def channelsPerChannel(self):
        return self.inputs["Input"].axistags.axisTypeCount(vigra.AxisType.Space)
    
class OpGaussianGradientMagnitude(OpBaseVigraFilter):
    inputSlots = [InputSlot("Input"), InputSlot("Sigma", stype = "float")]
    name = "GaussianGradientMagnitude"
    
    def __init__(self,parent):
        OpBaseVigraFilter.__init__(self,parent)
        self.vigraFilter = None
        
    def setupFilter(self):
        sigma = self.inputs["Sigma"].value
                
        def tmpFilter(source,sigma):
            tmpfilter = vigra.filters.gaussianGradientMagnitude
            return tmpfilter(source,sigma=sigma)

        self.vigraFilter = partial(tmpFilter,sigma=sigma)
        return sigma

    def resultingChannels(self):
        return self.inputs["Input"].meta.shape[self.inputs["Input"].meta.axistags.index('c')]
    
    def channelsPerChannel(self):
        return 1
    

class OpPixelFeaturesPresmoothed(Operator):
    
    name="OpPixelFeaturesPresmoothed"
    inputSlots = [InputSlot("Input"), InputSlot("Matrix"), InputSlot("Scales")]
    outputSlots = [OutputSlot("Output"), OutputSlot("arrayOfOperators")]
    
    def __init__(self,parent):
        Operator.__init__(self, parent, register=True)
        
        self.multi = Op50ToMulti(self.graph)
        self.stacker = OpMultiArrayStacker(self.graph)
        self.smoother = OpGaussianSmoothing(self.graph)
        self.destSigma = 1.0
        self.windowSize = 4
        self.operatorList = [OpGaussianSmoothing,OpLaplacianOfGaussian,\
                        OpStructureTensorEigenvalues,OpHessianOfGaussianEigenvalues,\
                        OpGaussianGradientMagnitude,OpDifferenceOfGaussians]
        
    def setupOutputs(self):
        
        #TODO: Different assertions and stuff.
        self.inMatrix = self.inputs["Matrix"].value
        self.inScales = self.inputs["Scales"].value
        self.modSigmas = [0]*len(self.inScales)
        self.maxSigma = numpy.max(self.inScales)
        self.incrSigmas = [0]*len(self.inScales)
        
        #set modified sigmas
        for i in xrange(len(self.inScales)):
            if self.inScales[i] > self.destSigma:
                self.modSigmas[i]=(sqrt(self.inScales[i]**2-self.destSigma**2))
                
        self.modSigmas.insert(0,0)
        for i in xrange(len(self.modSigmas)-1):
            self.incrSigmas[i]=sqrt(self.modSigmas[i+1]**2-self.modSigmas[i]**2)
        self.modSigmas.remove(0)
            
        #set Operators
        operatorList = self.operatorList
        
        scaleMultiplyList = [False,False,0.5,False,False,0.66]
        
        self.operatorMatrix = [[None]*len(self.inMatrix[i]) for i in xrange(len(self.inMatrix))]
        
        
        k=0
        for i in xrange(len(self.inMatrix)): #Cycle through operators == i
            for j in xrange(len(self.inMatrix[i])): #Cycle through sigmas == j
                if self.inMatrix[i][j]:
                    self.operatorMatrix[i][j] = operatorList[i](self.graph)
                    self.operatorMatrix[i][j].inputs["Input"].connect(self.inputs["Input"])
                    self.operatorMatrix[i][j].inputs["Sigma"].setValue(self.destSigma)
                    if scaleMultiplyList[i]:
                        self.operatorMatrix[i][j].inputs["Sigma2"].setValue(self.destSigma*scaleMultiplyList[i])
                    self.multi.inputs["Input%02d"%(k)].connect(self.operatorMatrix[i][j].outputs["Output"])
                    k += 1
                    
        self.stacker.inputs["AxisFlag"].setValue('c')
        self.stacker.inputs["AxisIndex"].setValue(self.inputs["Input"].meta.axistags.index('c'))
        self.stacker.inputs["Images"].connect(self.multi.outputs["Outputs"])
        
        self.outputs["Output"].meta.axistags = self.stacker.outputs["Output"].meta.axistags
        self.outputs["Output"].meta.shape = self.stacker.outputs["Output"].meta.shape
        self.outputs["Output"].meta.dtype = numpy.float32 
        
        #transpose operatorMatrix for better handling
        opMatrix = self.operatorMatrix
        newOpMatrix = [[None]*len(opMatrix) for i in xrange(len(opMatrix[0]))]
        opList = []
        for l in opMatrix:
            opList += l
        for i in xrange(len(opList)):
            newOpMatrix[i/len(opMatrix)][i%len(opMatrix)] = opList[i]
        self.operatorMatrix = newOpMatrix
        
        
    def execute(self,slot,roi,result):
        
        #Get axistags and inputShape
        axistags = self.inputs["Input"].axistags
        inputShape  = self.inputs["Input"].shape
        cIndex = self.outputs["Output"].axistags.channelIndex
        
        #Set up roi 
        roi.setInputShape(inputShape)

        #Request Required Region
        srcRoi = roi.expandByShape(self.maxSigma*self.windowSize,cIndex)
        source = self.inputs["Input"](srcRoi.start,srcRoi.stop).wait()
        
        #disconnect all operators
        opM = self.operatorMatrix
        cIter = 0
        for sig in xrange(len(opM)):#for each sigma
            self.smoother.inputs["Sigma"].setValue(self.incrSigmas[sig])
            self.smoother.inputs["Input"].setValue(source)
            source = self.smoother.outputs["Output"]().wait()
            for op in xrange(len(opM[sig])):#for each operator with this sigma
                if opM[sig][op] is not None:
                    opM[sig][op].inputs["Input"].disconnect()
                    opM[sig][op].inputs["Input"].setValue(source)
                    cIndex = opM[sig][op].outputs["Output"].axistags.channelIndex
                    cSize  = opM[sig][op].outputs["Output"].shape[cIndex]
                    slicing = [slice(0,result.shape[i],None) if i != cIndex \
                               else slice(cIter,cIter+cSize,None) for i in \
                               range(len(result.shape))]
                    result[slicing] = opM[sig][op].outputs["Output"]().wait()
                    cIter += cSize
        return result     
    
from lazyflow.graph import Graph
import vigra

if __name__ == "__main__":
    
    v = vigra.VigraArray((100,100,100))
    g = Graph()
    op = OpGaussianSmoothing(g)
    op.inputs["Input"].setValue(v)
    op.inputs["Sigma"].setValue(1.0)
    print op.outputs["Output"]([1,1,1],[7,7,4]).wait()  
                
                
        