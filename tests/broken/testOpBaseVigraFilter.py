import unittest
import itertools
import vigra
import numpy
from lazyflow.roi import roiToSlice,TinyVector
from lazyflow.graph import Graph
from lazyflow.operators.imgFilterOperators import OpGaussianSmoothing,\
     OpLaplacianOfGaussian, OpStructureTensorEigenvalues,\
     OpHessianOfGaussianEigenvalues, OpGaussianGradientMagnitude,\
     OpDifferenceOfGaussians, OpHessianOfGaussian

class TestOpBaseVigraFilter(unittest.TestCase):
    def setUp(self):

        self.testDimensions = ['xyzc','xyc','txyc','txyzc']
        self.graph = Graph()
        self.gaugeDim = 100
        self.gauge = vigra.VigraArray((self.gaugeDim,)*3,axistags=vigra.VigraArray.defaultAxistags('xyc'))
        for i in range(self.gauge.shape[2]):
            self.gauge[:,:,i] = i
            
    def expandByShape(self,start,stop,shape,inputShape):
        """
        extend a roi by a given in shape
        """
        #TODO: Warn if bounds are exceeded
        dim = len(start)
        if type(shape == int):
            tmp = shape
            shape = numpy.zeros(dim).astype(int)
            shape[:] = tmp
        tmpStart = [x-s for x,s in zip(start,shape)]
        tmpStop = [x+s for x,s in zip(stop,shape)]
        start = [max(t,i) for t,i in zip(tmpStart,numpy.zeros_like(inputShape))]
        stop = [min(t,i) for t,i in zip(tmpStop,inputShape)]
        return start,stop
    
    def trimChannel(self,start,stop,cPerC,cIndex):
        start = [start[i]/cPerC if i == cIndex else start[i] for i in range(len(start))]
        stop = [stop[i]/cPerC if i==cIndex else stop[i] for i in range(len(stop))]
        return start,stop
    
    def setStartToZero(self,start,stop):
        start = [0]*len(start)
        stop = [end-begin for begin,end in zip(start,stop)]
        start = TinyVector(start)
        stop = TinyVector(stop)
        return start,stop


    def generalOperatorTest(self,operator,sigma1,sigma2=None):
        for dim in self.testDimensions:
            testArray = vigra.VigraArray((10,)*len(dim),axistags=vigra.VigraArray.defaultAxistags(dim))
            operator.inputs["Input"].setValue(testArray)
            operator.inputs["Sigma"].setValue(sigma1)
            if sigma2 is not None:
                operator.inputs["Sigma2"].setValue(sigma2)
            for i,j in [(i,j) for i,j in itertools.permutations(range(0,10),2) if i<j]:
                start = [i]*len(dim)
                stop = [j]*len(dim)
                operator.outputs["Output"](start,stop).wait()
                
    def visualTest(self,operator,sigma1,sigma2=None):
        start,stop  = [200,200,0],[400,400,1]
        testArray = vigra.VigraArray((400,400,3))
        roiResult = vigra.VigraArray(tuple([sto-sta for sta,sto in zip(start,stop)]))
        testArray[100:300,100:300,0] = 1
        testArray[200:300,200:300,1] = 1
        testArray[100:200,100:200,2] = 1
        operator.inputs["Input"].setValue(testArray)
        operator.inputs["Sigma"].setValue(sigma1)
        if sigma2 is not None:
            operator.inputs["Sigma2"].setValue(sigma2)
        wholeResult = operator.outputs["Output"]().wait()
        wholeResult = wholeResult[:,:,0:3]
        roiResult[:,:,0:1] = operator.outputs["Output"](start,stop).wait()
        vigra.impex.writeImage(testArray,operator.name+'before.png')
        vigra.impex.writeImage(wholeResult,operator.name+'afterWhole.png')
        vigra.impex.writeImage(roiResult,operator.name+'afterRoi.png')
                
    def test_GaussianSmoothing(self):
        opGaussianSmoothing = OpGaussianSmoothing(self.graph)
        self.generalOperatorTest(opGaussianSmoothing, 2.0)
        self.visualTest(opGaussianSmoothing, 2.0)
    
    def test_DifferenceOfGaussians(self):
        opDifferenceOfGaussians = OpDifferenceOfGaussians(self.graph)
        self.generalOperatorTest(opDifferenceOfGaussians, 2.0, 3.0)
        self.visualTest(opDifferenceOfGaussians, 2.0)
    
    def test_LaplacianOfGaussian(self):
        opLaplacianOfGaussian = OpLaplacianOfGaussian(self.graph)
        self.generalOperatorTest(opLaplacianOfGaussian, 2.0)
        self.visualTest(opLaplacianOfGaussian,2.0)
        
    def test_GaussianGradientMagnitude(self):
        opGaussianGradientMagnitude = OpGaussianGradientMagnitude(self.graph)
        self.generalOperatorTest(opGaussianGradientMagnitude, 2.0)
        self.visualTest(opGaussianGradientMagnitude,2.0)
        
    def test_StructureTensorEigenvalues(self):
        opStructureTensorEigenvalues = OpStructureTensorEigenvalues(self.graph)
        self.generalOperatorTest(opStructureTensorEigenvalues, 1.5,2.0)
        self.visualTest(opStructureTensorEigenvalues, 1.5, 2.0)

    def test_HessianOfGaussian(self):
        opHessianOfGaussian = OpHessianOfGaussian(self.graph)
        self.generalOperatorTest(opHessianOfGaussian, 2.0)
        self.visualTest(opHessianOfGaussian, 2.0)
        
    def test_HessianOfGaussianEigenvalues(self):
        opHessianOfGaussianEigenvalues = OpHessianOfGaussianEigenvalues(self.graph)
        self.generalOperatorTest(opHessianOfGaussianEigenvalues,2.0)
        self.visualTest(opHessianOfGaussianEigenvalues,2.0)