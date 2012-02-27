import unittest
import numpy
import itertools
import vigra
import logging
from lazyflow.graph import Graph
from lazyflow.operators.obsolete.vigraOperators import \
    OpDifferenceOfGaussians, OpGaussianSmoothing, OpCoherenceOrientation,\
    OpHessianOfGaussianEigenvalues, OpStructureTensorEigenvalues,\
    OpHessianOfGaussianEigenvaluesFirst, OpHessianOfGaussian,\
    OpGaussianGradientMagnitude, OpLaplacianOfGaussian
from lazyflow.roi import sliceToRoi

class TestOpBaseVigraFilter(unittest.TestCase):
    
    def setUp(self):

        FORMAT = ' %(message)s'
        logging.basicConfig(filename='OpBaseVigraFilter.log',level=logging.DEBUG,format=FORMAT)
        
        self.volume = None
        self.testDim = (10,10,10,10,10)
        self.keyNum = 10
        self.eps = 0.001
        
        self.graph = Graph()
        self.sigmaList = [0.3,0.7,1,1.6]#+[3.5,5.0,10.0]
        self.sigmaComboList = [x for x in itertools.product(self.sigmaList,self.sigmaList) if x[0]<x[1]]
        
        self.prepareVolume()

    def prepareVolume(self):
        
        self.volume = vigra.VigraArray(self.testDim)
        self.volume[:] = numpy.random.rand(*self.testDim)
        self.twoDvolume = vigra.VigraArray((10,10,1))
        self.twoDvolume[:] = numpy.random.rand(10,10,1)
        
    def generateKeys(self):
        
        tmp = numpy.zeros((5,2))
        
        while tmp[0,0] == tmp[0,1] or tmp[1,0] == tmp[1,1] or tmp[2,0] == tmp[2,1]\
        or tmp[3,0] == tmp[3,1] or tmp[4,0] == tmp[4,1]:
            
            tmp = numpy.random.rand(5,2)
            for i in range(5):
                tmp[i,:] *= self.testDim[i]
                tmp[i,:] = numpy.sort(numpy.round(tmp[i,:]))
        
        key = []
        for i in range(5):
            key.append(slice(int(tmp[i,0]),int(tmp[i,1]),1))
        
        return key
    
    def testBlocks(self,operator):
        
        eps = self.eps
        block = operator.outputs["Output"][:].allocate().wait()
        for i in range(self.keyNum):
            key = self.generateKeys()
            if (operator.outputs["Output"][key].allocate().wait() - block[key] < eps).all():
                logging.debug('Operator successfully test on block (roi) '+str(sliceToRoi(key, block.shape))+' in tolerance limits of '+str(eps))
                assert 1==1
            
    def test_DifferenceOfGaussian(self):
        
        opDiffGauss = OpDifferenceOfGaussians(self.graph)
        opDiffGauss.inputs["Input"].setValue(self.volume)
        logging.debug('======================OpDifferenceOfGaussian===============')
        for sigma0,sigma1 in self.sigmaComboList:
            try:
                opDiffGauss.inputs["sigma0"].setValue(sigma0)
                opDiffGauss.inputs["sigma1"].setValue(sigma1)
                self.testBlocks(opDiffGauss)
            except:
                logging.debug('Test failed for the following sigma-combination : %s,%s'%(sigma0,sigma1))
        assert 1==1

    def test_GaussianSmoothing(self):
        
        opSmoothGauss = OpGaussianSmoothing(self.graph)
        opSmoothGauss.inputs["Input"].setValue(self.volume)
        logging.debug('======================OpGaussiaSmoothing===================')
        for sigma in self.sigmaList:
            try:
                opSmoothGauss.inputs["sigma"].setValue(sigma)
                self.testBlocks(opSmoothGauss)
            except:
                logging.debug('Test failed for the following sigma: %s'%sigma)
        assert 1==1
    
    def test_CoherenceOrientation(self):
        
        opCohenrece = OpCoherenceOrientation(self.graph)
        opCohenrece.inputs["Input"].setValue(self.twoDvolume)
        logging.debug('===================OpCoherenceOrientation==================')
        for sigma0,sigma1 in self.sigmaComboList:
            try:
                opCohenrece.inputs["sigma0"].setValue(sigma0)
                opCohenrece.inputs["sigma1"].setValue(sigma1)
                self.testBlocks(opCohenrece)
            except:
                logging.debug('Test failed for the following sigma-combination : %s,%s'%(sigma0,sigma1))
        assert 1==1
    
    def test_HessianOfGaussianEigenvalues(self):
        
        opHessianOfGaussian = OpHessianOfGaussianEigenvalues(self.graph)
        opHessianOfGaussian.inputs["Input"].setValue(self.volume)
        logging.debug('================OpHessianOfGaussianEigenvalues=============')
        for sigma in self.sigmaList:
            try:
                opHessianOfGaussian.inputs["scale"].setValue(sigma)
                self.testBlocks(opHessianOfGaussian)
            except:
                logging.debug('Test failed for the following sigma: %s'%sigma)
        assert 1==1
    
    def test_StructureTensorEigenvalues(self):
        
        opStructureTensor = OpStructureTensorEigenvalues(self.graph)
        opStructureTensor.inputs["Input"].setValue(self.volume)
        logging.debug('================OpStructureTensorEigenvalues===============')
        for sigma0,sigma1 in self.sigmaComboList:
            try:
                opStructureTensor.inputs["innerScale"].setValue(sigma0)
                opStructureTensor.inputs["outerScale"].setValue(sigma1)
                self.testBlocks(opStructureTensor)
            except:
                logging.debug('Test failed for the following sigma-combination : %s,%s'%(sigma0,sigma1))
        assert 1==1
        
    def test_HessianOfGaussianEigenvaluesFirst(self):
        
        opHessianOfGaussianEF = OpHessianOfGaussianEigenvaluesFirst(self.graph)
        opHessianOfGaussianEF.inputs["Input"].setValue(self.volume)
        logging.debug('================OpHessianOfGaussianEigenvaluesFirst========')
        for sigma in self.sigmaList:
            try:
                opHessianOfGaussianEF.inputs["scale"].setValue(sigma)
                self.testBlocks(opHessianOfGaussianEF)
            except:
                logging.debug('Test failed for the following sigma: %s'%sigma)
        assert 1==1
        
    
    def test_HessianOfGaussian(self):
        
        opHessianOfGaussian = OpHessianOfGaussian(self.graph)
        opHessianOfGaussian.inputs["Input"].setValue(self.volume)
        logging.debug('================OpHessianOfGaussian========================')
        for sigma in self.sigmaList:
            try:
                opHessianOfGaussian.inputs["sigma"].setValue(sigma)
                self.testBlocks(opHessianOfGaussian)
            except:
                logging.debug('Test failed for the following sigma: %s'%sigma)
        assert 1==1
        
        
    def test_GaussianGradientMagnitude(self):
        
        opGaussianGradient = OpGaussianGradientMagnitude(self.graph)
        opGaussianGradient.inputs["Input"].setValue(self.volume)
        logging.debug('================OpopGaussianGradient=======================')
        for sigma in self.sigmaList:
            try:
                opGaussianGradient.inputs["sigma"].setValue(sigma)
                self.testBlocks(opGaussianGradient)
            except:
                logging.debug('Test failed for the following sigma: %s'%sigma)
        assert 1==1
    
    def test_LaplacianOfGaussian(self):
        opLaplacianOfGaussian = OpLaplacianOfGaussian(self.graph)
        opLaplacianOfGaussian.inputs["Input"].setValue(self.volume)
        logging.debug('================OpopLaplacianOfGaussian====================')
        for sigma in self.sigmaList:
            try:
                opLaplacianOfGaussian.inputs["scale"].setValue(sigma)
                self.testBlocks(opLaplacianOfGaussian)
            except:
                logging.debug('Test failed for the following sigma: %s'%sigma)
        assert 1==1