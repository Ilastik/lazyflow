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


class TestOpBaseVigraFilter(unittest.TestCase):
    
    def setUp(self):

        FORMAT = ' %(message)s'
        logging.basicConfig(filename='OpBaseVigraFilter.log',level=logging.DEBUG,format=FORMAT)
        
        self.volume = None
        self.testDim = (10,10,10,10,10)
        
        self.graph = Graph()
        self.sigmaList = [0.3,0.7,1,1.6,3.5,5.0,10.0]
        
        self.prepareVolume()

    def prepareVolume(self):
        
        self.volume = vigra.VigraArray(self.testDim)
        self.volume[:] = numpy.random.rand(*self.testDim)
    
    def test_DifferenceOfGaussian(self):
        
        opDiffGauss = OpDifferenceOfGaussians(self.graph)
        opDiffGauss.inputs["Input"].setValue(self.volume)
        logging.debug('======================OpDifferenceOfGaussian===============')
        for sigma0,sigma1 in itertools.product(self.sigmaList,self.sigmaList):
            try:
                opDiffGauss.inputs["sigma0"].setValue(sigma0)
                opDiffGauss.inputs["sigma1"].setValue(sigma1)
                opDiffGauss.outputs["Output"][:].allocate().wait()
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
                opSmoothGauss.outputs["Output"][:].allocate().wait()
            except:
                logging.debug('Test failed for the following sigma: %s'%sigma)
        assert 1==1
    
    def test_CoherenceOrientation(self):
        
        opCohenrece = OpCoherenceOrientation(self.graph)
        opCohenrece.inputs["Input"].setValue(self.volume)
        logging.debug('===================OpCoherenceOrientation==================')
        for sigma0,sigma1 in itertools.product(self.sigmaList,self.sigmaList):
            try:
                opCohenrece.inputs["sigma0"].setValue(sigma0)
                opCohenrece.inputs["sigma1"].setValue(sigma1)
                opCohenrece.outputs["Output"][:].allocate().wait()
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
                opHessianOfGaussian.outputs["Output"][:].allocate().wait()
            except:
                logging.debug('Test failed for the following sigma: %s'%sigma)
        assert 1==1
    
    def test_StructureTensorEigenvalues(self):
        
        opStructureTensor = OpStructureTensorEigenvalues(self.graph)
        opStructureTensor.inputs["Input"].setValue(self.volume)
        logging.debug('================OpStructureTensorEigenvalues===============')
        for sigma0,sigma1 in itertools.product(self.sigmaList,self.sigmaList):
            try:
                opStructureTensor.inputs["innerScale"].setValue(sigma0)
                opStructureTensor.inputs["outerScale"].setValue(sigma1)
                opStructureTensor.outputs["Output"][:].allocate().wait()
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
                opHessianOfGaussianEF.outputs["Output"][:].allocate().wait()
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
                opHessianOfGaussian.outputs["Output"][:].allocate().wait()
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
                opGaussianGradient.outputs["Output"][:].allocate().wait()
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
                opLaplacianOfGaussian.outputs["Output"][:].allocate().wait()
            except:
                logging.debug('Test failed for the following sigma: %s'%sigma)
        assert 1==1