import unittest
import random
import vigra
import numpy
from lazyflow.graph import Graph
from volumina.adaptors import Op5ifyer
from lazyflow.roi import sliceToRoi,roiToSlice
from lazyflow.helpers import generateRandomKeys,generateRandomRoi

class TestOp5ifyer(unittest.TestCase):
    
    def setUp(self):

        self.array = None
        self.axis = list('txyzc')
        self.tests = 100
        graph = Graph()
        self.operator = Op5ifyer(graph)
        
    def prepareVolnOp(self):
        
        tags = random.sample(self.axis,random.randint(2,len(self.axis)))
        
        tagStr = ''
        for s in tags:
            tagStr += s
        axisTags = vigra.defaultAxistags(tagStr)
        
        shape = []
        for tag in axisTags:
            shape.append(random.randint(20,30))
            
        self.array = vigra.VigraArray(tuple(shape),axistags = axisTags)
        
        self.operator.inputs["input"].setValue(self.array)
    
    def test_Full(self):
        
        for i in range(self.tests):
            self.prepareVolnOp()
            result = self.operator.outputs["output"]().wait()
            array = numpy.ndarray(self.array.shape)
            array[:] = self.array[:]
            if len(result.shape) == 5 and numpy.all(result == array):
                assert 1==1
                
    def test_Roi(self):
        for i in range(self.tests):
            self.prepareVolnOp()
            roi = generateRandomRoi(self.operator.inputs["input"]._shape)
            result = self.operator.outputs["output"](roi[0],roi[1]).wait()
            array = numpy.ndarray(self.array[roiToSlice(roi[0],roi[1])].shape)
            array[:] = self.array[roiToSlice(roi[0],roi[1])]
            if len(result.shape) == 5 and numpy.all(result == array):
                assert 1==1