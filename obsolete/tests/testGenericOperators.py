import vigra
import threading
from lazyflow.graph import *
import copy


from lazyflow import operators
from lazyflow.operators import *
import numpy
from numpy.testing import *




if __name__=="__main__":


    g = Graph()



    op1=OpArrayPiper(graph=g)
    op1.inputs["Input"].setValue(numpy.zeros((2,2))+10)

    op2=OpArrayPiper(graph=g)
    op2.inputs["Input"].setValue(numpy.zeros((2,2))+20)

    multi=Op20ToMulti(graph=g)

    multi.inputs["Input00"].connect(op1.outputs["Output"])
    multi.inputs["Input01"].connect(op2.outputs["Output"])




    merge=OpMultiArrayMerger(graph=g)
    merge.inputs["Inputs"].connect(multi.outputs["Outputs"])
    merge.inputs["MergingFunction"].setValue(sum)


    res=merge.outputs["Output"][:].wait()

    assert (res==30).all()
