import numpy, vigra
import time
from lazyflow.graph import *
import gc
from lazyflow import roi
import threading

from lazyflow.operators.operators import OpArrayCache, OpArrayPiper, OpMultiArrayPiper
from lazyflow.operators.obsoleteOperators import OpArrayBlockCache, OpArraySliceCache, OpArraySliceCacheBounding

__testing__ = False

from tests.mockOperators import OpA, OpB, OpC

g = Graph()

source0 = OpArrayPiper(graph=g)
source0.inputs["Input"].setValue(numpy.zeros((200,200),dtype = numpy.uint8))

source1 = OpArrayPiper(graph=g)
source1.inputs["Input"].setValue(numpy.zeros((375,50),dtype = numpy.uint8))


opa = OpMultiArrayPiper(graph=g)
opb = OpMultiArrayPiper(graph=g)
opc = OpMultiArrayPiper(graph=g)

opd = OpA(graph=g)
ope = OpB(graph=g)
ope2 = OpB(graph=g)

opa.inputs["MultiInput"].resize(2)
opa.inputs["MultiInput"].connect(source0.outputs["Output"])
opa.inputs["MultiInput"].connect(source1.outputs["Output"])

opb.inputs["MultiInput"].connect(opa.outputs["MultiOutput"])
assert len(opb.outputs["MultiOutput"]) == 2, len(opb.outputs["MultiOutput"])

opd.inputs["Input"].connect(opb.outputs["MultiOutput"][0])
ope.inputs["Input"].connect(opb.outputs["MultiOutput"][1])
ope2.inputs["Input"].connect(ope.outputs["Output"])

assert (opa.outputs["MultiOutput"][0][:,:].wait() == 0).all()
assert (opa.outputs["MultiOutput"][1][:,:].wait() == 0).all()

assert (opb.outputs["MultiOutput"][0][:,:].wait() == 0).all()
assert (opb.outputs["MultiOutput"][0][:,:].wait() == 0).all()

assert (opd.outputs["Output"][:,:].wait() == 0).all()
assert (ope.outputs["Output"][:,:].wait() == 1).all()

opc.inputs["MultiInput"].resize(2)
opc.inputs["MultiInput"][0].connect(opd.outputs["Output"])
opc.inputs["MultiInput"][1].connect(ope2.outputs["Output"])

print "aksjdkajsdkjad", len(opc.outputs["MultiOutput"])

assert (opc.outputs["MultiOutput"][0][:,:].wait() == 0).all(), numpy.nonzero(opc.outputs["MultiOutput"][0][:,:])
assert (opc.outputs["MultiOutput"][1][:,:].wait() == 2).all(), numpy.nonzero(opc.outputs["MultiOutput"][0][:,:] - 2)

g.finalize()
