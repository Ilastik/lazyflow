import numpy as np
import vigra

import unittest

from lazyflow.graph import Graph
from lazyflow.operators import OpLabelVolume


class TestVigra(unittest.TestCase):

    def setUp(self):
        self.method = np.asarray(['vigra'], dtype=np.object)

    def testSimpleUsage(self):
        vol = np.random.randint(255, size=(1000, 100, 10))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')

        op = OpLabelVolume(graph=Graph())
        op.Method.setValue(self.method)
        op.Input.setValue(vol)

        out = op.Output[...].wait()

        assert np.all(vol.shape == out.shape)

    def testCorrectLabeling(self):
        vol = np.zeros((1000, 100, 10))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')

        vol[20:40, 10:30, 2:4] = 1

        op = OpLabelVolume(graph=Graph())
        op.Method.setValue(self.method)
        op.Input.setValue(vol)

        out = op.Output[...].wait()
        tags = op.Output.meta.getTaggedShape()
        out = vigra.taggedView(out, axistags="".join([s for s in tags]))

        assertEquivalentLabeling(vol, out)

    def testMultiDim(self):
        vol = np.zeros((82, 70, 75, 5, 5), dtype=np.uint8)
        vol = vigra.taggedView(vol, axistags='xyzct')

        blocks = np.zeros(vol.shape, dtype=np.uint8)
        blocks[30:50, 40:60, 50:70, 2:4, 3:5] = 1
        blocks[30:50, 40:60, 50:70, 2:4, 0:2] = 2
        blocks[60:70, 30:40, 10:33, :, :] = 3

        vol[blocks == 1] = 255
        vol[blocks == 2] = 255
        vol[blocks == 3] = 255

        op = OpLabelVolume(graph=Graph())
        op.Method.setValue(self.method)
        op.Input.setValue(vol)

        out = op.Output[...].wait()
        tags = op.Output.meta.getTaggedShape()
        print(tags)
        out = vigra.taggedView(out, axistags="".join([s for s in tags]))

        vigra.writeHDF5(vol, '/tmp/in.h5', '/volume/data')
        vigra.writeHDF5(out, '/tmp/out.h5', '/volume/data')

        for c in range(out.shape[3]):
            for t in range(out.shape[4]):
                print("t={}, c={}".format(t, c))
                assertEquivalentLabeling(blocks[..., c, t], out[..., c, t])


def assertEquivalentLabeling(x, y):
    assert np.all(x.shape == y.shape),\
        "Shapes do not agree ({} vs {})".format(x.shape, y.shape)

    # identify labels used in x
    labels = set(x.flat)
    for label in labels:
        if label == 0:
            continue
        idx = np.where(x == label)
        block = y[idx]
        # check that labels are the same
        an_index = [a[0] for a in idx]
        print("Inspecting block of shape {} at".format(block.shape))
        print(an_index)
        assert np.all(block == block[0]),\
            "Block at {} has multiple labels".format(an_index)
        # check that nothing else is labeled with this label
        m = block.size
        n = len(np.where(y == block[0])[0])
        assert m == n, "Label {} is used somewhere else.".format(label)

