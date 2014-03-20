# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# Copyright 2011-2014, the ilastik developers

from lazyflow.operators.generic import OpSingleChannelSelector
import numpy, vigra
from lazyflow.graph import Graph

class TestSingleChannelSelector():
    def setUp(self):
        self.data2d = numpy.zeros((3, 3, 3))
        self.data2d[:, :, 0] = 1
        self.data2d[:, :, 1] = 2
        self.data2d[:, :, 2] = 3
        
        self.data2d = self.data2d.view(vigra.VigraArray)
        self.data2d.axistags = vigra.defaultAxistags("xyc")
        
        
        self.data3d = numpy.zeros((3, 3, 3, 3))
        self.data3d[:, :, :, 0] = 1
        self.data3d[:, :, :, 1] = 2
        self.data3d[:, :, :, 2] = 3
        
        self.data3d = self.data3d.view(vigra.VigraArray)
        self.data3d.axistags = vigra.defaultAxistags("xyzc")
        
        self.data_bad_channel = numpy.zeros((3, 3, 3, 3))
        self.data_bad_channel[:, :, 0, :] = 1
        self.data_bad_channel[:, :, 1, :] = 2
        self.data_bad_channel[:, :, 2, :] = 3
        
        self.data_bad_channel = self.data_bad_channel.view(vigra.VigraArray)
        self.data_bad_channel.axistags = vigra.defaultAxistags("xycz")
        
    def test2d(self):
        graph = Graph()
        op = OpSingleChannelSelector(graph=graph)
        op.Input.setValue(self.data2d)
        for i in range(2):
            op.Index.setValue(i)
            out = op.Output[:].wait()
            assert numpy.all(out==i+1)
            
    def test3d(self):
        graph = Graph()
        op = OpSingleChannelSelector(graph=graph)
        op.Input.setValue(self.data3d)
        for i in range(2):
            op.Index.setValue(i)
            out = op.Output[:].wait()
            assert numpy.all(out==i+1)
            
    def test_bad_channel_index(self):
        
        graph = Graph()
        op = OpSingleChannelSelector(graph=graph)
        op.Input.setValue(self.data_bad_channel)
        for i in range(2):
            op.Index.setValue(i)
            out = op.Output[:].wait()
            assert numpy.all(out==i+1)
        

if __name__ == "__main__":
    import sys
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    ret = nose.run(defaultTest=__file__)
    if not ret: sys.exit(1)
