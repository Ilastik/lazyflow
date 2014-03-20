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


from os.path import join
from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

drtile = Extension('lazyflow.drtile.drtile',
                   sources = [join('lazyflow','drtile', 'drtile.cpp')],
                   libraries  = ['boost_python'],
                   include_dirs = get_numpy_include_dirs(),
                   language = 'c++')

packages = ['lazyflow', 'lazyflow.drtile', 'lazyflow.request',
            'lazyflow.utility', 'lazyflow.utility.io', 'lazyflow.operators',
            'lazyflow.operators.ioOperators', 'lazyflow.tools']


setup(name = "lazyflow",
      version = "0.1",
      scripts = [join('bin', 'exportRoi.py')],
      packages = packages,
      ext_modules = [drtile],
      install_requires = ['greenlet', 'psutil', 'h5py'],
      author = "Christoph Straehle",
      author_email = "christoph.straehle@iwr.uni-heidelberg.de",
      description = "Lazyflow - graph based lazy numpy data flows",
      license = "BSD",
      keywords = "graph numpy dataflow",
      url = "http://ilastik.org/lazyflow")
