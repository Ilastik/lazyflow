###############################################################################
#   lazyflow: data flow based lazy parallel computation framework
#
#       Copyright (C) 2011-2014, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the Lesser GNU General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# See the files LICENSE.lgpl2 and LICENSE.lgpl3 for full text of the
# GNU Lesser General Public License version 2.1 and 3 respectively.
# This information is also available on the ilastik web site at:
#		   http://ilastik.org/license/
###############################################################################

from os.path import join
from distutils.core import setup, Extension

drtile = Extension('lazyflow.drtile.drtile',
                   sources = [join('lazyflow','drtile', 'drtile.cpp')],
                   libraries  = ['boost_python'],
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
