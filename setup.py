
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
