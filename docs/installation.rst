Installation
============
lazyflow requires python 2.7, numpy, vigra, greenlet and psutil packages:


``sudo easy_install numpy greenlet psutil``

Vigra can be obtained from  http://github.com/ukoethe/vigra
Optional requirements for lazyflow are the h5py library

``sudo easy_install h5py``

After installing the prerequisites lazyflow can be installed:

``sudo python setup.py install``

To build the extensions in the source tree:

``python setup.py build_ext --inplace``
