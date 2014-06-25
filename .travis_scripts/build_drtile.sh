#!/bin/sh

# First arg is the virtual_env path
VIRTUAL_ENV=$1

# Second arg is path to lazyflow (optional)
# If not provided, we assume that we started in the lazyflow directory already.
if [ $# -eq 2 ]
then
  # cd to lazyflow
  cd $2
fi

python setup.py build_ext
