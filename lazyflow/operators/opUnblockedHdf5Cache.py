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
#           http://ilastik.org/license/
###############################################################################

import time
import collections
import numpy

from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.request import RequestLock
from lazyflow.roi import getIntersection, roiFromShape, roiToSlice, containing_rois

import logging
logger = logging.getLogger(__name__)

class OpUnblockedHdf5Cache(Operator):
    """
    This cache operator stores the results of all requests that pass through 
    it, in exactly the same blocks that were requested.

    - If there are any overlapping requests, then the data for the overlapping portion will 
        be stored multiple times, except for the special case where the new request happens 
        to fall ENTIRELY within an existing block of data.
    - If any portion of a stored block is marked dirty, the entire block is discarded.

    Unlike other caches, this cache does not impose its own blocking on the data.
    Instead, it is assumed that the downstream operators have chosen some reasonable blocking.
    Hopefully the downstream operators are reasonably consistent in the blocks they request data with,
    since every unique result is cached separately.
    """
    Input = InputSlot()
    H5CacheGroup = InputSlot()
    Output = OutputSlot()
    
    def __init__(self, dataset_kwargs, *args, **kwargs):
        super( OpUnblockedHdf5Cache, self ).__init__(*args, **kwargs)
        self._lock = RequestLock()
        self._reader_lock_count = 0
        self._writer_blocks = {}
        self._h5group = None
        dataset_kwargs = dataset_kwargs or { 'compression' : 'lzf' }
        self._dataset_kwargs = dataset_kwargs

    @classmethod
    def _standardize_roi(cls, start, stop):
        # We use rois as dict keys.
        # For comparison purposes, all rois in the dict keys are assumed to be tuple-of-tuples-of-int
        start = tuple(map(int, start))
        stop = tuple(map(int, stop))        
        return (start, stop)

    def setupOutputs(self):
        done = False
        while not done:
            with self._lock:
                if self._reader_lock_count == 0:
                    self._h5group = self.H5CacheGroup.value
                    done = True

        self.Output.meta.assignFrom(self.Input.meta)
    
    def _clear_blocks(self):
        assert not self._writer_blocks
        assert self._reader_lock_count == 0
        keys = self._h5group.keys()
        for k in keys:
            del self._h5group[k]

    def execute(self, slot, subindex, roi, result):
        block_is_cached = False
        has_create_permission = False

        # Standardize roi for usage as dict key
        block_roi = self._standardize_roi( roi.start, roi.stop )
        block_roi_key = str(block_roi)
        block_relative_roi = None

        # determine block status (cached or uncached)
        with self._lock:
            if block_roi_key in self._h5group:
                block_is_cached = True
                self._reader_lock_count += 1
            else:
                # Does this roi happen to fit ENTIRELY within an existing stored block?
                block_rois = map( eval, self._h5group.keys() )
                outer_rois = containing_rois( block_rois, (roi.start, roi.stop) )
                outer_rois_count = len(outer_rois)
                if outer_rois_count == 1:
                    block_roi = self._standardize_roi( *outer_rois[0] )
                    block_roi_key = str(block_roi)
                    block_relative_roi = numpy.array( (roi.start, roi.stop) ) - block_roi[0]
                    block_is_cached = True
                    self._reader_lock_count += 1
                elif outer_rois_count > 1:
                    # handle case correctly; or make sure roi is within single block
                    raise RuntimeError("internal error")
                elif outer_rois_count == 0:
                    # handle writer case
                    if self.Input.meta.dontcache:
                        has_create_permission = True
                    elif block_roi_key in self._writer_blocks:
                        has_create_permission = False
                    else:
                        assert not self.Input.meta.dontcache
                        assert not self._writer_blocks.has_key(block_roi_key)
                        self._writer_blocks[block_roi_key] = True
                        has_create_permission = True

        # block exists; read it in
        if block_is_cached:
            # multiple readers allowed
            assert self._reader_lock_count >= 1
            if block_relative_roi:
                self.Output.stype.copy_data(result, self._h5group[block_roi_key][ roiToSlice(*block_relative_roi) ])
            else:
                self.Output.stype.copy_data(result, self._h5group[block_roi_key])

            # update reader count
            with self._lock:
                self._reader_lock_count -= 1
                assert self._reader_lock_count >= 0

            return


        # create data; not yet stored
        if has_create_permission:
            # Request data now.
            self.Input(roi.start, roi.stop).writeInto(result).block()

            # Does upstream operator says don't cache the data? (e.g., OpCacheFixer)
            if self.Input.meta.dontcache:
                return

            # Store the data (wait until there are no readers).
            while True:
                with self._lock:
                    assert block_roi_key in self._writer_blocks

                    if self._reader_lock_count == 0:
                        self._h5group.create_dataset( block_roi_key, data=result,
                                                      **self._dataset_kwargs )
                        self._h5group.file.flush()
                        del self._writer_blocks[block_roi_key]
                        return

        else:
            # delayed read: wait until block is available
            while True:
                with self._lock:
                    if block_roi_key in self._h5group:
                        assert not self._writer_blocks.has_key(block_roi_key)
                        self.Output.stype.copy_data(result, self._h5group[block_roi_key])
                        return



    def propagateDirty(self, slot, subindex, roi):
        maximum_roi = roiFromShape(self.Input.meta.shape)

        if slot == self.Input:
            assert not self._writer_blocks
            assert self._reader_lock_count == 0

            maximum_roi = self._standardize_roi( *maximum_roi )
            dirty_roi = self._standardize_roi( roi.start, roi.stop )

            if dirty_roi == maximum_roi:
                # Optimize the common case:
                # Everything is dirty, so no need to loop
                self._clear_blocks()
            else:
                # FIXME: This is O(N) for now.
                #        We should speed this up by maintaining a bookkeeping data structure in execute().
                for block_roi_str in self._h5group.keys():
                    block_roi = eval(block_roi_str)
                    if getIntersection(block_roi, dirty_roi, assertIntersect=False):
                        del self._h5group[block_roi_str]

            self._h5group.file.flush()

            self.Output.setDirty( roi )
        else:
            self.Output.setDirty( slice(None) )

