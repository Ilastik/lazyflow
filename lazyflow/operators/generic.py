from lazyflow.graph import *
from lazyflow import roi 


def axisTagObjectFromFlag(flag):
    
    if flag in ['x','y','z']:
        type=vigra.AxisType.Space
    elif flag=='c':
        type=vigra.AxisType.Channel
    elif flag=='t':
        type=vigra.AxisType.Time
    else:
        print "Requested flag", str(flag)
        raise
    
    return vigra.AxisTags(vigra.AxisInfo(flag,type)) 


def axisType(flag):
    if flag in ['x','y','z']:
        return vigra.AxisType.Space
    elif flag=='c':
        return vigra.AxisType.Channels
    
    elif flag=='t':
        return vigra.AxisType.Time
    else:
        raise


def axisTagsToString(axistags):
    res=[]
    for axistag in axistags:
        res.append(axistag.key)
    return res
    
    


def getSubKeyWithFlags(key,axistags,axisflags):
    assert len(axistags)==len(key)
    assert len(axisflags)<=len(key)
    
    d=dict(zip(axisTagsToString(axistags),key))

    newKey=[]
    for flag in axisflags:
        slice=d[flag]
        newKey.append(slice)
    
    return tuple(newKey)
    

def popFlagsFromTheKey(key,axistags,flags):
    d=dict(zip(axisTagsToString(axistags),key))
     
    newKey=[]
    for flag in axisTagsToString(axistags):
        if flag not in flags:
            slice=d[flag]
            newKey.append(slice)
    
    return newKey 





class OpMultiArraySlicer(Operator):
    inputSlots = [InputSlot("Input"),InputSlot('AxisFlag')]
    outputSlots = [MultiOutputSlot("Slices",level=1)]

    name = "Multi Array Slicer"
    category = "Misc"
    
    def notifyConnectAll(self):
        
        dtype=self.inputs["Input"].dtype
        flag=self.inputs["AxisFlag"].value
        
        indexAxis=self.inputs["Input"].axistags.index(flag)
        outshape=list(self.inputs["Input"].shape)
        n=outshape.pop(indexAxis)
        outshape=tuple(outshape)
        
        outaxistags=copy.copy(self.inputs["Input"].axistags) 
        
        del outaxistags[flag]
    
        self.outputs["Slices"].resize(n)
        
        for o in self.outputs["Slices"]:
            o._dtype=dtype
            o._axistags=copy.copy(outaxistags)
            o._shape=outshape 
        
            
    def getSubOutSlot(self, slots, indexes, key, result):
        
        #print "SLICER: key", key, "indexes[0]", indexes[0], "result", result.shape
        
        start,stop=roi.sliceToRoi(key,self.outputs["Slices"][indexes[0]].shape)
        
        oldstart,oldstop=start,stop
        
        start=list(start)
        stop=list(stop)
        
        flag=self.inputs["AxisFlag"].value
        indexAxis=self.inputs["Input"].axistags.index(flag)
        
        start.insert(indexAxis,indexes[0])
        stop.insert(indexAxis,indexes[0])
        
        newKey=roi.roiToSlice(numpy.array(start),numpy.array(stop))
        
        ttt = self.inputs["Input"][newKey].allocate().wait()
        
        writeKey = [slice(None, None, None) for k in key]
        writeKey.insert(indexAxis, 0)
        writeKey = tuple(writeKey)

        result[:]=ttt[writeKey ]#+ (0,)]


class OpMultiArraySlicer2(Operator):
    #FIXME: This operator return a sigleton in the sliced direction
    #Should be integrated with the above one to have a more consistent notation
    inputSlots = [InputSlot("Input"),InputSlot('AxisFlag')]
    outputSlots = [MultiOutputSlot("Slices",level=1)]

    name = "Multi Array Slicer"
    category = "Misc"
    
    def notifyConnectAll(self):
        
        dtype=self.inputs["Input"].dtype
        flag=self.inputs["AxisFlag"].value
        
        indexAxis=self.inputs["Input"].axistags.index(flag)
        outshape=list(self.inputs["Input"].shape)
        n=outshape.pop(indexAxis)
        
        
        outshape.insert(indexAxis, 1)
        outshape=tuple(outshape)
        
        outaxistags=copy.copy(self.inputs["Input"].axistags) 
        
        del outaxistags[flag]
    
        self.outputs["Slices"].resize(n)
        
        for o in self.outputs["Slices"]:
            o._dtype=dtype
            o._axistags=copy.copy(outaxistags)
            o._shape=outshape 
        
            
    def getSubOutSlot(self, slots, indexes, key, result):
        
        outshape = self.outputs["Slices"][indexes[0]].shape
            
        
        start,stop=roi.sliceToRoi(key,outshape)
        oldstart,oldstop=start,stop
        
        start=list(start)
        stop=list(stop)
        
        flag=self.inputs["AxisFlag"].value
        indexAxis=self.inputs["Input"].axistags.index(flag)
        
        start.pop(indexAxis)
        stop.pop(indexAxis)
        
        start.insert(indexAxis,indexes[0])
        stop.insert(indexAxis,indexes[0])
        
        
        newKey=roi.roiToSlice(numpy.array(start),numpy.array(stop))
      
        ttt = self.inputs["Input"][newKey].allocate().wait()
        result[:]=ttt[:]

class OpMultiArrayStacker(Operator):
    inputSlots = [MultiInputSlot("Images"), InputSlot("AxisFlag"), InputSlot("AxisIndex")]
    outputSlots = [OutputSlot("Output")]

    name = "Multi Array Stacker"
    description = "Stack inputs on any axis, including the ones which are not there yet"
    category = "Misc"

    def notifySubConnect(self, slots, indexes):
        
        if self.inputs["AxisFlag"].partner is None or self.inputs["AxisIndex"].partner is None:
            return
        else :
            self.setRightShape()
        
    def notifyConnectAll(self):
        #This function is needed so that we don't depend on the order of connections.
        #If axis flag or axis index is connected after the input images, the shape is calculated
        #here
        self.setRightShape()
        
    def setRightShape(self):
        c = 0
        flag = self.inputs["AxisFlag"].value
        axistype = axisType(flag)
        axisindex = self.inputs["AxisIndex"].value
        
        for inSlot in self.inputs["Images"]:
            inTagKeys = [ax.key for ax in inSlot.axistags]
            if inSlot.partner is not None:
                self.outputs["Output"]._dtype = inSlot.dtype
                self.outputs["Output"]._axistags = copy.copy(inSlot.axistags)
                #indexAxis=inSlot.axistags.index(flag)
               
                outTagKeys = [ax.key for ax in self.outputs["Output"]._axistags]
                
                if not flag in outTagKeys:
                    self.outputs["Output"]._axistags.insert(axisindex, vigra.AxisInfo(flag, axisType(flag)))
                if flag in inTagKeys:
                    c += inSlot.shape[inSlot.axistags.index(flag)]
                else:
                    c += 1
                    
        if len(self.inputs["Images"]) > 0:
            newshape = list(self.inputs["Images"][0].shape)
            if flag in inTagKeys:
                #here we assume that all axis are present
                newshape[axisindex]=c
            else:
                newshape.insert(axisindex, c)
            self.outputs["Output"]._shape=tuple(newshape)
        else:
            self.outputs["Output"]._shape = None



    def getOutSlot(self, slot, key, result):
        cnt = 0
        written = 0
        start, stop = roi.sliceToRoi(key, self.outputs["Output"].shape)
        assert (stop<=self.outputs["Output"].shape).all()
        axisindex = self.inputs["AxisIndex"].value
        flag = self.inputs["AxisFlag"].value
        #ugly-ugly-ugly
        oldkey = list(key)
        oldkey.pop(axisindex)
        #print "STACKER: ", flag, axisindex
        #print "requesting an outslot from stacker:", key, result.shape
        #print "input slots total: ", len(self.inputs['Images'])
        requests = []
        
        
        for i, inSlot in enumerate(self.inputs['Images']):
            if inSlot.connected():
                req = None
                inTagKeys = [ax.key for ax in inSlot.axistags]
                if flag in inTagKeys:
                    slices = inSlot.shape[axisindex]
                    if cnt + slices >= start[axisindex] and start[axisindex]-cnt<slices and start[axisindex]+written<stop[axisindex]:
                        begin = 0
                        if cnt < start[axisindex]:
                            begin = start[axisindex] - cnt
                        end = slices
                        if cnt + end > stop[axisindex]:
                            end -= cnt + end - stop[axisindex]
                        key_ = copy.copy(oldkey)
                        key_.insert(axisindex, slice(begin, end, None))
                        reskey = [slice(None, None, None) for x in range(len(result.shape))]
                        reskey[axisindex] = slice(written, written+end-begin, None)
                        
                        req = inSlot[tuple(key_)].writeInto(result[tuple(reskey)])
                        written += end - begin
                    cnt += slices
                else:
                    if cnt>=start[axisindex] and start[axisindex] + written < stop[axisindex]:
                        #print "key: ", key, "reskey: ", reskey, "oldkey: ", oldkey
                        #print "result: ", result.shape, "inslot:", inSlot.shape
                        reskey = [slice(None, None, None) for s in oldkey]
                        reskey.insert(axisindex, written)
                        destArea = result[tuple(reskey)]
                        req = inSlot[tuple(oldkey)].writeInto(destArea)
                        written += 1
                    cnt += 1
                
                if req is not None:
                    requests.append(req)
        
        for r in requests:
            r.wait()


class OpSingleChannelSelector(Operator):
    name = "SingleChannelSelector"
    description = "Select One channel from a Multichannel Image"
    
    inputSlots = [InputSlot("Input"),InputSlot("Index",stype='integer')]
    outputSlots = [OutputSlot("Output")]
    
    def notifyConnectAll(self):
        self.outputs["Output"]._dtype =self.inputs["Input"].dtype 
        self.outputs["Output"]._shape = self.inputs["Input"].shape[:-1]+(1,)
        self.outputs["Output"]._axistags = self.inputs["Input"].axistags
        
        
        
    def getOutSlot(self, slot, key, result):
        
        index=self.inputs["Index"].value
        #FIXME: check the axistags for a multichannel image
        assert self.inputs["Input"].shape[-1]>index, "Requested channel out of Range"       
        newKey = key[:-1]
        newKey += (slice(0,self.inputs["Input"].shape[-1],None),)        
        
        
        im=self.inputs["Input"][newKey].allocate().wait()
        
        
        result[...,0]=im[...,index]
        

    def notifyDirty(self, slot, key):
        key = key[:-1] + (slice(0,1,None),)
        self.outputs["Output"].setDirty(key)   


        
        
class OpSubRegion(Operator):
    name = "OpSubRegion"
    description = "Select a region of interest from an numpy array"
    
    inputSlots = [InputSlot("Input"), InputSlot("Start"), InputSlot("Stop")]
    outputSlots = [OutputSlot("Output")]
    
    def notifyConnectAll(self):
        start = self.inputs["Start"].value
        stop = self.inputs["Stop"].value
        assert isinstance(start, tuple)
        assert isinstance(stop, tuple)
        assert len(start) == len(self.inputs["Input"].shape)
        assert len(start) == len(stop)
        assert (numpy.array(stop)>= numpy.array(start)).all()
        
        temp = tuple(numpy.array(stop) - numpy.array(start))        
        #drop singleton dimensions
        outShape = ()        
        for e in temp:
            if e > 0:
                outShape = outShape + (e,)
                
        self.outputs["Output"]._shape = outShape
        self.outputs["Output"]._axistags = self.inputs["Input"].axistags
        self.outputs["Output"]._dtype = self.inputs["Input"].dtype

    def getOutSlot(self, slot, key, resultArea):
        start = self.inputs["Start"].value
        stop = self.inputs["Stop"].value
        
        temp = tuple()
        for i in xrange(len(start)):
          if stop[i] - start[i] > 0:
            temp += (stop[i]-start[i],)

        readStart, readStop = sliceToRoi(key, temp)
        


        newKey = ()
        resultKey = ()
        i = 0
        i2 = 0
        for kkk in xrange(len(start)):
            e = stop[kkk] - start[kkk]
            if e > 0:
                newKey += (slice(start[i2] + readStart[i], start[i2] + readStop[i],None),)
                resultKey += (slice(0,temp[i2],None),)
                i +=1
            else:
                newKey += (slice(start[i2], start[i2], None),)
                resultKey += (0,)
            i2 += 1
      
        res = self.inputs["Input"][newKey].allocate().wait()
        resultArea[:] = res[resultKey]
        
 
 
 
class OpMultiArrayMerger(Operator):
    inputSlots = [MultiInputSlot("Inputs"),InputSlot('MergingFunction')]
    outputSlots = [OutputSlot("Output")]

    name = "Merge Multi Arrays based on a variadic merging function"
    category = "Misc"
    
    def notifyConnectAll(self):
        
        shape=self.inputs["Inputs"][0].shape
        axistags=copy.copy(self.inputs["Inputs"][0].axistags)
        
        
        self.outputs["Output"]._shape = shape
        self.outputs["Output"]._axistags = axistags
        self.outputs["Output"]._dtype = self.inputs["Inputs"][0].dtype
        
        
        
        for input in self.inputs["Inputs"]:
            assert input.shape==shape, "Only possible merging consistent shapes"
            assert input.axistags==axistags, "Only possible merging same axistags"
                       
        
            
    def getOutSlot(self, slot, key, result):
        
        requests=[]
        for input in self.inputs["Inputs"]:
            requests.append(input[key].allocate())
        
        data=[]
        for req in requests:
            data.append(req.wait())
        
        fun=self.inputs["MergingFunction"].value
        
        result[:]=fun(data)
        
        
        
class OpPixelOperator(Operator):
    name = "OpPixelOperator"
    description = "simple pixel operations"

    inputSlots = [InputSlot("Input"), InputSlot("Function")]
    outputSlots = [OutputSlot("Output")]    
    
    def notifyConnectAll(self):

        inputSlot = self.inputs["Input"]

        self.function = self.inputs["Function"].value
   
        self.outputs["Output"]._shape = inputSlot.shape
        self.outputs["Output"]._dtype = inputSlot.dtype
        self.outputs["Output"]._axistags = inputSlot.axistags
        

        
    def getOutSlot(self, slot, key, result):
        
        
        matrix = self.inputs["Input"][key].allocate().wait()
        matrix = self.function(matrix)
        
        result[:] = matrix[:]


    def notifyDirty(selfut,slot,key):
        self.outputs["Output"].setDirty(key)

    @property
    def shape(self):
        return self.outputs["Output"]._shape
    
    @property
    def dtype(self):
        return self.outputs["Output"]._dtype
        
        
