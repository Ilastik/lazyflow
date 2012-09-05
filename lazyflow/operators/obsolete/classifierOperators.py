import numpy

from lazyflow.graph import Operator, InputSlot, OutputSlot, MultiInputSlot, OrderedSignal
from lazyflow.roi import sliceToRoi, roiToSlice
import vigra
import copy

import logging
logger = logging.getLogger(__name__)
traceLogger = logging.getLogger("TRACE." + __name__)
from lazyflow.tracer import traceLogged

class OpTrainRandomForest(Operator):
    name = "TrainRandomForest"
    description = "Train a random forest on multiple images"
    category = "Learning"

    inputSlots = [MultiInputSlot("Images"),MultiInputSlot("Labels"), InputSlot("fixClassifier", stype="bool")]
    outputSlots = [OutputSlot("Classifier")]

    def setupOutputs(self):
        if self.inputs["fixClassifier"].value == False:
            self.outputs["Classifier"]._dtype = object
            self.outputs["Classifier"]._shape = (1,)
            self.outputs["Classifier"]._axistags  = "classifier"
            self.outputs["Classifier"].setDirty((slice(0,1,None),))


    @traceLogged(logger, level=logging.INFO, msg="OpTrainRandomForest: Training Classifier")
    def getOutSlot(self, slot, key, result):
        featMatrix=[]
        labelsMatrix=[]
        for i,labels in enumerate(self.inputs["Labels"]):
            if labels.shape is not None:
                labels=labels[:].allocate().wait()

                indexes=numpy.nonzero(labels[...,0].view(numpy.ndarray))
                #Maybe later request only part of the region?

                image=self.inputs["Images"][i][:].allocate().wait()

                features=image[indexes]
                labels=labels[indexes]

                featMatrix.append(features)
                labelsMatrix.append(labels)


        featMatrix=numpy.concatenate(featMatrix,axis=0)
        labelsMatrix=numpy.concatenate(labelsMatrix,axis=0)

        # TODO: Make treecount configurable via an InputSlot
        RF=vigra.learning.RandomForest(100)
        try:
            RF.learnRF(featMatrix.astype(numpy.float32),labelsMatrix.astype(numpy.uint32))
        except:
            logger.error( "ERROR: could not learn classifier" )
            logger.error( "featMatrix={}, labelsMatrix={}".format(featMatrix, labelsMatrix) )
            logger.error( "featMatrix shape={}, dtype={}".format(featMatrix.shape, featMatrix.dtype) )
            logger.error( "labelsMatrix shape={}, dtype={}".format(labelsMatrix.shape, labelsMatrix.dtype ) )
            raise

        result[0]=RF

    def setInSlot(self, slot, key, value):
        if self.inputs["fixClassifier"].value == False:
            self.outputs["Classifier"].setDirty((slice(0,1,None),))

    def setSubInSlot(self,slots,indexes, key,value):
        if self.inputs["fixClassifier"].value == False:
            self.outputs["Classifier"].setDirty((slice(0,1,None),))

    def notifySubSlotDirty(self, slots, indexes, key):
        if self.inputs["fixClassifier"].value == False:
            self.outputs["Classifier"].setDirty((slice(0,1,None),))

    def notifyDirty(self, slot, key):
        if slot is not self.inputs["fixClassifier"] and self.inputs["fixClassifier"].value == False:
            self.outputs["Classifier"].setDirty((slice(0,1,None),))


class OpTrainRandomForestBlocked(Operator):
    name = "TrainRandomForestBlocked"
    description = "Train a random forest on multiple images"
    category = "Learning"

    inputSlots = [MultiInputSlot("Images"),MultiInputSlot("Labels"), InputSlot("fixClassifier", stype="bool"), \
                  MultiInputSlot("nonzeroLabelBlocks")]
    outputSlots = [OutputSlot("Classifier")]

    WarningEmitted = False

    def __init__(self, *args, **kwargs):
        super(OpTrainRandomForestBlocked, self).__init__(*args, **kwargs)
        self.progressSignal = OrderedSignal()

    def setupOutputs(self):
        if self.inputs["fixClassifier"].value == False:
            self.outputs["Classifier"]._dtype = object
            self.outputs["Classifier"]._shape = (1,)
            self.outputs["Classifier"]._axistags  = "classifier"

            # No need to set dirty here: notifyDirty handles it.
            #self.outputs["Classifier"].setDirty((slice(0,1,None),))

    @traceLogged(logger, level=logging.INFO, msg="OpTrainRandomForestBlocked: Training Classifier")
    def execute(self, slot, roi, result):
        progress = 0
        self.progressSignal(progress)
        numImages = len(self.Images)

        key = roi.toSlice()
        featMatrix=[]
        labelsMatrix=[]
        for i,labels in enumerate(self.inputs["Labels"]):
            if labels.shape is not None:
                #labels=labels[:].allocate().wait()
                blocks = self.inputs["nonzeroLabelBlocks"][i][0].allocate().wait()

                progress += 10/numImages
                self.progressSignal(progress)

                reqlistlabels = []
                reqlistfeat = []
                traceLogger.debug("Sending requests for {} non-zero blocks (labels and data)".format( len(blocks[0])) )
                for b in blocks[0]:

                    request = labels[b].allocate()
                    featurekey = list(b)
                    featurekey[-1] = slice(None, None, None)
                    request2 = self.inputs["Images"][i][featurekey].allocate()

                    reqlistlabels.append(request)
                    reqlistfeat.append(request2)

                traceLogger.debug("Requests prepared")

                numLabelBlocks = len(reqlistlabels)
                progress_outer = [progress] # Store in list for closure access
                if numLabelBlocks > 0:
                    progressInc = (80-10)/numLabelBlocks/numImages

                def progressNotify(req):
                    # Note: If we wanted perfect progress reporting, we could use lock here 
                    #       to protect the progress from being incremented simultaneously.
                    #       But that would slow things down and imperfect reporting is okay for our purposes.
                    progress_outer[0] += progressInc/2
                    self.progressSignal(progress_outer[0])

                for ir, req in enumerate(reqlistfeat):
                    image = req.notify(progressNotify)

                for ir, req in enumerate(reqlistlabels):
                    labblock = req.notify(progressNotify)

                traceLogger.debug("Requests fired")

                for ir, req in enumerate(reqlistlabels):
                    traceLogger.debug("Waiting for a label block...")
                    labblock = req.wait()

                    traceLogger.debug("Waiting for an image block...")
                    image = reqlistfeat[ir].wait()

                    indexes=numpy.nonzero(labblock[...,0].view(numpy.ndarray))
                    features=image[indexes]
                    labbla=labblock[indexes]

                    featMatrix.append(features)
                    labelsMatrix.append(labbla)

                progress = progress_outer[0]

                traceLogger.debug("Requests processed")
        
        self.progressSignal(80/numImages)

        if len(featMatrix) == 0 or len(labelsMatrix) == 0:
            # If there was no actual data for the random forest to train with, we return None
            result[0] = None
        else:
            featMatrix=numpy.concatenate(featMatrix,axis=0)
            labelsMatrix=numpy.concatenate(labelsMatrix,axis=0)

            RF=vigra.learning.RandomForest(100)
            try:
                logger.debug("Learning with Vigra...")
                RF.learnRF(featMatrix.astype(numpy.float32),labelsMatrix.astype(numpy.uint32))
                logger.debug("Vigra finished")
            except:
                logger.error( "ERROR: could not learn classifier" )
                logger.error( "featMatrix shape={}, max={}, dtype={}".format(featMatrix.shape, featMatrix.max(), featMatrix.dtype) )
                logger.error( "labelsMatrix shape={}, max={}, dtype={}".format(labelsMatrix.shape, labelsMatrix.max(), labelsMatrix.dtype ) )
                raise
            finally:
                self.progressSignal(100)
            assert RF is not None, "RF = %r" % RF
            result[0]=RF
        
        return result

    def setInSlot(self, slot, key, value):
        if self.inputs["fixClassifier"].value == False:
            self.outputs["Classifier"].setDirty((slice(0,1,None),))

    def setSubInSlot(self,slots,indexes, key,value):
        if self.inputs["fixClassifier"].value == False:
            self.outputs["Classifier"].setDirty((slice(0,1,None),))

    def notifySubSlotDirty(self, slots, indexes, key):
        if self.inputs["fixClassifier"].value == False:
            self.outputs["Classifier"].setDirty((slice(0,1,None),))

    def notifyDirty(self, slot, key):
        if slot is not self.fixClassifier and self.inputs["fixClassifier"].value == False:
            self.outputs["Classifier"].setDirty((slice(0,1,None),))


class OpPredictRandomForest(Operator):
    name = "PredictRandomForest"
    description = "Predict on multiple images"
    category = "Learning"

    inputSlots = [InputSlot("Image"),InputSlot("Classifier"),InputSlot("LabelsCount",stype='integer')]
    outputSlots = [OutputSlot("PMaps")]

    def setupOutputs(self):
        nlabels=self.inputs["LabelsCount"].value
        self.PMaps.meta.dtype = numpy.float32
        self.PMaps.meta.axistags = copy.copy(self.Image.meta.axistags)
        self.PMaps.meta.shape = self.Image.meta.shape[:-1]+(nlabels,) # FIXME: This assumes that channel is the last axis
        self.PMaps.meta.drange = (0.0, 1.0)

    def execute(self,slot, roi, result):
        key = roi.toSlice()
        nlabels=self.inputs["LabelsCount"].value

        traceLogger.debug("OpPredictRandomForest: Requesting classifier. roi={}".format(roi))
        RF=self.inputs["Classifier"].value
        if RF is None:
            # Training operator may return 'None' if there was no data to train with
            result[...] = numpy.zeros(numpy.subtract(roi.stop, roi.start), dtype=numpy.float32)[...]
            return
        traceLogger.debug("OpPredictRandomForest: Got classifier")        
        #assert RF.labelCount() == nlabels, "ERROR: OpPredictRandomForest, labelCount differs from true labelCount! %r vs. %r" % (RF.labelCount(), nlabels)

        newKey = key[:-1]
        newKey += (slice(0,self.inputs["Image"].shape[-1],None),)

        res = self.inputs["Image"][newKey].allocate().wait()

        shape=res.shape
        prod = numpy.prod(shape[:-1])
        features=res.reshape(prod, shape[-1])

        prediction=RF.predictProbabilities(features.astype(numpy.float32))
        prediction = prediction.reshape(*(shape[:-1] + (RF.labelCount(),)))

        # If our LabelsCount is higher than the number of labels in the training set,
        # then our results aren't really valid.
        # Duplicate the last label's predictions
        chanslice = slice(min(key[-1].start, RF.labelCount()-1), min(key[-1].stop, RF.labelCount()))
        result[...]=prediction[...,chanslice] # FIXME: This assumes that channel is the last axis



    def notifyDirty(self, slot, key):
        if slot == self.inputs["Classifier"]:
            logger.debug("OpPredictRandomForest: Classifier changed, setting dirty")
            if self.LabelsCount.ready() and self.LabelsCount.value > 0:
                self.outputs["PMaps"].setDirty(slice(None,None,None))
        elif slot == self.inputs["Image"]:
            nlabels=self.inputs["LabelsCount"].value
            if nlabels > 0:
                self.outputs["PMaps"].setDirty(key[:-1] + (slice(0,nlabels,None),))
        elif slot == self.inputs["LabelsCount"]:
            # When the labels count changes, we must resize the output
            if self.configured():
                # FIXME: It's ugly that we call the 'private' _setupOutputs() function here,
                #  but the output shape needs to change when this input becomes dirty,
                #  and the output change needs to be propagated to the rest of the graph.
                self._setupOutputs()
            self.outputs["PMaps"].setDirty(slice(None,None,None))


class OpSegmentation(Operator):
    name = "OpSegmentation"
    description = "displaying highest probability class for each pixel"

    inputSlots = [InputSlot("Input")]
    outputSlots = [OutputSlot("Output")]

    def setupOutputs(self):

        inputSlot = self.inputs["Input"]

        self.outputs["Output"]._shape = inputSlot.shape[:-1]
        self.outputs["Output"]._dtype = inputSlot.dtype
        self.outputs["Output"]._axistags = inputSlot.axistags


    def getOutSlot(self, slot, key, result):

        shape = self.inputs["Input"].shape
        rstart, rstop = sliceToRoi(key, self.outputs["Output"]._shape)
        rstart.append(0)
        rstop.append(shape[-1])
        rkey = roiToSlice(rstart,rstop)
        img = self.inputs["Input"][rkey].allocate().wait()

        stop = img.size

        seg = []

        for i in range(0,stop,img.shape[-1]):
            curr_prob = -1
            highest_class = -1
            for c in range(img.shape[-1]):
                prob = img.ravel()[i+c]
                if prob > curr_prob:
                    curr_prob = prob
                    highest_class = c
            assert highest_class != -1, "OpSegmentation: Strange classes/probabilities"

            seg.append(highest_class)

        seg = numpy.array(seg)
        seg.resize(img.shape[:-1])

        result[:] = seg[:]



    def notifyDirty(self,slot,key):
        self.outputs["Output"].setDirty(key)

    @property
    def shape(self):
        return self.outputs["Output"]._shape

    @property
    def dtype(self):
        return self.outputs["Output"]._dtype


class OpAreas(Operator):
    name = "OpAreas"
    description = "counting pixel areas"

    inputSlots = [InputSlot("Input"), InputSlot("NumberOfChannels")]
    outputSlots = [OutputSlot("Areas")]

    def setupOutputs(self):

        self.outputs["Areas"]._shape = (self.inputs["NumberOfChannels"].value,)

    def getOutSlot(self, slot, key, result):

        img = self.inputs["Input"][:].allocate().wait()

        numC = self.inputs["NumberOfChannels"].value

        areas = []
        for i in range(numC):
            areas.append(0)

        for i in img.flat:
            areas[int(i)] +=1

        result[:] = numpy.array(areas)



    def notifyDirty(self,slot,key):
        self.outputs["Output"].setDirty(key)

    @property
    def shape(self):
        return self.outputs["Output"]._shape

    @property
    def dtype(self):
        return self.outputs["Output"]._dtype
