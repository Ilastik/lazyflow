import vigra
from PyQt4.QtCore import pyqtSignal, QObject
from PyQt4.QtGui import QApplication
from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.stype import Opaque
from lazyflow.rtype import List
import logging

logger = logging.getLogger(__name__)

import h5py
import numpy as np

from itertools import izip


class Emitter(QObject):
    UpdateProgress = pyqtSignal(int)
    FinishStep = pyqtSignal()
    Busy = pyqtSignal(bool)

    def __init__(self, bar):
        super(Emitter, self).__init__()
        self.UpdateProgress.connect(bar.update_step)
        self.FinishStep.connect(bar.finish_step)
        self.Busy.connect(bar.set_busy)

    def update(self, i):
        self.UpdateProgress.emit(i)

    def finish(self):
        self.FinishStep.emit()

    def busy(self, state):
        # noinspection PyCallByClass
        self.Busy.emit(state)
        # noinspection PyArgumentList
        QApplication.processEvents()


class OpExportObjectInfo(Operator):

    name = "Knime Export"

    RawImage = InputSlot()
    LabelImage = InputSlot()
    
    ObjectFeatures = InputSlot(rtype=List, stype=Opaque)
    SelectedFeatures = InputSlot(rtype=List, stype=Opaque)
    
    WriteData = OutputSlot(stype='bool', rtype=List)

    """
    This operator exports the feature table and the raw and labeling images to a file

    :param settings: the settings from the exportObjectInfoDialog
    :type settings: dict
    :param bar: the progressbar dialog
    :type bar: MultiProgressDialog
    """
    def __init__(self, settings, bar, *args, **kwargs):
        super(OpExportObjectInfo, self).__init__(*args, **kwargs)
        self.settings = settings
        self.feature_table = None
        self.feature_selection = None
        self.emitter = Emitter(bar)

    def _create_feature_table(self):
        frames = self.ObjectFeatures.meta.shape[0]

        if frames > 1:
            computed_features = {}
            for frame in xrange(frames - 1):
                request = self.ObjectFeatures([frame, frame+1])
                computed_features.update(request.wait())
                self.emitter.update(100 * frame / frames)
        else:
            computed_features = self.ObjectFeatures([]).wait()

        selection = self.SelectedFeatures([]).wait()

        logger.debug("Features ready")
        self.emitter.update(50)
        feature_names = []
        feature_cats = []
        feature_channels = []
        feature_types = []
        for cat_name, category in computed_features[0].iteritems():
            for feat_name, feat_array in category.iteritems():
                if cat_name == "Default features" or \
                        feat_name not in feature_names and \
                        feat_name in selection:
                    feature_names.append(feat_name)
                    feature_cats.append(cat_name)
                    feature_channels.append(feat_array.shape[1])
                    feature_types.append(feat_array.dtype)

        obj_count = []
        for time in computed_features.iterkeys():
            obj_count.append(computed_features[time]["Default features"]["Count"].shape[0])

        channel_name = ["x", "y", "z"]
        dtype_names = ["image id", "object id", "time"]
        dtype_types = ["I", "I", "I"]
        dtype_to_key = {}
        for i, name in enumerate(feature_names):
            if feature_channels[i] > 1:
                for j in xrange(feature_channels[i]):
                    dtype_names.append("%s_%s" % (name, channel_name[j]))
                    dtype_types.append(feature_types[i].name)
                    dtype_to_key[dtype_names[-1]] = (feature_cats[i], name, j)
            else:
                dtype_names.append(name)
                dtype_types.append(feature_types[i].name)
                dtype_to_key[dtype_names[-1]] = (feature_cats[i], name, 0)
        dtype_names.extend(["raw", "labels"])
        dtype_types.extend(["a63", "a63"])

        self.feature_table = np.zeros((sum(obj_count),), dtype=",".join(dtype_types))
        self.feature_table.dtype.names = map(str, dtype_names)

        start = 0
        end = obj_count[0]
        for time in computed_features.iterkeys():
            for name in dtype_names[3:-2]:
                cat, feat_name, index = dtype_to_key[name]
                self.feature_table[name][start:end] = computed_features[time][cat][feat_name][:, index]
            self.feature_table["time"][start:end] = int(time)
            start = end
            try:
                end += obj_count[int(time) + 1]
            except IndexError:
                end = sum(obj_count)

        logger.debug("Feature table ready")

        self.emitter.finish()

    def setupOutputs(self):
        self.WriteData.meta.shape = (1,)
        self.WriteData.meta.dtype = object
    
    def propagateDirty(self, slot, subindex, roi): 
        pass

    def execute(self, slot, subindex, roi, result):
        assert slot == self.WriteData

        self._create_feature_table()

        if self.settings["force unique ids"]:
            self._force_unique_ids()

        filename = unicode(self.settings["file path"])

        if self.settings["file type"] == "h5":
            self._export_to_h5(result, filename)
        elif self.settings["file type"] == "csv":
            self._export_to_csv(result, filename)
        else:
            logger.warn("'%s' is not a valid file type!" % self.settings["file type"])

    def _export_to_csv(self, result, filename):
        with open(filename, "w") as fout:
            line = ",".join(self.feature_table.dtype.names)
            fout.write(line)
            fout.write("\n")

            for row in self.feature_table:
                line = ",".join(map(str, row))
                fout.write(line)
                fout.write("\n")
        self.emitter.finish()
        result[0] = True

    def _export_to_h5(self, result, filename):

        obj_num = self.feature_table.shape[0]
        max_id_len = len(str(obj_num))

        compression = self.settings["compression"]

        with h5py.File(filename, "w") as fout:

            raw_axistags = self.RawImage.meta.axistags
            labeling_axistags = self.LabelImage.meta.axistags
            if self.settings["include raw"]:
                meta = {
                    "type": "image",
                    "axistags": raw_axistags.toJSON()
                }
                raw = self.RawImage([]).wait()
                self._make_dset(fout, "images/raw", raw.squeeze(), compression, meta)
                self.feature_table["raw"] = ["raw"] * obj_num

                coords = self._get_coords(labeling_axistags)
            else:
                coords = izip(self._get_coords(raw_axistags), self._get_coords(labeling_axistags))

            for i, slicing in enumerate(coords):
                oid = self.feature_table["object id"][i]
                #folder_id = str(i).zfill(max_id_len)
                folder_id = str(i)
                path = "images/%s/%%s" % folder_id
                if not self.settings["include raw"]:
                    raw_slicing, labeling_slicing = slicing
                    self.feature_table[i]["raw"] = (path % "raw")[7:]
                    raw = self.RawImage[raw_slicing].wait()
                    actual_atags = [raw_axistags[j] for j, shape in enumerate(raw.shape) if shape > 1]
                    meta = {
                        "type": "image",
                        "axistags": vigra.AxisTags(actual_atags).toJSON(),
                    }
                    self._make_dset(fout, path % "raw", raw.squeeze(), compression, meta)
                else:
                    labeling_slicing = slicing
                self.feature_table[i]["labels"] = (path % "labels")[7:]
                labeling = self.LabelImage[labeling_slicing].wait()
                if self.settings["normalize"]:
                    id_ = i if self.settings["force unique ids"] else oid
                    normalize = np.vectorize(lambda p: 1 if p == id_ else 0)
                    labeling = normalize(labeling).view("uint64")
                    labeling.dtype = np.uint64
                actual_atags = [labeling_axistags[j] for j, shape in enumerate(labeling.shape) if shape > 1]
                meta = {
                    "type": "labeling",
                    "axistags": vigra.AxisTags(actual_atags).toJSON(),
                }

                self._make_dset(fout, path % "labels", labeling.squeeze(), compression, meta)
                self.emitter.update(100 * i / obj_num)

            self._make_dset(fout, "table", self.feature_table, compression)
        self.emitter.finish()
        result[0] = True

    def _force_unique_ids(self):
        obj_num = self.feature_table.shape[0]
        self.feature_table["object id"] = range(0, obj_num)
        self.feature_table["image id"] = range(0, obj_num)

    @staticmethod
    def _make_dset(fout, path, data, compression, meta=None):
        if meta is None:
            meta = {}
        try:
            dset = fout.create_dataset(path, data.shape, data=data, **compression)
        except TypeError:
            dset = fout.create_dataset(path, data.shape, data=data)
        for k, v in meta.iteritems():
            dset.attrs[k] = v
        return dset

    def _get_coords(self, axistags):
        margin = self.settings["margin"]
        dimensions = self.settings["dimensions"]
        assert margin >= 0, "Margin muss be greater than or equal to 0"
        time = self.feature_table["time"].astype(np.int32)
        minx = self.feature_table["Coord<Minimum>_x"].astype(np.int32)
        maxx = self.feature_table["Coord<Maximum>_x"].astype(np.int32)
        miny = self.feature_table["Coord<Minimum>_y"].astype(np.int32)
        maxy = self.feature_table["Coord<Maximum>_y"].astype(np.int32)
        table_shape = self.feature_table.shape[0]
        try:
            minz = self.feature_table["Coord<Minimum>_z"].astype(np.int32)
            maxz = self.feature_table["Coord<Maximum>_z"].astype(np.int32)
        except ValueError:
            minz = maxz = [0] * table_shape

        indices = map(axistags.index, "txyzc")
        excludes = indices.count(-1)
        for i in xrange(table_shape):

            slicing = [
                slice(time[i], time[i]+1),
                slice(max(0, minx[i] - margin),
                      min(maxx[i] + margin, dimensions[1])),
                slice(max(0, miny[i] - margin),
                      min(maxy[i] + margin, dimensions[2])),
                slice(max(0, minz[i] - margin),
                      min(maxz[i] + margin, dimensions[3])),
                slice(0, 1)
            ]
            yield map(slicing.__getitem__, indices)[:5-excludes]
