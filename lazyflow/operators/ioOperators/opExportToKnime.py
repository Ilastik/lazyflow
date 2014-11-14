#lazyflow
from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.stype import Opaque
from lazyflow.rtype import List
import logging

logger = logging.getLogger(__name__)

import h5py
import re
import numpy as np
from operator import itemgetter

class OpExportToKnime(Operator):
    
    name = "Knime Export"

    RawImage = InputSlot()
    LabelImage = InputSlot()
    
    ObjectFeatures = InputSlot(rtype=List, stype=Opaque)
    SelectedFeatures = InputSlot(rtype=List, stype=Opaque)
    IncludeRawImage = InputSlot(stype="bool")
    FileType = InputSlot(value="h5", stype="str")
    OutputFileName = InputSlot(value="test_export_for_now.h5", stype="str")
    
    WriteData = OutputSlot(stype='bool', rtype=List)
    
    def __init__(self, settings, bar, *args, **kwargs):
        super(OpExportToKnime, self).__init__(*args, **kwargs)
        self.settings = settings
        self.feature_table = None
        self.feature_selection = None
        self.bar = bar
        
    def setupOutputs(self):
        self.WriteData.meta.shape = (1,)
        self.WriteData.meta.dtype = object
    
    def propagateDirty(self, slot, subindex, roi): 
        pass

    def write_to_csv(self, table):
        pass

    def execute(self, slot, subindex, roi, result):
        assert slot == self.WriteData

        self.feature_table = self.ObjectFeatures.value
        self.feature_selection = self.SelectedFeatures.value
        self._filter_features()
        if self.settings["force unique ids"]:
            self._force_unique_ids()

        filename = str(self.OutputFileName.value)

        obj_num = self.feature_table.shape[0]
        max_id_len = len(str(obj_num))
        paths = np.zeros((obj_num,), dtype="q,a63,a63")
        paths.dtype.names = ("Object id", "Image path", "Labeling path")
        paths["Object id"] = self.feature_table["Object id"]

        compression = self.settings["compression"]

        multiframe = self.RawImage.meta["shape"][0] > 1
        with h5py.File(filename, "w") as fout:
            self._make_dset(fout, "tables/FeatureTable", self.feature_table, compression)

            if self.IncludeRawImage.value:
                self._make_dset(fout, "images/raw", self.RawImage([]).wait().squeeze(), compression)
                self.bar.update_step(33)
                self._make_dset(fout, "images/labeling", self.LabelImage([]).wait().squeeze(), compression)
                self.bar.update_step(66)

                for i in xrange(obj_num):
                    paths[i][1] = "images/raw"
                    paths[i][2] = "images/labeling"

            else:
                for i, slicing in enumerate(self._get_coords()):
                    oid = self.feature_table["Object id"][i]
                    folder_id = str(oid).zfill(max_id_len)
                    if multiframe:
                        time = self.feature_table["Time"][i]
                        path = "images/%s/%s/%%s" % (time, folder_id)
                    else:
                        path = "images/%s/%%s" % folder_id
                    paths[i][1] = path % "raw"
                    paths[i][2] = path % "labeling"
                    raw = self.RawImage[slicing].wait()
                    labeling = self.LabelImage[slicing].wait()
                    if self.settings["normalize"]:
                        normalize = np.vectorize(lambda p: 1 if p == oid else 0)
                        labeling = normalize(labeling)

                    self._make_dset(fout, path % "raw", raw.squeeze(), compression)
                    self._make_dset(fout, path % "labeling", labeling.squeeze(), compression)
                    self.bar.update_step(float(i)/obj_num)
            self._make_dset(fout, "tables/ImagePathTable", paths, compression)
        self.bar.update_step(100)
        self.bar.finish_step()
        result[0] = True

    def _force_unique_ids(self):
        obj_num = self.feature_table.shape[0]
        self.feature_table["Object id"] = range(obj_num)

    @staticmethod
    def _clean_channel_name(name):
        if not "_ch_" in name:
            return name
        return name[:-4] + chr(ord("x") + int(name[-1]))

    @staticmethod
    def _make_dset(fout, path, data, compression):
        try:
            fout.create_dataset(path, data.shape, data=data, **compression)
        except TypeError:
            fout.create_dataset(path, data.shape, data=data)

    def _filter_features(self):
        new_names = []
        clean_names = []
        get_cat = re.compile(r"(.+),.(.+)")
        no_chan = re.compile(r"(.+)_ch_(.)")
        names = self.feature_table.dtype.names
        name_count = len(names)
        for i, name in enumerate(names):
            match = get_cat.findall(name)
            if not match:
                if name not in clean_names:
                    new_names.append(name)
                    clean_names.append(name)
            else:
                if match[0][0] == "Default features":
                    if name not in clean_names:
                        new_names.append(name)
                        clean_names.append(self._clean_channel_name(match[0][1]))
                else:
                    channel_match = no_chan.findall(match[0][1])
                    if not channel_match:
                        if match[0][1] not in clean_names and match[0][1] in self.feature_selection:
                            new_names.append(name)
                            clean_names.append(match[0][1])
                    else:
                        if match[0][1] not in clean_names and channel_match[0][0] in self.feature_selection:
                            new_names.append(name)
                            clean_names.append(self._clean_channel_name(match[0][1]))
            self.bar.update_step(float(i)/name_count)

        self.feature_table = self.feature_table[new_names]
        self.feature_table.dtype.names = clean_names
        self.bar.finish_step()

    def _get_coords(self):
        margin = self.settings["margin"]
        dimensions = self.settings["dimensions"]
        assert margin >= 0, "Margin muss be greater than or equal to 0"
        time = self.feature_table["Time"].astype(np.int32)
        minx = self.feature_table["Coord<Minimum>_x"].astype(np.int32)
        maxx = self.feature_table["Coord<Maximum>_x"].astype(np.int32)
        miny = self.feature_table["Coord<Minimum>_y"].astype(np.int32)
        maxy = self.feature_table["Coord<Maximum>_y"].astype(np.int32)
        try:
            minz = self.feature_table["Coord<Minimum>_z"].astype(np.int32)
            maxz = self.feature_table["Coord<Maximum>_z"].astype(np.int32)
        except ValueError:
            minz = maxz = None
        for i in xrange(self.feature_table.shape[0]):
            yield [
                slice(time[i], time[i]+1),
                slice(max(0, minx[i] - margin),
                      min(maxx[i] + margin, dimensions[1])),
                slice(max(0, miny[i] - margin),
                      min(maxy[i] + margin, dimensions[2])),
            ] + ([] if minz is None else [
                slice(max(0, minz[i] - margin),
                      min(maxz[i] + margin, dimensions[3])),
            ]) + [
                slice(0, 1)
            ]
