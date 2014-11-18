#lazyflow
from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.stype import Opaque
from lazyflow.rtype import List
import logging

logger = logging.getLogger(__name__)

import h5py
import re
import numpy as np


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

    def _create_feature_table(self):
        computed_features = self.ObjectFeatures([]).wait()
        selection = self.SelectedFeatures([]).wait()
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
        dtype_names = ["Object id", "Time"]
        dtype_types = ["q", "q"]
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

        self.feature_table = np.zeros((sum(obj_count),), dtype=",".join(dtype_types))
        self.feature_table.dtype.names = map(str, dtype_names)

        start = 0
        end = obj_count[0]
        for time in computed_features.iterkeys():
            for name in dtype_names[2:]:
                cat, feat_name, index = dtype_to_key[name]
                self.feature_table[name][start:end] = computed_features[time][cat][feat_name][:, index]
            self.feature_table["Time"][start:end] = int(time)
            start = end
            try:
                end += obj_count[int(time) + 1]
            except IndexError:
                end = -1

        logger.info("Object feature table created")
        self.bar.finish_step()


    def setupOutputs(self):
        self.WriteData.meta.shape = (1,)
        self.WriteData.meta.dtype = object
    
    def propagateDirty(self, slot, subindex, roi): 
        pass

    def write_to_csv(self, table):
        pass

    def execute(self, slot, subindex, roi, result):
        assert slot == self.WriteData

        self._create_feature_table()

        if self.settings["force unique ids"]:
            self._force_unique_ids()

        filename = str(self.OutputFileName.value)

        obj_num = self.feature_table.shape[0]
        max_id_len = len(str(obj_num))
        paths = np.zeros((obj_num,), dtype="q,a63,a63")
        paths.dtype.names = ("Object id", "Image path", "Labeling path")
        paths["Object id"] = self.feature_table["Object id"]

        compression = self.settings["compression"]

        with h5py.File(filename, "w") as fout:
            self._make_dset(fout, "tables/FeatureTable", self.feature_table, compression)

            if self.IncludeRawImage.value:
                self._make_dset(fout, "images/raw", self.RawImage([]).wait().squeeze(), compression)
                self.bar.update_step(50)
                for i in xrange(obj_num):
                    paths[i][1] = "images/raw"

            for i, slicing in enumerate(self._get_coords()):
                oid = self.feature_table["Object id"][i]
                folder_id = str(i).zfill(max_id_len)
                path = "images/%s/%%s" % folder_id
                if not self.IncludeRawImage.value:
                    paths[i][1] = path % "raw"
                    raw = self.RawImage[slicing].wait()
                    self._make_dset(fout, path % "raw", raw.squeeze(), compression)

                paths[i][2] = path % "labeling"
                labeling = self.LabelImage[slicing].wait()
                if self.settings["normalize"]:
                    id_ = i if self.settings["force unique ids"] else oid
                    normalize = np.vectorize(lambda p: 1 if p == id_ else 0)
                    labeling = normalize(labeling)
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
