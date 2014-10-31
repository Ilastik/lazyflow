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
    FileType = InputSlot(value="hdf5", stype="str")
    OutputFileName = InputSlot(value="test_export_for_now.h5", stype="str")
    
    WriteData = OutputSlot(stype='bool', rtype=List)
    
    def __init__(self, settings, *args, **kwargs):
        super(OpExportToKnime, self).__init__(*args, **kwargs)
        self.settings = settings
        
    def setupOutputs(self):
        self.WriteData.meta.shape = (1,)
        self.WriteData.meta.dtype = object
    
    def propagateDirty(self, slot, subindex, roi): 
        pass

    def write_to_csv(self, table):
        pass

    def execute(self, slot, subindex, roi, result):
        assert slot == self.WriteData

        table = self.ObjectFeatures.value
        selection = self.SelectedFeatures.value
        inc_raw = self.IncludeRawImage.value
        filename = str(self.OutputFileName.value)

        new_names = []
        clean_names = []
        get_cat = re.compile(r"(.+),.(.+)")
        no_chan = re.compile(r"(.+)_ch_.")
        for name in table.dtype.names:
            match = get_cat.findall(name)
            if not match:
                if name not in clean_names:
                    new_names.append(name)
                    clean_names.append(name)
            else:
                if match[0][0] == "Default features":
                    if name not in clean_names:
                        new_names.append(name)
                        clean_names.append(match[0][1])
                else:
                    cmatch = no_chan.findall(match[0][1])
                    if not cmatch:
                        if match[0][1] not in clean_names and match[0][1] in selection:
                            new_names.append(name)
                            clean_names.append(match[0][1])
                    else:
                        if match[0][1] not in clean_names and cmatch[0] in selection:
                            new_names.append(name)
                            clean_names.append(match[0][1])

        wanted_table = table[new_names]
        wanted_table.dtype.names = clean_names

        obj_num = wanted_table.shape[0]
        max_id_len = len(str(obj_num))
        paths = np.zeros((obj_num,), dtype="q,a10,a10")
        paths.dtype.names = ("Object id", "Image path", "Labeling path")
        paths["Object id"] = wanted_table["Object id"]
        with h5py.File(filename, "w") as fout:
            fout["tables/FeatureTable"] = wanted_table

            if inc_raw:
                fout["images/raw"] = self.RawImage([]).wait().squeeze()
                fout["images/labeling"] = self.RawImage([]).wait().squeeze()

                for i in xrange(obj_num):
                    paths[i][1] = "images/raw"
                    paths[i][2] = "images/labeling"

            else:
                coords = self._get_coords(wanted_table)
                for i in xrange(obj_num):
                    oid = wanted_table["Object id"][i]
                    folder = str(oid).zfill(max_id_len)
                    path = "images/%s/%%s" % folder
                    paths[i][1] = path % "raw"
                    paths[i][2] = path % "labeling"
                    slicing = coords.next()
                    raw = self.RawImage[slicing].wait()
                    labeling = self.LabelImage[slicing].wait()
                    if self.settings["normalize"]:
                        normalize = np.vectorize(lambda p: 1 if p == oid else 0)
                        labeling = normalize(labeling)

                    fout[path % "raw"] = raw.squeeze()
                    fout[path % "labeling"] = labeling.squeeze()
            fout["tables/ImagePathTable"] = paths

        result[0] = True

    def _get_coords(self, table):
        margin = self.settings["margin"]
        dimensions = self.settings["dimensions"]
        assert margin >= 0, "Margin muss be greater than or equal to 0"
        time = table["Time"].astype(np.int32)
        minx = table["Coord<Minimum>_ch_0"].astype(np.int32)
        maxx = table["Coord<Maximum>_ch_0"].astype(np.int32)
        miny = table["Coord<Minimum>_ch_1"].astype(np.int32)
        maxy = table["Coord<Maximum>_ch_1"].astype(np.int32)
        try:
            minz = table["Coord<Minimum>_ch_2"].astype(np.int32)
            maxz = table["Coord<Maximum>_ch_2"].astype(np.int32)
        except ValueError:
            minz = maxz = None
        for i in xrange(table.shape[0]):
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

