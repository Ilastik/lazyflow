#lazyflow
from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.stype import Opaque
from lazyflow.rtype import List
import logging
logger = logging.getLogger(__name__)

import h5py
import re


class OpExportToKnime(Operator):
    
    name = "Knime Export"
    
    RawImage = InputSlot()
    #CCImage = InputSlot()
    
    ObjectFeatures = InputSlot(rtype=List, stype=Opaque)
    SelectedFeatures = InputSlot(rtype=List, stype=Opaque)
    IncludeRawImage = InputSlot(stype="bool")
    FileType = InputSlot(value="hdf5", stype="str")
    OutputFileName = InputSlot(value="test_export_for_now.h5", stype="str")
    
    WriteData = OutputSlot(stype='bool', rtype=List)
    
    def __init__(self, *args, **kwargs):
        super(OpExportToKnime, self).__init__(*args, **kwargs)
        
    def setupOutputs(self):
        self.WriteData.meta.shape = (1,)
        self.WriteData.meta.dtype = object
    
    def propagateDirty(self, slot, subindex, roi): 
        pass
    
    def write_to_hdf5(self, filename, table):
        #with h5py.File(filename, "w") as fout:
        import pprint
        print "\n\n\nTABLE\n\n"
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(table)
        
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

        with h5py.File(filename, "w") as fout:
            fout["tables/FeatureTable"] = wanted_table
            fout["tables/ImagePathTable"] = "NOTHING"
            max_id_len = len(str(len(wanted_table["Object id"])-1))
            for obj in wanted_table["Object id"]:
                folder = str(obj).zfill(max_id_len)
                fout["images/%s/labels" % folder] = "NOTHING (LABELS)"
                if not inc_raw:
                    fout["images/%s/raw" % folder] = "NOTHING (RAW)"
            if inc_raw:
                fout["images/raw"] = "NOTHING (RAW)"


        result[0] = True