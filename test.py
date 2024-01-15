#%%
from cil.framework import ImageGeometry
from cil.utilities.jupyter import islicer
from cil.utilities.display import show2D, show_geometry
from cil.processors import CentreOfRotationCorrector, TransmissionAbsorptionConverter, RingRemover
from cil.recon import FBP
from cil.io import NEXUSDataWriter
import numpy as np
import os
import hdf5plugin

from scripts.ESRF_ID15_DataReader import ESRF_ID15_DataReader
from scripts.WeightDuplicateAngles import WeightDuplicateAngles

import h5py




filename = '/data/ESRF/test_data/PC811_1000cycles_absct_final_0001.h5'#
reader = ESRF_ID15_DataReader(filename)
geometry = reader.get_geometry()
slice_index = 507
reader.set_roi(vertical=slice_index)
data_raw = reader.read(normalise=False)
show2D(data_raw)

#%%
dset_path = '1.1/measurement/hrrz_center'
dtype=np.float32
with h5py.File(filename, 'r') as f:
    dset = f.get(dset_path)
    source_sel = tuple([slice(None)]*dset.ndim)
    arr = dset.astype(dtype)[source_sel]
print(arr)
# %%


#%%


filename = '/data/ESRF/Wedgescan_Iterative_ASSB/InSitu-LPSCL-20Ton-30Min_0001.h5'#
projection_filename = '/data/ESRF/Wedgescan_Iterative_ASSB/projections.h5'#
dark_filename = '/data/ESRF/Wedgescan_Iterative_ASSB/darks.h5'#
flat_filename = '/data/ESRF/Wedgescan_Iterative_ASSB/flats.h5'#
reader = ESRF_ID15_DataReader(filename, projection_filename=projection_filename, dark_filename=dark_filename, flat_filename=flat_filename)
geometry = reader.get_geometry(dataset_id=2)
slice_index = 507
reader.set_roi(vertical=slice_index)
data_raw = reader.read(dataset_id=2, normalise=False)
show2D(data_raw)

#%%
filename = '/data/ESRF/Wedgescan_Iterative_ASSB/InSitu-LPSCL-20Ton-30Min_0001.h5'#
dset_path = '4.1/instrument/hrsrot_center'
dtype=np.float32
with h5py.File(filename, 'r') as f:
    dset = f.get(dset_path)
    source_sel = tuple([slice(None)]*dset.ndim)
    arr = dset.astype(dtype)[source_sel]
print(arr)

# %%
filename = '/data/ESRF/Wedgescan_Iterative_ASSB/projections.h5'#
f = h5py.File(filename, 'r')
group_keys = list(f.keys())
for group_key in group_keys:
    group = f[group_key]
    data_keys = list(group)
    for data_key in data_keys:
        print(str(group[data_key]))
        data = group[data_key]
        try:
            sub_data_keys = list(data)
            for sub_data_key in sub_data_keys:
                print('\t' + str(data[sub_data_key]))
        except:
            print('\t' + str(data))

# %%
filename = '/data/ESRF/Wedgescan_Iterative_ASSB/InSitu-LPSCL-20Ton-30Min_0001.h5'#
reader = ESRF_ID15_DataReader(filename)
f = h5py.File(filename, 'r')
group_keys = list(f.keys())
for group_key in group_keys:
    group = f[group_key]
    data_keys = list(group)
    for data_key in data_keys:
        print(str(group[data_key]))
        data = group[data_key]
        try:
            sub_data_keys = list(data)
            for sub_data_key in sub_data_keys:
                print('\t' + str(data[sub_data_key]))
        except:
            print('\t' + str(data))



# %%
filename = '/data/ESRF/test_data/PC811_1000cycles_absct_final_0001.h5'#
reader = ESRF_ID15_DataReader(filename)
f = h5py.File(filename, 'r')
group_keys = list(f.keys())
for group_key in group_keys:
    group = f[group_key]
    data_keys = list(group)
    for data_key in data_keys:
        print(str(group[data_key]))
        data = group[data_key]
        try:
            sub_data_keys = list(data)
            for sub_data_key in sub_data_keys:
                print('\t' + str(data[sub_data_key]))
        except:
            print('\t' + str(data))
# %%
filename = '/data/ESRF/Wedgescan_Iterative_ASSB/flats.h5'#
f = h5py.File(filename, 'r')
group_keys = list(f.keys())
for group_key in group_keys:
    group = f[group_key]
    data_keys = list(group)
    for data_key in data_keys:
        print(str(group[data_key]))
        data = group[data_key]
        try:
            sub_data_keys = list(data)
            for sub_data_key in sub_data_keys:
                print('\t' + str(data[sub_data_key]))
        except:
            print('\t' + str(data))


# %%
projection_filename = '/data/ESRF/Wedgescan_Iterative_ASSB/projections.h5'#
source_sel = None
dset_path = 'entry_0000/measurement/data/'
dtype=np.float32
with h5py.File(projection_filename, 'r') as f:
    dset = f.get(dset_path)
    print(dset)
    if source_sel == None:
        source_sel = tuple([slice(None)]*dset.ndim)

    arr = dset.astype(dtype)[source_sel]


# %%
projection_filename = '/data/ESRF/Wedgescan_Iterative_ASSB/projections.h5'#
dset_path = 'entry_0000/instrument/pco2linux/data'
with h5py.File(projection_filename, 'r') as f:
    dset = f.get(dset_path)
    A = [dset.ndim, dset.shape, dset.size, dset.dtype, dset.nbytes, dset.compression, dset.chunks]
shape = A[1]
print(shape[0])
# %%
