#%%
from cil.framework import AcquisitionGeometry, AcquisitionData
from cil.utilities.display import show2D, show_geometry
from cil.processors import TransmissionAbsorptionConverter, Padder
from cil.io.utilities import HDF5_utilities
from cil.utilities.jupyter import islicer
import numpy as np
import matplotlib.pyplot as plt
#%%
filename = '/data/ESRF/Wedgescan_Iterative_ASSB/InSitu-LPSCL-20Ton-30Min_0001.h5'#
HDF5_utilities.print_metadata(filename, '/4.1/instrument/pco2linux/data', 2)
HDF5_utilities.get_dataset_metadata(filename, '/4.1/instrument/pco2linux/data')
array = HDF5_utilities.read(filename, '/4.1/instrument/pco2linux/data')
print(array)

#%%

# %%

# %%
filename = '/data/ESRF/Wedgescan_Iterative_ASSB/InSitu-LPSCL-20Ton-30Min_0001.h5'#
angles = HDF5_utilities.read(filename, '/4.1/measurement/hrsrot_center')

for i in np.arange(0,15):
    print('from ' + str(i*200) + ' to ' + str(i*200+200))
    angles[i*200:1*200+200]


# %% 
roi = [slice(None), slice(320, 1050), slice(None)]
source_sel=tuple(roi)
# %%

filename = '/data/ESRF/Wedgescan_Iterative_ASSB/flats.h5'#
HDF5_utilities.print_metadata(filename, '/entry_0000/measurement', 2)
HDF5_utilities.get_dataset_metadata(filename, 'entry_0000/measurement/data')
flats = HDF5_utilities.read(filename, 'entry_0000/measurement/data', source_sel=tuple(roi))
flat = np.mean(flats, axis = 0)


filename = '/data/ESRF/Wedgescan_Iterative_ASSB/darks.h5'#
HDF5_utilities.print_metadata(filename, 'entry_0000/measurement', 2)
HDF5_utilities.get_dataset_metadata(filename, 'entry_0000/measurement/data')
darks = HDF5_utilities.read(filename, 'entry_0000/measurement/data', source_sel=tuple(roi))
dark = np.mean(darks, axis = 0)

# %%
i = 10
filename = '/data/ESRF/Wedgescan_Iterative_ASSB/pco2linux_{0:04}.h5'.format(i)
print(filename)
ds_metadata = HDF5_utilities.get_dataset_metadata(filename, 'entry_0000/measurement/data')
ds_arr = HDF5_utilities.read(filename, 'entry_0000/measurement/data', source_sel).squeeze()
# ag = AcquisitionGeometry.create_Parallel3D().set_panel([ds_metadata['shape'][2],ds_metadata['shape'][1]]).set_angles(angles[i*200:i*200+200])
# show_geometry(ag)
# data = AcquisitionData(ds_arr, geometry=ag.get_slice(vertical = slice_index), deep_copy=False)
data = ds_arr
flat = flat - dark
data_norm = (data - dark)/ flat
# data = TransmissionAbsorptionConverter(min_intensity=-0.00011)(data)
# %%
show2D([dark, flat, data, data_norm])
islicer(data_norm)

# %%
plt.plot(angles)
## 3.1 or 5.1
# %%

## 2.1
# %%


## 4.1 -> full dataset?
## 4.2 -> limited set but 647