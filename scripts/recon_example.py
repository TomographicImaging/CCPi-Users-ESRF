"""
conda create -n cil-esrf cil=22.0.0 tigre h5py hdf5-external-filter-plugins  ipywidgets -c conda-forge -c intel -c ccpi

"""

#%% imports
from cil.framework import ImageGeometry
from cil.utilities.jupyter import islicer
from cil.utilities.display import show2D
from cil.processors import CentreOfRotationCorrector, TransmissionAbsorptionConverter
from cil.recon import FBP
from cil.io import NEXUSDataWriter
import numpy as np
import os

import matplotlib.pyplot as plt

from ESRF_ID15_DataReader import ESRF_ID15_DataReader
from weight_duplicate_angles import weight_duplicate_angles

#%%
# set path to master hdf5 file and create a reader

filename = '/data/ESRF/test_data/PC811_1000cycles_absct_final_0001.h5'
reader = ESRF_ID15_DataReader(filename)

#%%
# print the geometry of either scan 1, 2, or the fill dataset
print(reader.get_geometry(dataset_id=1))
print(reader.get_geometry(dataset_id=2))
print(reader.get_geometry())

#%%
#set a single slice of interest
reader.set_roi(vertical=400)
print(reader.get_geometry())

#%% 
# read in the data for the slice from either dataset or full
data_centre_slice_1 = reader.read(1)
data_centre_slice_2 = reader.read(2)
data = reader.read()
show2D(data_centre_slice_1)
show2D(data_centre_slice_2)
show2D(data)


#%% 
# Take the -log to convert the data to absorption
data = TransmissionAbsorptionConverter()(data)


# %%
# centre the dataset
processor = CentreOfRotationCorrector.xcorrelation()
processor.set_input(data)
data = processor.get_output()

#todo - add ring removal

#%% reconstruct slice
reco = FBP(data).run()
show2D(reco)

#%% reconstruct slice by configuring reconstruction window
ig_slice = ImageGeometry(1400, 1300, center_x=0, center_y=50)
reco = FBP(data, ig_slice).run()
show2D(reco)

#%% see the problem with angles
angles_full = np.mod(data.geometry.angles,360)
plt.plot(angles_full)

#%% weight duplicate angles
processor = weight_duplicate_angles()
processor.set_input(data)
data = processor.get_output()

#%% reconstruct new slice
ig_slice = ImageGeometry(1400, 1300, center_x=0, center_y=50)
reco2 = FBP(data, ig_slice).run()
show2D(reco2)

#%%
show2D([reco, reco2], origin='upper-left')

#%%
# read in the full dataset by resetting the slice
reader.set_roi(None)
data = reader.read()

#%% 
# Take the -log to convert the data to absorption
data = TransmissionAbsorptionConverter()(data)

#%% weight duplicate angles
processor = weight_duplicate_angles()
processor.set_input(data)
data = processor.get_output()

#%%
# view the data interactively
islicer(data, origin='upper-left')

# %%
# centre the dataset
processor = CentreOfRotationCorrector.xcorrelation(slice_index=400)
processor.set_input(data)
data = processor.get_output()

#todo - add ring removal

#%% 
# reconstruct splitting the data in chunks as we go
ig = ImageGeometry(voxel_num_x = 1500, voxel_num_y = 1500, voxel_num_z = 500)
fbp = FBP(data, ig)

# if you don;t have enough RAM to do it in one then set the slices per chunk
#fbp.set_split_processing(64)

reco = fbp.run()
#%%
# show the volume
islicer(reco)

#%%
# save array as binary
path_out = os.path.abspath('/data/ESRF/test_data/vol_{0}_{1}_{2}'.format(*reco.shape))
#reco.as_array().astype(np.float32).tofile(path_out + '.raw')

#%%
# save array as a CIL Nexus file (hdf5)
path_out = os.path.abspath('/data/ESRF/test_data/vol.nxs')
writer = NEXUSDataWriter(data=reco, file_name=path_out, compression=0)
#writer.write()
#%%