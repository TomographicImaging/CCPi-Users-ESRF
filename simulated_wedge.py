#%%
from cil.io import NEXUSDataReader
from cil.utilities.display import show2D
from cil.utilities.jupyter import islicer
from cil.framework import AcquisitionData
from matplotlib import pyplot as plt
#%%
reader = NEXUSDataReader()
reader.set_up(file_name='/data/simulations/parallel/tigre_interpolated/2048/sim.nxs')
# load data
ad1 = reader.read()
# get AcquisitionGeometry
ag1 = reader.get_geometry()
# islicer(ad1)
# %%
idx = (ag1.angles > 20) & (ag1.angles < 160)

data_array = ad1.array[idx,:]
# %%
ag2 = ag1.copy()
ag2.set_angles(ag2.angles[idx])
ad2 = AcquisitionData(data_array, geometry=ag2, deep_copy=False)
print(ad2.geometry)
# islicer(ad2)
# %%
from cil.processors import CentreOfRotationCorrector
import logging
# logging.basicConfig(level=logging.WARNING)
# cil_log_level = logging.getLogger('cil.processors')
# cil_log_level.setLevel(logging.DEBUG)
#%%
# processor = CentreOfRotationCorrector.image_sharpness('centre', 'tigre')
# processor.set_input(ad1) 
# processor.get_output(out=ad1)

# %%
# processor = CentreOfRotationCorrector.image_sharpness('centre', 'tigre', 0.01, 40,2)
# processor.set_input(ad2) 
# processor.get_output(out=ad2)
# %%
# processor = CentreOfRotationCorrector.xcorrelation(slice_index = 'centre', projection_index = 20.0, ang_tol=45)
# processor.set_input(ad2) 
# processor.get_output(out=ad2)
# %%
idx = (ag1.angles > 0.4) & (ag1.angles < 179.6)

data_array = ad1.array[idx,:]
# %%
ag3 = ag1.copy()
ag3.set_angles(ag3.angles[idx])
ad3 = AcquisitionData(data_array, geometry=ag3, deep_copy=False)
#%%
print(ad3.geometry.angles)
#%%
processor = CentreOfRotationCorrector.xcorrelation(slice_index = 'centre', projection_index = 0, ang_tol=2)
processor.set_input(ad3) 
processor.get_output(out=ad3)
# %%
