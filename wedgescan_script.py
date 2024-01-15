from cil.framework import ImageGeometry
from cil.utilities.jupyter import islicer
from cil.utilities.display import show2D, show1D, show_geometry
from cil.processors import CentreOfRotationCorrector, TransmissionAbsorptionConverter, RingRemover, Padder, Slicer
from cil.recon import FBP
from cil.io import NEXUSDataWriter
from cil.io.utilities import HDF5_utilities
import numpy as np
import hdf5plugin
from cil.framework import AcquisitionGeometry, AcquisitionData
import matplotlib.pyplot as plt
from tomopy.prep.phase import retrieve_phase
from cil.optimisation.functions import L2NormSquared, BlockFunction, L1Norm, ZeroFunction, MixedL21Norm, TotalVariation
from cil.optimisation.algorithms import PDHG
from cil.optimisation.operators import BlockOperator, FiniteDifferenceOperator, GradientOperator
from cil.plugins.astra.operators import ProjectionOperator

filename = '/data/ESRF/Wedgescan_Iterative_ASSB/InSitu-LPSCL-20Ton-30Min_0001.h5'
angles = HDF5_utilities.read(filename, '/4.1/measurement/hrsrot')
ds_metadata = HDF5_utilities.get_dataset_metadata(filename, '4.1/instrument/pco2linux/data')

roi = [slice(None), slice(450, 950), slice(None)]
source_sel=tuple(roi)

filename = '/data/ESRF/Wedgescan_Iterative_ASSB/flats.h5'
HDF5_utilities.print_metadata(filename, '/entry_0000/measurement', 2)
HDF5_utilities.get_dataset_metadata(filename, 'entry_0000/measurement/data')
flats = HDF5_utilities.read(filename, 'entry_0000/measurement/data', tuple(source_sel))
flat = np.mean(flats, axis = 0) #median?


filename = '/data/ESRF/Wedgescan_Iterative_ASSB/darks.h5'
HDF5_utilities.print_metadata(filename, 'entry_0000/measurement', 2)
HDF5_utilities.get_dataset_metadata(filename, 'entry_0000/measurement/data')
darks = HDF5_utilities.read(filename, 'entry_0000/measurement/data', tuple(source_sel))
dark = np.mean(darks, axis = 0) 

projections = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
data = np.empty((3000, roi[1].stop-roi[1].start, ds_metadata['shape'][2]), dtype=np.float32)
for i in projections:
    filename = '/data/ESRF/Wedgescan_Iterative_ASSB/pco2linux_{0:04}.h5'.format(i)
    print(filename)
    ds_arr = HDF5_utilities.read_to(filename, 'entry_0000/measurement/data',data,source_sel, np.s_[i*200:i*200+200,:,:])

flat = flat - dark
data = (data - dark)/ flat

ag = AcquisitionGeometry.create_Parallel3D().set_panel([np.shape(data)[2],np.shape(data)[1]]).set_angles(angles)
data = AcquisitionData(data, deep_copy=False, geometry = ag)

data = Slicer(roi={'angle': (300, 2710, 1)})(data)

processor = TransmissionAbsorptionConverter()
processor.set_input(data)
processor.get_output(out=data)
show2D(data)

# Normalise the gradient
gradient = 0
offset = 0
mean_intensity = 0
for i in np.arange(data.shape[0]):
    y = data.array[i,int(data.shape[1]/2),:]
    x = np.arange(len(y))
    coef = np.polyfit(x,y,1)
    gradient += coef[0]
    offset += coef[1]
    mean_intensity += np.mean(data.array[i,:,:])
    poly1d_fn = np.poly1d(coef)

gradient = gradient/data.shape[0]
offset = offset/data.shape[0]
mean_intensity = mean_intensity/data.shape[0]
poly1d_fn = np.poly1d([gradient, offset])

for i in np.arange(data.shape[0]):
    proj_index = i
    data.array[i,:,:] = data.array[i,:,:] / poly1d_fn(x)
data.array[:,:,:] = mean_intensity*data.array[:,:,:]/np.mean(data.array[:,:,:])

temp = retrieve_phase(data.array, alpha=0.01)
data = AcquisitionData(temp, deep_copy=False, geometry = data.geometry)

# take a single slice
vertical = 400
data_slice = data.get_slice(vertical=vertical)
show2D(data_slice)
ig = data_slice.geometry.get_ImageGeometry()

padsize = 3000
data_slice = Padder.linear_ramp( padsize, 0)(data_slice)

r = RingRemover(8,'db20', 1.5)
r.set_input(data_slice)
data_slice = r.get_output()

data_slice.geometry.set_centre_of_rotation(16.5, distance_units='pixels')

def algo_isotropic_TV(data_slice, alpha, initial=None):

    ag = data_slice.geometry
    ig = ag.get_ImageGeometry()

    F = BlockFunction(alpha * MixedL21Norm(),
        0.5*L2NormSquared(b=data_slice))

    K = BlockOperator(GradientOperator(ig), 
                      ProjectionOperator(ig, ag, device="gpu"))
    
    G = ZeroFunction()

    normK = K.norm()
    sigma = 0.5
    tau = 1./(sigma*normK**2)

    myPDHG = PDHG(f=F, 
                g=G, 
                operator=K, 
                sigma = sigma, tau = tau,
                initial = initial,
                max_iteration=2000, 
                update_objective_interval = 10,
                )
    
    return myPDHG

def algo_isotropic_TV_implicit(data_slice, alpha, sigma=0.5, initial=None):

    ag = data_slice.geometry
    ig = ag.get_ImageGeometry()

    F = 0.5 * L2NormSquared(b=data_slice)
    
    G = (alpha/ig.voxel_size_y) *TotalVariation(max_iteration=10)
    
    K = ProjectionOperator(ig, ag, device='gpu')

    normK = K.norm()
    tau = 1./(sigma*normK**2)

    myPDHG = PDHG(f=F, 
                g=G, 
                operator=K, 
                sigma = sigma, tau = tau,
                initial = initial,
                max_iteration=2000, 
                update_objective_interval = 10,
                )

    return myPDHG

def algo_anisotropic_TV(data_slice, alpha, alpha_dx, sigma=0.5, initial=None):

    ag = data_slice.geometry
    ig = ag.get_ImageGeometry()

    F = BlockFunction(0.5*L2NormSquared(b=data_slice),
                          alpha*MixedL21Norm(), 
                          alpha_dx*L1Norm())

    K = BlockOperator(ProjectionOperator(ig, ag), 
                         GradientOperator(ig), 
                         FiniteDifferenceOperator(ig, direction='horizontal_x'))

    G = ZeroFunction()

    normK = K.norm()
    tau = 1./(sigma*normK**2)

    myPDHG = PDHG(f=F, 
                g=G, 
                operator=K, 
                sigma = sigma, tau = tau,
                initial = initial,
                max_iteration=2000, 
                update_objective_interval = 10,
                )

    return myPDHG

# alphas = [0.01, 0.05, 0.1, 0.5, 1, 5]

# for i in np.arange(len(alphas)):
#     myPDHG = algo_isotropic_TV_implicit(data_slice, alpha=alphas[i], sigma=0.5, initial=None)
#     myPDHG.run(300,verbose=2)
#     reco = myPDHG.solution

#     reco = Slicer(roi={'horizontal_y': (padsize,padsize+2560), 'horizontal_x' : (padsize,padsize+2560)})(reco)

#     writer = NEXUSDataWriter()
#     writer.set_up(data=reco,
#             file_name='sigma05_reco_alpha_loop_'+str(i)+'.nxs')
#     writer.write()

#     np.save('sigma05_obj_alpha_loop_'+str(i)+'.npy', myPDHG.objective)


## 
alpha = 0.05
alpha_dx = 0.01
sigma = 0.1

myPDHG = algo_anisotropic_TV(data_slice, alpha=alpha, alpha_dx=alpha_dx, sigma=sigma, initial=None)
myPDHG.run(1000,verbose=2)
reco = myPDHG.solution

reco = Slicer(roi={'horizontal_y': (padsize,padsize+2560), 'horizontal_x' : (padsize,padsize+2560)})(reco)

writer = NEXUSDataWriter()
writer.set_up(data=reco,
        file_name='aniso_reco_alpha005_alphadx001_sigma01.nxs')
writer.write()

np.save('aniso_obj_alpha005_alphadx001_sigma01.npy', myPDHG.objective)

#### long iso TV
alpha = 0.1
sigma = 0.1

myPDHG = algo_isotropic_TV_implicit(data_slice, alpha=alpha, sigma=sigma, initial=None)
myPDHG.run(1000,verbose=2)
reco = myPDHG.solution

reco = Slicer(roi={'horizontal_y': (padsize,padsize+2560), 'horizontal_x' : (padsize,padsize+2560)})(reco)

writer = NEXUSDataWriter()
writer.set_up(data=reco,
        file_name='long_sigma01_reco_alpha01.nxs')
writer.write()

np.save('long_sigma01_obj_alpha01.npy', myPDHG.objective)