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
from cil.optimisation.functions import L2NormSquared, BlockFunction, L1Norm, ZeroFunction, MixedL21Norm
from cil.optimisation.algorithms import PDHG
from cil.optimisation.operators import BlockOperator, FiniteDifferenceOperator, GradientOperator
from cil.plugins.astra.operators import ProjectionOperator

filename = '/data/ESRF/Wedgescan_Iterative_ASSB/InSitu-LPSCL-20Ton-30Min_0001.h5'#
angles = HDF5_utilities.read(filename, '/4.1/measurement/hrsrot')
ds_metadata = HDF5_utilities.get_dataset_metadata(filename, '4.1/instrument/pco2linux/data')

roi = [slice(None), slice(450, 950), slice(None)]
source_sel=tuple(roi)

filename = '/data/ESRF/Wedgescan_Iterative_ASSB/flats.h5'#
HDF5_utilities.print_metadata(filename, '/entry_0000/measurement', 2)
HDF5_utilities.get_dataset_metadata(filename, 'entry_0000/measurement/data')
flats = HDF5_utilities.read(filename, 'entry_0000/measurement/data', tuple(source_sel))
flat = np.mean(flats, axis = 0) #median?


filename = '/data/ESRF/Wedgescan_Iterative_ASSB/darks.h5'#
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
    # ds_arr[ds_arr>1000] = np.mean(ds_arr) ### this isn't right - how should I be doing this?
    #data[i*200:i*200+200] = ds_arr
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

# take a single slice
vertical = 400
data_slice = data.get_slice(vertical=vertical)
show2D(data_slice)
ig = data_slice.geometry.get_ImageGeometry()

padsize = 50
data_slice = Padder.linear_ramp( padsize, 0)(data_slice)

r = RingRemover(8,'db20', 1.5)
r.set_input(data_slice)
data_slice = r.get_output()

data_slice.geometry.set_centre_of_rotation(16.5, distance_units='pixels')

temp = data_slice.array.reshape(data_slice.shape + (1,))
alpha = 0.005
temp = retrieve_phase(temp, pixel_size=6.5e-4, dist=30, energy=19,  alpha=alpha)
temp = temp.squeeze()
data_slice = AcquisitionData(temp, deep_copy=False, geometry = data_slice.geometry)

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

    myPDHG.run(500,verbose=2)
    
    return myPDHG

alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
i=0
for alpha in alphas:
    myPDHG = algo_isotropic_TV(data_slice, alpha=alpha, initial=None)
    reco = myPDHG.solution
    reco.apply_circular_mask(0.5)

    writer = NEXUSDataWriter()
    writer.set_up(data=reco,
            file_name='reco_alpha_x_loop_'+str(i)+'.nxs')
    writer.write()

    np.save('obj_alpha_x_loop_'+str(i)+'.npy', myPDHG.objective)

    i=i+1

