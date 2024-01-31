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
from wedgescan_functions import get_slice_processed





# take a single slice
vertical = 400
data_slice = get_slice_processed(vertical)

data = data_slice.copy()
# process data
ig = data.geometry.get_ImageGeometry()
r = RingRemover(8,'db20', 1.5)
r.set_input(data)
data = r.get_output()
data.geometry.set_centre_of_rotation(16.5, distance_units='pixels')

start_array = np.zeros(data.shape[1])
end_array = np.zeros(data.shape[1])
x = np.arange(data.shape[1])
for i in x:

    index = np.where(abs(np.diff(data.array[:,i])) > 0.75)[0]
    start_array[i] = index[0]
    end_array[i] = index[-1]

poly1d_fn = np.poly1d(np.polyfit(x,start_array,1)) 
start_fit = np.round(poly1d_fn(x)).astype(int)

poly1d_fn = np.poly1d(np.polyfit(x,end_array,1)) 
end_fit = np.round(poly1d_fn(x)).astype(int)


data_fill = data.copy()

lower_limit = np.min(start_fit)
upper_limit = np.max(end_fit)

ramp_width = 50
sino_mean = np.mean(data.array[np.max(start_fit):np.min(end_fit),i])

for i in np.arange(data.shape[1]):
    start = start_fit[i]+10
    end = end_fit[i]-10

    data_fill.array[lower_limit:start,i] = np.mean(data_fill.array[start:start+100,i])
    y_new = np.interp(np.arange(lower_limit-ramp_width,lower_limit), [lower_limit-ramp_width,lower_limit], [sino_mean, data_fill.array[lower_limit,i]])
    data_fill.array[lower_limit-ramp_width:lower_limit,i] = y_new
    data_fill.array[0:lower_limit-ramp_width,i] = sino_mean

    data_fill.array[end:upper_limit,i] = np.mean(data_fill.array[end-100:end,i])
    y_new = np.interp(np.arange(upper_limit-1,upper_limit+ramp_width-1), [upper_limit-1, upper_limit+ramp_width-1], [data_fill.array[upper_limit-1,i], sino_mean])
    data_fill.array[upper_limit-1:upper_limit+ramp_width-1,i] = y_new
    data_fill.array[upper_limit+ramp_width-1:,i] = sino_mean

padsize = 3000
data_pad = Padder.linear_ramp(padsize, 0)(data_fill)

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

def algo_anisotropic_TV_y(data_slice, alpha, alpha_dy, sigma=0.5, initial=None):

    ag = data_slice.geometry
    ig = ag.get_ImageGeometry()

    F = BlockFunction(0.5*L2NormSquared(b=data_slice),
                          alpha*MixedL21Norm(), 
                          alpha_dy*L1Norm())

    K = BlockOperator(ProjectionOperator(ig, ag), 
                         GradientOperator(ig), 
                         FiniteDifferenceOperator(ig, direction='horizontal_y'))

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

# #### long iso TV
# alpha = 0.5
# sigma = 0.1

# myPDHG = algo_isotropic_TV_implicit(data_pad, alpha=alpha, sigma=sigma, initial=None)
# myPDHG.run(2000,verbose=2)
# reco = myPDHG.solution

# reco = Slicer(roi={'horizontal_y': (padsize,padsize+2560), 'horizontal_x' : (padsize,padsize+2560)})(reco)

# writer = NEXUSDataWriter()
# writer.set_up(data=reco,
#         file_name='long_sigma01_reco_alpha05.nxs')
# writer.write()

# np.save('long_sigma01_obj_alpha05.npy', myPDHG.objective)

# #### long aniso x TV
# alpha = 0.5
# alpha_dx = 0.1
# sigma = 0.1

# myPDHG = algo_anisotropic_TV(data_pad, alpha=alpha, alpha_dx=alpha_dx, sigma=sigma, initial=None)
# myPDHG.run(2000,verbose=2)
# reco = myPDHG.solution

# reco = Slicer(roi={'horizontal_y': (padsize,padsize+2560), 'horizontal_x' : (padsize,padsize+2560)})(reco)

# writer = NEXUSDataWriter()
# writer.set_up(data=reco,
#         file_name='long_sigma01_reco_alpha05_alphadx01.nxs')
# writer.write()

# np.save('long_sigma01_obj_alpha05_alphadx01.npy', myPDHG.objective)
# # 

# #### long aniso y TV
alpha = 0.5
alpha_dy = 0.4
sigma = 0.1

myPDHG = algo_anisotropic_TV_y(data_pad, alpha=alpha, alpha_dy=alpha_dy, sigma=sigma, initial=None)
myPDHG.run(2000,verbose=2)
reco = myPDHG.solution

reco = Slicer(roi={'horizontal_y': (padsize,padsize+2560), 'horizontal_x' : (padsize,padsize+2560)})(reco)

writer = NEXUSDataWriter()
writer.set_up(data=reco,
        file_name='long_sigma01_reco_alpha05_alphady05.nxs')
writer.write()

np.save('long_sigma01_obj_alpha05_alphady05.npy', myPDHG.objective)
# 



### filter
# data_filter = data_pad.copy()

# fbp = FBP(data_filter, ig)
# fbp.set_filter_inplace(True)
# fbp.run()

# roi = {'horizontal':(padsize, data_filter.shape[1]-padsize, 1)}
# processor = Slicer(roi)
# processor.set_input(data_filter)
# data_filter = processor.get_output()

# from cil.plugins.tigre import ProjectionOperator
# reco = FBP(data_pad, ig).run()
# projection_operator = ProjectionOperator(image_geometry=None, acquisition_geometry=data_filter.geometry,adjoint_weights='FDK')
# reco_filter = projection_operator.adjoint(data_filter)

# projection_operator = ProjectionOperator(image_geometry=ImageGeometry(5000,5000), acquisition_geometry=data_filter.geometry,adjoint_weights='FDK')
# reco_backprojected = projection_operator.adjoint(data_filter)

# reco_backprojected_crop = reco_backprojected.copy()
# mask = (np.arange(reco_backprojected_crop.shape[0])[np.newaxis,:]-2500)**2 + (np.arange(reco_backprojected_crop.shape[0])[:,np.newaxis]-2500)**2 < (2500/2)**2
# reco_backprojected_crop.array[mask] = 0

# forward_projection = projection_operator.direct(reco_backprojected_crop)

# projection_operator_full = ProjectionOperator(image_geometry=ig, acquisition_geometry=data_fill.geometry, adjoint_weights='FDK')
# forward_projection_full = projection_operator.direct(reco_backprojected_crop)

# subtracted = data_fill - forward_projection

# ## try aniso PDHG with no padding
# alpha = 0.05
# alpha_dx = 0.01
# sigma = 0.1

# myPDHG = algo_anisotropic_TV(subtracted, alpha=alpha, alpha_dx=alpha_dx, sigma=sigma, initial=None)
# myPDHG.run(1000,verbose=2)
# reco = myPDHG.solution


# writer = NEXUSDataWriter()
# writer.set_up(data=reco,
#         file_name='filter_nopad_aniso_reco_alpha005_alphadx001_sigma01.nxs')
# writer.write()

# np.save('filter_nopad_aniso_obj_alpha005_alphadx001_sigma01.npy', myPDHG.objective)



# ## try aniso PDHG with padding
# padsize = 1000
# subtracted_pad = Padder.linear_ramp(padsize, 0)(subtracted)
# myPDHG = algo_anisotropic_TV(subtracted_pad, alpha=alpha, alpha_dx=alpha_dx, sigma=sigma, initial=None)
# myPDHG.run(1000,verbose=2)
# reco = myPDHG.solution


# writer = NEXUSDataWriter()
# writer.set_up(data=reco,
#         file_name='filter_pad_aniso_reco_alpha005_alphadx001_sigma01.nxs')
# writer.write()

# np.save('filter_pad_aniso_obj_alpha005_alphadx001_sigma01.npy', myPDHG.objective)