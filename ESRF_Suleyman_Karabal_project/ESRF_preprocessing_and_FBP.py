# Author : Evelina Ametova, Evangelos Papoutsellis

import h5py
import hdf5plugin # need to import this, 
# otherwise get OSError Can't read data (can't open directory: /opt/anaconda3/envs/cil_devel_epaps/lib/hdf5/plugin)
# pip install hdf5plugin

import numpy as np
import matplotlib.pyplot as plt
import os
from tifffile import imsave
import dask
import time
from cil.framework import AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry
from cil.plugins.astra.processors import FBP
from cil.processors import RingRemover

import numpy as np
from scipy import stats
from scipy.ndimage import shift
from scipy.ndimage import gaussian_filter

import dask
import dask.array as da
import algotom.prep.calculation as calc

# from multiprocessing.pool import ThreadPool
# dask.config.set(pool=ThreadPool(5))

from scipy.fftpack import fftshift, ifftshift, fft, ifft
import pywt
import numba
import logging

###########################################################
#### utils 

### Ring Remover from CIL repo. It is a numpy API based function and 
# will be used by dask library
def xRemoveStripesVertical(ima, decNum, wname, sigma):
        
        ''' Code from https://doi.org/10.1364/OE.17.008567 
            translated in Python
                            
        Returns
        -------
        Corrected 2D sinogram data (Numpy Array)
        
        '''              
                            
        # allocate cH, cV, cD
        Ch = [None]*decNum
        Cv = [None]*decNum
        Cd = [None]*decNum
            
        # wavelets decomposition
        for i in range(decNum):
            ima, (Ch[i], Cv[i], Cd[i]) = pywt.dwt2(ima,wname) 
    
        # FFT transform of horizontal frequency bands
        for i in range(decNum):
            
            # use to axis=0, which correspond to the angles direction
            fCv = fftshift(fft(Cv[i], axis=0))
            my, mx = fCv.shape
            
            # damping of vertical stripe information
            damp = 1 - np.exp(-np.array([range(-int(np.floor(my/2)),-int(np.floor(my/2))+my)])**2/(2*sigma**2))
            fCv *= damp.T
             
            # inverse FFT          
            Cv[i] = np.real(ifft(ifftshift(fCv), axis=0))
                                                                    
        # wavelet reconstruction
        nima = ima
        for i in range(decNum-1,-1,-1):
            nima = nima[0:Ch[i].shape[0],0:Ch[i].shape[1]]
            nima = pywt.idwt2((nima,(Ch[i],Cv[i],Cd[i])),wname)
            
        return nima

# Numba acceleration for flat/dark correction    
@numba.njit(parallel=True)
def flat_dark_correct(data, dark, denom):
    
    for i in numba.prange(data.shape[0]):
        
        data[i,:,:] = (data[i,:,:] - dark)/denom
        
    return data    

# Prepare for ring remover
@dask.delayed
def prepare_data_for_ring(data2D):
    
    tmp1 = data2D
    tmp2 = np.zeros_like(tmp1)
    tmp2[:3000,:] = tmp1[3000:,:]
    tmp2[3000:,:] = tmp1[:3000,:] 

    arr1 = tmp1[:,:-(2560-center0)]
    arr2 = np.fliplr(tmp2[:,:-(2560-center0)])
    tmp3 = np.concatenate((arr1, arr2), axis=1)

    tmp3 = tmp3[:3000, :]
    np.log(tmp3, out=tmp3)
    tmp3 *= -1 
    
    return tmp3

# arrays to uint8
def to_uint8(rec):
    
    vmax = np.quantile(np.ravel(rec), 0.99)
    vmin = np.quantile(np.ravel(rec), 0.01)

    rec[rec>vmax] = vmax
    rec[rec<vmin] = vmin

    rec = (rec-vmin)/(vmax-vmin)
    rec *= 255
    rec = np.uint8(rec)
    
    return rec

###### end of utils #############################


### Path to ESRF data
path = '/media/vaggelis/Seagate Basic1/'
path1 = '/media/vaggelis/Seagate Backup Plus Drive/'
#path1 = '/media/vaggelis/Backup Plus/'
data_path = path + 'NewBatch_SampleB_No4/'

### Output folder
#output_folder = path1 + 'NewBatch_SampleB_No4_recon_with_CIL_ring_optimised_53_to_77_partIII/'

output_folder = path1 + 'NewBatch_SampleB_No4_recon_with_CIL_ring_optimised_21_to_48/'


num_of_tomogram = [45, 46, 47, 48]
#num_of_tomogram = [15] # run in 27295.pts-12.E-EULXA1002430
for j in num_of_tomogram:
        
    tomogram = 'NewBatch_SampleB_No4_00{:02d}'.format(j)    
    
    ### prepare logger 
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
            
    filehandler = logging.FileHandler(filename = output_folder + 'Log_files/' + tomogram  +'.txt')
    filehandler.setLevel(logging.INFO)
    format_style = logging.Formatter(
                    fmt='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )
    filehandler.setFormatter(format_style)
    logger.addHandler(filehandler)

    logging.info("-------Start--------")
    
    #### Read raw data
    logging.info(" Start reading raw data")
    filename = data_path + tomogram + '/scan0001/pcoedgehs_0000.h5'.format(j)
    f = h5py.File(filename,'r')
    data = np.array(f['entry_0000/measurement/data/'][:], dtype=np.float32)        
    logging.info(" End reading raw data")

    logging.info(" Start reading flat data")
    filename = data_path + tomogram + '/scan0002/pcoedgehs_0000.h5'.format(j)
    f = h5py.File(filename,'r')
    flat = np.array(f['entry_0000/measurement/data/'][:], dtype=np.float32)    
    logging.info(" End reading flat data")

    logging.info(" Start reading dark data")
    filename = data_path + tomogram + '/scan0003/pcoedgehs_0000.h5'.format(j)
    f = h5py.File(filename,'r')
    dark = np.array(f['entry_0000/measurement/data/'][:], dtype=np.float32) 
    logging.info(" End reading dark data")
        
    flat = np.mean(flat, axis=0)
    dark = np.mean(dark, axis=0)
    denom = (flat - dark)       

    logging.info("Start flat-dark field correction")
    data = flat_dark_correct(data, dark, denom)
    logging.info("End flat-dark field correction")

    sino_360 = data[:, 1000, :]

    logging.info("Start Center of rotation correction")
    (center0, overlap, side,_) = calc.find_center_360(sino_360, 100, side=1)
    center0 = np.int32(np.round(center0))    
    logging.info("End: Center of rotation correction: center0 = {}".format(center0))

    # lazy computation of ring remover
    
    logging.info(" Total number of slices is {}. 130 slices from top and bottom are not considered.".format(data.shape[1]))
    
    results = []
    for i in range(130, data.shape[1]-130+1):

        data2D = data[:, i, :]
        arr1 = prepare_data_for_ring(data2D)
        arr2 = dask.delayed(xRemoveStripesVertical)(arr1, decNum=10, wname='db30', sigma=5.5)
        results.append(arr2)

    logging.info("Start Remove vertical stripes: L = 10, Wavelet = db30, sigma = 5.5")
    results_list = dask.compute(*results)
    logging.info("End Remove vertical stripes: L = 10, Wavelet = db30, sigma = 5.5")    

    # delete  
    del data, flat, dark, results    

    # number of slices
    num_of_slices = len(results_list)

    n_proj = results_list[0].shape[0]
    imsize = results_list[0].shape[1]
    
    output_folder_fbp = output_folder + 'FBP/' + tomogram
    os.mkdir(output_folder_fbp)    
    
    # Create acquisition geometry
    ag = AcquisitionGeometry.create_Parallel2D()
    ag.set_panel(imsize)
    ag.set_angles(np.linspace(0, 180, n_proj))
    ad = ag.allocate()
    ag.config.system.rotation_axis.position[0] = 0

    # Create image geometry
    ig = ag.get_ImageGeometry()

    logging.info("Start FBP reconstruction for {} slices".format(num_of_slices))
    for i in range(num_of_slices):

        ad.fill(results_list[i])
        t1 = time.time()
        rec = FBP(ig, ag,  device = 'gpu')(ad)
        rec_np = to_uint8(rec.array)
        imsave(output_folder_fbp + '/slice_{:04d}.tif'.format(i), rec_np) 
        
    logging.info("End FBP reconstruction for {} slices".format(num_of_slices))        
        
    del rec, rec_np, results_list    
    
    logger.removeHandler(filehandler)
    del logger,filehandler    
    


