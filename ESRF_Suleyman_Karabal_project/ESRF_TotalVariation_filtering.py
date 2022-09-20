# Author : Evelina Ametova, Evangelos Papoutsellis


import numpy
import dask
import dask.array as da
import dask.array.image
from cil.framework import ImageGeometry
from cil.plugins.ccpi_regularisation.functions import FGP_TV
import logging
from tifffile import imsave
import os


### utils

def im2double(data):
    
    m1 = data.min()
    m2 = data.max()
    tmp = (data - m1)/(m2 - m1)
    return tmp

# arrays to uint8
def to_uint8(rec):
    
    vmax = numpy.quantile(numpy.ravel(rec), 0.99)
    vmin = numpy.quantile(numpy.ravel(rec), 0.01)

    rec[rec>vmax] = vmax
    rec[rec<vmin] = vmin

    rec = (rec-vmin)/(vmax-vmin)
    rec *= 255
    rec = numpy.uint8(rec)
    
    return rec

###

### number of tomogram
num_of_tomogram = [45, 46, 47, 48]

### path to ESRF FBP reconstruction

tmp_path = "/media/vaggelis/Seagate Backup Plus Drive/NewBatch_SampleB_No4_recon_with_CIL_ring_optimised_21_to_48/"
#tmp_path = "/media/vaggelis/Backup Plus/NewBatch_SampleB_No4_recon_with_CIL_ring_optimised/"
path_folder_FBP = tmp_path + "/FBP/"
output_folder = tmp_path + "TV/"

for j in num_of_tomogram:
    
    tomogram = 'NewBatch_SampleB_No4_00{:02d}'.format(j)
            
    ### prepare logger 
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
            
    filehandler = logging.FileHandler(filename = tmp_path + 'TV_Log_files/' + tomogram  +'.txt')
    filehandler.setLevel(logging.INFO)
    format_style = logging.Formatter(
                    fmt='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )
    filehandler.setFormatter(format_style)
    logger.addHandler(filehandler)

    logging.info("-------Start--------")        
            
    # lazy tiff read via dask
    fbp_recon = dask.array.image.imread(path_folder_FBP + tomogram + "/*.tif") 
    
    logging.info(" Shape of FBP reconstruction is {}".format(fbp_recon.shape))    
    
    # shape of fbp_recon
    n1, n2, n3 = fbp_recon.shape
    
    
    # ImageGeometry    
    ig = ImageGeometry(voxel_num_x = n2, voxel_num_y = n3)
    im_cil = ig.allocate()
    tv_recon = ig.allocate()
    
    # Total variation filtering
    regularisation_parameter = 0.2  
    max_iteration = 500
    tv_filter = regularisation_parameter * FGP_TV(max_iteration=max_iteration, device = 'gpu')
    
    logging.info(" TV regularisation: parameter = {}, iterations = {} ".format(regularisation_parameter, max_iteration))
    
    # create directory and save tv regularised slice
    output_folder_tv = output_folder + tomogram
    os.mkdir(output_folder_tv)         
    
    
    logging.info(" Start TV regularisation for {} slices ".format(n1)) 
    for i in range(n1):
        
        tmp_slice = im2double(fbp_recon[i].compute())
        im_cil.fill(tmp_slice)
        tv_filter.proximal(im_cil, tau = 1.0, out = tv_recon)
        
        # to_uint8
        result = to_uint8(tv_recon.array)
                             
        imsave(output_folder_tv + '/slice_{:04d}.tif'.format(i), result) 
        
    logging.info(" End TV regularisation for {} slices ".format(n1)) 
    
    logger.removeHandler(filehandler)
    del logger,filehandler  

        
        
        
        
    

    
    
