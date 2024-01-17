from cil.processors import TransmissionAbsorptionConverter
from cil.io.utilities import HDF5_utilities
import numpy as np
import hdf5plugin
from cil.framework import AcquisitionGeometry, AcquisitionData
from tomopy.prep.phase import retrieve_phase

def get_slice_processed(vertical_slice):
    filename = '/mnt/data/ESRF/Wedgescan_Iterative_ASSB/InSitu-LPSCL-20Ton-30Min_0001.h5'#
    angles = HDF5_utilities.read(filename, '/4.1/measurement/hrsrot')
    ds_metadata = HDF5_utilities.get_dataset_metadata(filename, '4.1/instrument/pco2linux/data')

    roi = [slice(None), slice(450, 950), slice(None)]
    # roi = [slice(None), slice(200, 1100), slice(None)]
    source_sel=tuple(roi)

    filename = '/mnt/data/ESRF/Wedgescan_Iterative_ASSB/flats.h5'#
    print(filename)
    HDF5_utilities.print_metadata(filename, '/entry_0000/measurement', 2)
    HDF5_utilities.get_dataset_metadata(filename, 'entry_0000/measurement/data')
    flats = HDF5_utilities.read(filename, 'entry_0000/measurement/data', tuple(source_sel))
    flat = np.mean(flats, axis = 0) #median?

    filename = '/mnt/data/ESRF/Wedgescan_Iterative_ASSB/darks.h5'#
    print(filename)
    HDF5_utilities.print_metadata(filename, 'entry_0000/measurement', 2)
    HDF5_utilities.get_dataset_metadata(filename, 'entry_0000/measurement/data')
    darks = HDF5_utilities.read(filename, 'entry_0000/measurement/data', tuple(source_sel))
    dark = np.mean(darks, axis = 0) 

    projections = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    data = np.empty((3000, roi[1].stop-roi[1].start, ds_metadata['shape'][2]), dtype=np.float32)
    for i in projections:
        filename = '/mnt/data/ESRF/Wedgescan_Iterative_ASSB/pco2linux_{0:04}.h5'.format(i)
        print(filename)
        HDF5_utilities.read_to(filename, 'entry_0000/measurement/data',data,source_sel, np.s_[i*200:i*200+200,:,:])

    flat = flat - dark
    data = (data - dark)/ flat

    ag = AcquisitionGeometry.create_Parallel3D().set_panel([np.shape(data)[2],np.shape(data)[1]]).set_angles(angles)
    data = AcquisitionData(data, deep_copy=False, geometry = ag)

    print('Running transmission-absorption converter')
    processor = TransmissionAbsorptionConverter(min_intensity=0.001)
    processor.set_input(data)
    processor.get_output(out=data)
    # show2D(data)

    print('Running gradient correction')
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
        data.array[i,:,:] = data.array[i,:,:] / poly1d_fn(x)
    data.array[:,:,:] = mean_intensity*data.array[:,:,:]/np.mean(data.array[:,:,:])

    print('Running phase retrieval')
    data_phase = retrieve_phase(data.array, alpha=0.01)
    data = AcquisitionData(data_phase, deep_copy=False, geometry = data.geometry)

    return data.get_slice(vertical=vertical_slice)