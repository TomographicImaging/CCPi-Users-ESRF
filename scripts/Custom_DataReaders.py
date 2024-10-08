from scripts.HDF5_ParallelDataReader import HDF5_ParallelDataReader

def ID15_DataReader(file_name, dataset_path): 
    """
    A custom HDF5 parallel data reader configured to read data from an ID15 experiment

    """
        
    reader = HDF5_ParallelDataReader(file_name=file_name, 
                                    dataset_path=dataset_path,
                                    distance_units='mm', angle_units='degree')

    reader.configure_angles(angles_path=('1.1/measurement/hrrz_center',
                            '4.1/measurement/hrrz_center'))

    reader.configure_pixel_sizes('1.1/instrument/pcoedgehs/x_pixel_size',
                                '1.1/instrument/pcoedgehs/y_pixel_size', HDF5_units = 'um')

    reader.configure_normalisation_data(flatfield_path='2.1/measurement/pcoedgehs/',
                                        darkfield_path='3.1/measurement/pcoedgehs/')

    reader.configure_sample_detector_distance(sample_detector_distance=90, HDF5_units='mm')

    return reader

def Custom_DataReader(file_name, dataset_path):
    """
    Configure custom filepaths here
    """
    reader = HDF5_ParallelDataReader(file_name=file_name, 
                                    dataset_path=dataset_path,
                                    distance_units='mm', angle_units='degree')

    reader.configure_angles(angles_path=('1.1/measurement/hrrz_center',
                            '4.1/measurement/hrrz_center'))

    reader.configure_pixel_sizes('1.1/instrument/pcoedgehs/x_pixel_size',
                                '1.1/instrument/pcoedgehs/y_pixel_size', HDF5_units = 'um')

    reader.configure_normalisation_data(flatfield_path='2.1/measurement/pcoedgehs/',
                                        darkfield_path='3.1/measurement/pcoedgehs/')

    reader.configure_sample_detector_distance(sample_detector_distance=90, HDF5_units='mm')
    return reader

        