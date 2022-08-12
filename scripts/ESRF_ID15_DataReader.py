
#conda create -n cil-esrf cil=22.0.0 tigre h5py hdf5-external-filter-plugins  ipywidgets -c conda-forge -c intel -c ccpi

from cil.framework import AcquisitionGeometry, AcquisitionData
import numpy as np
import h5py

class HDF5_utilities(object): 

    """
    Utility methods to read in from a generic HDF5 file and extract the relevant data
    """
    def __init__(self):
        pass
        

    @staticmethod
    def _descend_obj(obj, sep='\t', depth=-1):
        """
        Parameters
        ----------
        obj: str
            The initial group to print the metadata for
        sep: str, default '\t'
            The separator to use for the output
        depth : int
            depth to print from starting object. Values 1-N, if -1 will print all
        """
        if depth != 0:
            if type(obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
                for key in obj.keys():
                    print(sep, '-', key, ':', obj[key])
                    HDF5_utilities._descend_obj(obj[key], sep=sep+'\t', depth=depth-1)
            elif type(obj) == h5py._hl.dataset.Dataset:
                for key in obj.attrs.keys():
                    print(sep+'\t', '-', key, ':', obj.attrs[key])


    @staticmethod
    def print_metadata(filename, group='/', depth=-1):
        """
        Prints the file metadata

        Parameters
        ----------
        filename: str
            The full path to the HDF5 file
        group: (str), default: '/'
            a specific group to print the metadata for,
            defaults to the root group
        depth: int, default -1
            depth of group to output the metadata for, -1 is fully recursive
        """
        with h5py.File(filename, 'r') as f:
            HDF5_utilities._descend_obj(f[group], depth=depth)


    @staticmethod
    def print_dataset_metadata(filename, dset_path):
        """
        Prints the dataset metadata

        Parameters
        ----------
        filename: str
            The full path to the HDF5 file
        dset_path: str
            The internal path to the requested dataset
        """
        with h5py.File(filename, 'r') as f:
                dset = f.get(dset_path)
                print('ndim\t\t', dset.ndim)
                print('shape\t\t', dset.shape)    
                print('size\t\t', dset.size)
                print('dtype\t\t', dset.dtype)
                print('nbytes\t\t', dset.nbytes)
                print('compression\t', dset.compression)
                print('chunks\t\t', dset.chunks)


    @staticmethod
    def get_dataset_metadata(filename, dset_path):
        """
        Returns the dataset metadata

        Parameters
        ----------
        filename: str
            The full path to the HDF5 file
        dset_path: str
            The internal path to the requested dataset

        Returns
        -------
        Tuple containing:
        (ndim, shape, size, dtype, nbytes, compression, chunks)

        """
        with h5py.File(filename, 'r') as f:
                dset = f.get(dset_path)

                return [dset.ndim, dset.shape, dset.size, dset.dtype, dset.nbytes, dset.compression, dset.chunks]


    @staticmethod
    def read(filename, dset_path, source_sel=None, dtype=np.float32):
        """
        Reads dataset entry and returns it as a numpy array, source_sel will crop and slice.

        source_sel as a tuple of slice objects i.e.
        (slice(2, 4, None), slice(2, 10, 2))

        can be constructed with np.s_[2:4,2:10:2]
        """

        with h5py.File(filename, 'r') as f:
            dset = f.get(dset_path)

            if source_sel == None:
                source_sel = tuple([slice(None)]*dset.ndim)

            arr = dset.astype(dtype)[source_sel]

        return arr 


    @staticmethod
    def read_to(filename, dset_path, out, source_sel=None, dest_sel=None):
        """
        Reads data entry and fills a numpy array
        """
        with h5py.File(filename, 'r') as f:
            dset = f.get(dset_path)
            dset.read_direct(out, source_sel, dest_sel)


#%%
class ESRF_ID15_DataReader(object): 
    """
    ESRF ID15 data reader

    Parameters
    ----------
    file_name : string
        full path to master h5 ESRF file

    Example
    -------
    >>> from cil.io import ESRFID15Reader
    >>> reader = ESRFReader('path/to/master_0001.h5')
    >>> data_full_scan = reader.read()

    Example
    -------
    >>> from cil.io import ESRFReader
    >>> reader = ESRFReader('path/to/master_0001.h5')
    >>> data_geometry_scan2 = reader.get_geometry(dataset_id=2)
    >>> data_scan2 = reader.read(dataset_id=2, normalise=False)
    >>> darkfield_scan2 = reader.get_darkfield_images(dataset_id=2)
    >>> flatfield_scan2 = reader.get_flatfield_images(dataset_id=2)

    Example
    -------
    >>> from cil.io import ESRFID15Reader
    >>> reader = ESRFReader('path/to/master_0001.h5')
    >>> reader.set_roi(vertical=480)
    >>> data_single_slice = reader.read()
    >>> reader.set_roi(vertical=None)
    >>> data_full = reader.read()



    TODO
    accessing through the master file, it returns an empty dataset if the scan isn't found. this needs to be behind a crosscheck
    positioners id2obj -> gives pixel size (kinda of)
    idx image detector difference
    hry rotation axis position

    """

    @property
    def number_of_datasets(self):
        return self._number_of_datasets


    def __init__(self, filename):

        self.filename = filename
        self._get_number_of_datasets()
        self._ag_full = None
        self._roi = [slice(None), slice(None), slice(None)]


    def _get_number_of_datasets(self):

        with h5py.File(self.filename, 'r') as f:
            group_list = [float(key) for key in f.keys()]

            num_entries = len(group_list)
            num_data_entries = num_entries / 2

            if num_data_entries % 3:
                raise ValueError ("Expected a multiple of 3")

        self._number_of_datasets = int(num_data_entries//3)

    def _get_dataset_number(self, dataset_id, data_index):
        return str(dataset_id * 3 - 2 + data_index)

    def set_roi(self, vertical=None):
        """
        Can currently only take a single integer, None returns all

        i.e.
        vertical=1

        """
        if vertical is None:
            self._roi[1] = slice(None)
        else:
            self._roi[1] = slice(vertical, vertical+1)


    def get_flatfield_images(self, dataset_id=1):


        return HDF5_utilities.read(self.filename, self._get_dataset_number(dataset_id,1)+'.1/measurement/pcoedgehs/', source_sel=tuple(self._roi))
        

    def get_darkfield_images(self, dataset_id=1):

        return HDF5_utilities.read(self.filename, self._get_dataset_number(dataset_id,2)+'.1/measurement/pcoedgehs/', source_sel=tuple(self._roi))


    def get_geometry(self, dataset_id=None):

        if dataset_id is not None:
            data_shape = HDF5_utilities.get_dataset_metadata(self.filename, self._get_dataset_number(dataset_id,0)+'.1/measurement/pcoedgehs')[1]
            angles = HDF5_utilities.read(self.filename, self._get_dataset_number(dataset_id,0)+'.1/measurement/hrrz_center')
            
            ag = AcquisitionGeometry.create_Parallel3D().set_panel([data_shape[2],data_shape[1]]).set_angles(angles)

        else:
            if self._ag_full is None:
                data_shape = HDF5_utilities.get_dataset_metadata(self.filename, '1.1/measurement/pcoedgehs')[1]

                angles = []
                for i in range(1, self.number_of_datasets*3,3):
                    angles.append(HDF5_utilities.read(self.filename, str(i)+'.1/measurement/hrrz_center'))

                self._ag_full = AcquisitionGeometry.create_Parallel3D().set_panel([data_shape[2],data_shape[1]]).set_angles(np.concatenate(angles))
            ag = self._ag_full

        if self._roi[1].start is None:
            return ag
        else:
            return ag.get_slice(vertical = self._roi[1].start)


    def read(self, dataset_id=None, normalise=True):

        ag = self.get_geometry(dataset_id=dataset_id)

        if dataset_id is not None:

            data =  HDF5_utilities.read(self.filename, self._get_dataset_number(dataset_id,0)+'.1/measurement/pcoedgehs/', source_sel=tuple(self._roi))

            if normalise == True:
                flat = np.mean(self.get_flatfield_images(dataset_id), axis=0)
                dark = np.mean(self.get_darkfield_images(dataset_id), axis=0)

                flat = flat - dark

                data = (data - dark)/ flat

            ad = AcquisitionData(data.squeeze(), False, ag)

        else:
            ad = ag.allocate(None)

            start = 0
            for i in range(self.number_of_datasets):

                dark = np.mean(self.get_darkfield_images(i+1), axis=0)
                flat = np.mean(self.get_flatfield_images(i+1), axis=0)
                data = HDF5_utilities.read(self.filename, self._get_dataset_number(i+1,0)+'.1/measurement/pcoedgehs/', source_sel=tuple(self._roi))

                flat = flat - dark
                data = ((data - dark)/ flat).squeeze()
                length = data.shape[0]

                ad.array[start:(start+length)] = data
                start +=length

                del data

        return ad

