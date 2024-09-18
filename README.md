# Users-ESRF

`cil_pipeline.ipynb` provides an example of reading in, preprocessing and reconstructing an ESRF dataset.

`scripts` contains `HDF5_ParallelDataReader`, `WeightDuplicateAngles` and `Normaliser`. These are classes provided as extensions to CIL that are used by the `cil_pipeline.ipynb`

You will need to run the code inside a conda environment created with:

`conda create --name cil -c conda-forge -c https://software.repos.intel.com/python/conda -c ccpi cil=24.1.0 astra-toolbox=*=cuda* tigre ccpi-regulariser ipywidgets h5py hdf5plugin al
gotom
conda activate CIL
pip install nabu
`
