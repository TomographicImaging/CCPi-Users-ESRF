# Users-ESRF

`recon_example.ipynb` provides an example of reading in, preprocessing and reconstructing an ESRF dataset.

`scripts` contains `ESRF_IF15_DataReader` and `WeightDuplicateAngles`. These are classes provided as extensions to CIL that are used by the `recon_example.ipynb`

You will need to run the code inside a conda environment created with:

`conda create -n cil cil=22.0.0 tigre h5py hdf5-external-filter-plugins  ipywidgets -c conda-forge -c intel -c ccpi`
