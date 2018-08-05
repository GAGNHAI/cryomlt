#Instructions for configuring the Python 2.7 environment
#Conda must be installed as a prerequisite

#Add channel to source packages
conda config --add channels conda-forge

#Create a Python 2.7 environment
conda create -n py27 python=2.7

#Install required packages
conda install -n py27 geopandas
conda install -n py27 anaconda
conda install -n py27 spyder
conda install -n py27 configobj
conda install -n py27 basemap
conda install -n py27 netCDF4
conda install -n py27 PyTables
conda install -n py27 pytorch torchvision cuda91 -c pytorch
conda install -n py27 dask
conda install -n py27 progress
conda install -n py27 mpi4py
conda install -n py27 scikit-learn

#Activate environment - this will need to be done every time.
source activate py27

