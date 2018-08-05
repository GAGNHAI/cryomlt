#!/usr/bin/env python2

''' Code to demonstrate performane difference of loading sharded data vs a Matlab file '''

#Import libraries needed
import time
import scipy.io as sio
import numpy
from dhdt.datastoreHelper import DataStoreHelper as dsh
import pandas as pd


#Put in bounding box that is outside of shard
bbox = [-15,81,-9,84]

#Load shard
startTime = time.time()
fname = '/media/martin/DATA/Data/hdf/swath/2011/swath_20110629-234519_20110629-234519_0.h5'
shardData = dsh.loadData(fname,bbox,'LatLon')
shardTime = time.time()-startTime

#Load MatFile
startTime = time.time()
fname = '/media/martin/DATA/Data/MatFiles/Swath/2011/CS_LTA__SIR_SIN_1B_20110629T234519_20110629T234631_C001.mat'
mat = sio.loadmat(fname)

#Convert Matfile to useful format so could then apply filter
cols = []

#Define datatypes
dtypes = {}
dtypes['wf_number'] = numpy.uint16
dtypes['sampleNb'] = numpy.uint16
dtypes['powerScaled'] = numpy.uint16
dtypes['phaseAmb'] = numpy.int16
ndata = mat['lon'].shape[0]
matData = pd.DataFrame()

#Get data in to a usable form
for c in mat.keys():
    if type(mat[c]) == numpy.ndarray:
        if mat[c].shape[0] == ndata:
            cols.append(c)
            if not c in dtypes:
                dtypes[c] = numpy.float64
for c in cols:
    matData[c] = pd.Series(dtype=dtypes[c])
    
for c in cols:
    matData[c] = numpy.array(mat[c][:,0],dtype=dtypes[c])
#Done conversion
 
matTime = time.time()-startTime

#Calculate ratio
ratio = matTime/shardTime

#Print Results
print("Shard Load Time: {} seconds".format(shardTime))
print("Matfile Load Time: {} seconds".format(matTime))
print("Ratio: {}".format(ratio))
