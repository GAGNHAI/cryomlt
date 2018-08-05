import torch
import time
import pandas as pd

fname = '/media/martin/FastData/Data/hdf/areas/southeast2011.h5'
store = pd.HDFStore(fname,mode='r',complevel=9, complib='blosc')
nrows = store.get_storer('swath').nrows

#fname = '/media/martin/FastData/Data/hdf/timeseries/jakobshavn_Dist=50_ExcludePoca=False_maxSwathElev=None.h5'
#join= pd.read_hdf(fname,key="y2011")


#store = pd.HDFStore(fname,mode='r',complevel=9, complib='blosc')



#fname = '/media/martin/DATA/Data/hdf/swath/2011/swath_20110527-101600_20110527-101600_0.h5'
#bad = pd.read_hdf(fname,key="data")

#fname = '/media/martin/DATA/Data/hdf/swath/2011/swath_20110525-002914_20110525-002914_0.h5'
#good = pd.read_hdf(fname,key="data")



#fname = '/media/martin/FastData/Data/hdf/areas/southeast2011.h5'
#T= pd.read_hdf(fname,key="swath")

#oib= pd.read_hdf(fname,key="oib")

#merged = T_ind.merge(O_ind, indicator=True, how='outer')
#merged[merged['_merge'] == 'right_only']


#T_swath = pd.read_hdf(fname,key="swath")
#T_oib = pd.read_hdf(fname,key="oib")
#T_poca = pd.read_hdf(fname,key="poca")

#fname = '/media/martin/FastData/Models/Datasets/Jak2011-size=0.001-date=20180530-214840'
#fname = '/media/martin/DATA/Data/hdf/areas/jakobshavn2011old.h5'
#xx = pd.read_hdf(fname,key="y_train")
#O_oib = pd.read_hdf(fname,key="oib")
#O_poca = pd.read_hdf(fname,key="poca")
#store.close()


#store = pd.HDFStore(fname,mode='r',complevel=9, complib='blosc')

#fname = '/media/martin/DATA/Data/hdf/areas/jakobshavn_gpu.h5'
#distGPU = pd.read_hdf(fname,key="distIndex")