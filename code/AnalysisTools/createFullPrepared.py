#import torch
import time
import pandas as pd

startTime = time.time()

fname = '/media/martin/FastData/Data/hdf/timeseries-nn/mlprepared_withfull.h5'
store = pd.HDFStore(fname,mode='a',complevel=9)

key = 'alljak11'
fname = '/media/martin/FastData/Data/hdf/areas-swathpocajoin/jakobshavn2011.h5'
data= pd.read_hdf(fname,key="data")
store.append(key,data,index=False,data_columns=True)


key = 'alljak12'
fname = '/media/martin/FastData/Data/hdf/areas-swathpocajoin/jakobshavn2012.h5'
data= pd.read_hdf(fname,key="data")
store.append(key,data,index=False,data_columns=True)

key = 'allse11'
fname = '/media/martin/FastData/Data/hdf/areas-swathpocajoin/southeast2011.h5'
data= pd.read_hdf(fname,key="data")
store.append(key,data,index=False,data_columns=True)

key = 'allse12'
fname = '/media/martin/FastData/Data/hdf/areas-swathpocajoin/southeast2012.h5'
data= pd.read_hdf(fname,key="data")
store.append(key,data,index=False,data_columns=True)

store.close()
