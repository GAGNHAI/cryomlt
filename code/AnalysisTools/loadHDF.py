#import torch
import time
import pandas as pd

startTime = time.time()

#fname = '/media/martin/FastData/Data/hdf/timeseries-nn/jakobshavn_Dist=50_ExcludePoca=False_maxSwathElev=None_uniqueSwath=False.h5'
#fname = '/media/martin/FastData/Data/hdf/timeseries-nn/mlprepared_withfull.h5'
#store = pd.HDFStore(fname,mode='r',complevel=9, complib='blosc')
#data = store['y2011']
#nrows = store.get_storer('se14').nrows
#print nrows



#fname = '/media/martin/DATA/Data/hdf/swathOld/2011/swath_20110201-051606_20110201-051606_0.h5'

#fname = '/media/martin/DATA/Data/hdf/poca/2011/poca_20110201-065456_20110201-065456_0.h5'

#index = pd.read_hdf(fname,key="index")
#dataX = pd.read_hdf(fname,key="data")

#print(time.time()-startTime)


#fname = '/media/martin/FastData/Data/hdf/timeseries-nn/mlprepared.h5'
#store = pd.HDFStore(fname,mode='r',complevel=9, complib='blosc')
#nrows = store.get_storer('jak11').nrows
#print nrows

#fname = '/media/martin/FastData/Data/hdf/predictions/Models-nn-5k/BigRunNN_NN_Huber_Adamax_5000_NoScaleY/jak11/AllJak11_Full.h5'
#store = pd.HDFStore(fname,mode='r',complevel=9, complib='blosc')
#nrows = store.get_storer('data').nrows
#print nrows

#fname = '/media/martin/FastData/Data/hdf/timeseries/jakobshavn_Dist=50_ExcludePoca=False_maxSwathElev=None.h5'
#join= pd.read_hdf(fname,key="y2011")


#store = pd.HDFStore(fname,mode='r',complevel=9, complib='blosc')


#fname = '/media/martin/DATA/Data/hdf/swath/2012test/swath_20120201-001001_20120201-001001_1.h5'
#shard1= pd.read_hdf(fname,key="index")

#fname = '/media/martin/DATA/Data/hdf/swath/2012test/swath_20120201-001001_20120201-001001_1-b.h5'
#shardb= pd.read_hdf(fname,key="index")

fname = '/media/martin/FastData/Data/hdf/areas/jakobshavn2011.h5'
distIndex= pd.read_hdf(fname,key="distIndex")


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