#Move code to DataStore
#Create one with reduced columns

import pandas as pd
import time
from dhdt.datastoreHelper import DataStoreHelper as dsh


#Timer
start = time.time()

####### Config ######

#storePath = '/media/martin/DATA/Data/hdf/areas-nn/jakobshavn2012.h5'
storePath = '/media/martin/DATA/Data/hdf/areas-nn/strostrommen2011.h5'
#storePath = '/media/martin/FastData/Data/hdf/areas/jakobshavn2011-31days.h5'


####### Code ######

#Max height of Greenland is 3694
#Max Jakobshavn is around 2000 - depends on exact area used. Therefore specify 2500 as limit
#Have 50m threshold for distance pairing to Oib

#See all data
#join = dsh.loadJoinedArea(storePath,applyNoiseFilter=False,uniqueSwath=False,distThreshold=50)#,maxSwathElev=2500)

#Full Join
#joinFull = dsh.loadJoinedArea(storePath,applyNoiseFilter=True,uniqueSwath=False,excludeBeforePoca=True,pocaJoinOnly=True)


#ML prepared data
#join1 = dsh.loadJoinedAreaMlReady(storePath,uniqueSwath=False,distThreshold=500,applyNoiseFilter=False,excludeBeforePoca=False,maxSwathElev=None)

#ML Predict only
join = dsh.loadJoinedAreaMlPredict(storePath,applyNoiseFilter=True,excludeBeforePoca=True,maxSwathElev=None)
store = pd.HDFStore('/media/martin/FastData/Data/hdf/areas-swathpocajoin/strostrommen2011.h5',mode='w',complevel=9, complib='blosc')
store.append('data',join,index=False,data_columns=True)
store.close()

#taken = time.time()-start
#print("Finished! Time taken: {}".format(taken))