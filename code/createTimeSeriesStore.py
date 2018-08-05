#!/usr/bin/env python2

'''
Loads and joins the areas store data for a list of study areas and years
It then saves in a single store per area

Instructions:
    1) Set configuration section below
    2) Run 
'''

#Libraries
import pandas as pd
import time
from dhdt.datastoreHelper import DataStoreHelper as dsh

############ Config ###############
areas = ['jakobshavn','strostrommen','southeast']
timeSeries = ['2011','2012','2013','2014','2015','2016']

storeSaveFolder = '/media/martin/FastData/Data/hdf/timeseries/'
storeLoadFolder = '/media/martin/FastData/Data/hdf/areas/'

#Set the distance at which to exclude any points if further apart than.
distThreshold=50

##### Optional parameters - not final version but other functionality there if needed in future
excludePoca = False
maxSwathElev = None
uniqueSwath = False
applyNoiseFilter = False

#### Start of Code #########

for area in areas:

    storeSavePath = storeSaveFolder+area +'_Dist='+str(distThreshold)+'_ExcludePoca='+str(excludePoca)+'_maxSwathElev='+str(maxSwathElev)+'_uniqueSwath='+str(uniqueSwath)+'.h5'
          
    #Timer
    start = time.time()
    
    #Create store
    store = pd.HDFStore(storeSavePath,mode='w',complevel=9, complib='blosc')
    
    #Loop through years
    for year in timeSeries:
        storeLoadPath = storeLoadFolder+area+year+'.h5'
        print('Loading {}'.format(storeLoadPath))
        
        #Load ML ready data
        load = dsh.loadJoinedAreaMlReady(storeLoadPath,uniqueSwath=uniqueSwath,distThreshold=distThreshold,applyNoiseFilter=applyNoiseFilter,excludeBeforePoca=excludePoca,maxSwathElev=maxSwathElev)
        
        #Create store key
        key = 'y'+year
        
        #add data to store
        print('Saving {}'.format(key))
        store.append(key,load,index=False,data_columns=True)
        
    store.close()
    
    taken = time.time()-start
    print("Finished! Time taken: {}".format(taken))