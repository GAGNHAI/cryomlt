#!/usr/bin/env python2

'''
Queries the sharded data for a year and bounding box and creates an Area store
This is done for Swath, Poca and OIB data

Instructions:
    1) Set configuration section below
    2) Run 
'''

#Libaries
import pandas as pd
import time
from dhdt.datastoreHelper import DataStoreHelper

############ Config ###############

#Save Store Location
year = '2016'
area = 'areaname'
saveRootFolder = '/media/martin/FastData/Data/hdf/'

#Sharded data location - must have a swath, poca and oib folder
loadRootFolder = '/media/martin/DATA/Data/hdf/'

#Set polar coordinated bounding box
#Jakobshavn
#oibBbox = [-262187, -2386746, -93976, -2144979]
#Stostrommen
#oibBbox = [262956, -1317863, 625920, -985707]
#SouthEast
oibBbox = [53165, -2921176, 228166, -2671177]


########### Dynamic Links ############
# Change this section for non-standard config such as changing folder names or date ranges

#Store Path location
storePath = saveRootFolder + 'areas/' + area + year + '.h5'

#Globs
swathGlob = loadRootFolder + 'swath/'+year+'/swath_'+year+'*'
pocaGlob = loadRootFolder + 'poca/'+year+'/poca_'+year+'*'
oibGlob = loadRootFolder + 'oib/'+year+'/oib_'+year+'*'

#Date Range
oibDateRange = ["01/03/"+year,"31/05/"+year]
cs2DateRange = ["01/02/"+year,"30/06/"+year]

#Polarstereo or wgs86
latLonOrXY = 'XY'

#Create buffers for swath and poca
swathBuffer = [-1000,-1000,1000,1000] #1,000m more wide than OIB
pocaBuffer = [-10000,-10000,10000,10000] #10,000m more wide than Swath


########## start of code #################

startFull = time.time()

#Apply buffers
lists_of_lists = [oibBbox, swathBuffer]
swathBbox = [sum(x) for x in zip(*lists_of_lists)]

lists_of_lists = [swathBbox, pocaBuffer]
pocaBbox = [sum(x) for x in zip(*lists_of_lists)]

#Load
store = pd.HDFStore(storePath,mode='w',complevel=9, complib='blosc')

print("Loading Swath")
DataStoreHelper.loadManyToHDF(swathGlob,store,'swath',swathBbox,latLonOrXY,cs2DateRange)

### Fast mode for smaller datasets
print("Loading Poca")
dataPoca = DataStoreHelper.loadMany(pocaGlob,pocaBbox,latLonOrXY,cs2DateRange)
store.append('poca',dataPoca,index=False,data_columns=True)
del dataPoca

print("Loading Oib")
dataOib = DataStoreHelper.loadMany(oibGlob,oibBbox,latLonOrXY,oibDateRange)
store.append('oib',dataOib,index=False,data_columns=True)
del dataOib

store.close()

print('Full Time taken: {}'.format(time.time()-startFull))

