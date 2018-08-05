#!/usr/bin/env python2

'''
Specific code for Jakobshavn, Storstrommen and SouthEast to prepare ML
Loads the timeseries stores, performs quality filters
Also creates train/test split
It then saves one file with all data

Instructions:
    1) Set configuration section below
    2) Run 
'''

#Libraries
import pandas as pd
from sklearn.model_selection import train_test_split


##### Config #########

#Save name
saveName = 'mlprepared.h5'

#Timeseries store Info
storeFolder = '/media/martin/FastData/Data/hdf/timeseries-nn/'   
storeJakName = 'jakobshavn_Dist=50_ExcludePoca=False_maxSwathElev=None_uniqueSwath=False.h5'
storeSeName = 'southeast_Dist=50_ExcludePoca=False_maxSwathElev=None_uniqueSwath=False.h5'
storeStrName = 'strostrommen_Dist=50_ExcludePoca=False_maxSwathElev=None_uniqueSwath=False.h5'


##### Functions #######

def addData(store,newKey,curSet):
    'Filters, splits and appends data to datastore'''
    
    #CleanData
    data = curSet.dropna(axis=0)
    data = data[data['SampleNb_SwathMinusLeadEdgeS']>=0]
    data = data[(data['Coh_Swath']>0.2) & (data['PowerScaled_Swath']>2500)]
    
    #Append
    store.append(newKey,data,index=False,data_columns=True)
    
    #Create train test set for within same area
    train, test = train_test_split(data, train_size=0.6, test_size=0.4)
    store.append(newKey+'train',train,index=False,data_columns=True)
    store.append(newKey+'test',test,index=False,data_columns=True)

def addSet(store,prefix,curStore):
    '''Adds 6 years of areastores'''
    print('Adding {}'.format(prefix))
    addData(newStore,prefix+'11',curStore['y2011'])
    addData(newStore,prefix+'12',curStore['y2012'])
    addData(newStore,prefix+'13',curStore['y2013'])
    addData(newStore,prefix+'14',curStore['y2014'])
    addData(newStore,prefix+'15',curStore['y2015'])
    addData(newStore,prefix+'16',curStore['y2016'])


###### Main ###########

#Load Data
print('Load Data')
j = pd.HDFStore(storeFolder+storeJakName,mode='r',complevel=9, complib='blosc')
st = pd.HDFStore(storeFolder+storeStrName,mode='r',complevel=9, complib='blosc')
se = pd.HDFStore(storeFolder+storeSeName,mode='r',complevel=9, complib='blosc')

#Create Store
storeSavePath = storeFolder+saveName
newStore = pd.HDFStore(storeSavePath,mode='w',complevel=9, complib='blosc')

#Add to store
addSet(newStore,'jak',j)
addSet(newStore,'st',st)
addSet(newStore,'se',se)

#Close stores
newStore.close()
j.close()
st.close()
se.close()

print('Complete')