#!/usr/bin/env python2

'''
Creates a distance index (distIndex) for an AreaStore
This uses the All join method

Instructions:
    1) Set configuration section below
    2) Run 
'''

#Libaries
from __future__ import print_function
import torch
import psutil
import gc
import time
import sys
import numpy as np
import pandas as pd


###### Config #######

timeWindow = 864000 #10 days (seconds)
storePath = '/media/martin/FastData/Data/hdf/areas/jakobshavn2011.h5'
storeKeyMap = 'distIndex'

#Maximum memory settings - If having memory issues, reduce memory size
memorySizeGB = 6
maxCSLoadSize = 20000000 #This is the maximum number of CS rows that will be loaded into memory at once. 

#Maximum distance to join
maxDist = 50

######## Start of code #########

#Set maximum data chunk size
maxChunk = 27777778 * memorySizeGB

## Functions ##

def torchFromPandas(df,colList,transpose=False):
    ''' Returns a torch tensor from a pandas df'''
    if transpose:
        npObj = df.as_matrix(colList).transpose()
    else:        
        npObj = df.as_matrix(colList)
    return torch.from_numpy(npObj)

def getMinDist(setA,setB):
    ''' Calculates the minimum dist of 2 equal length sets'''
    
    dists = np.array([setA[0],setB[0]])
    inds = np.array([setA[1],setB[1]])
        
    minLocations = dists.argmin(axis=0)
    
    length = np.arange(dists.shape[1])
    minDists = dists[minLocations,length]
    minInds = inds[minLocations,length]
    
    return np.array([minDists,minInds])

def distToDF(a,oibBlock,swathBlock,maxDist):
    ''' Converts distance calculation to data frame'''
    oibLength = oibBlock.shape[0]
    swathLength = swathBlock.shape[0]
    
    #Throw error if indices are not same lengths
    assert oibLength == a.shape[0]
    assert swathLength == a.shape[1]
    
    dfLength = oibLength * swathLength
    
    print("oib:{},swath:{},comb:{}".format(oibLength,swathLength,dfLength))
    
    aT = a.view(dfLength)
    aTsqrt = aT.sqrt()
    aTnumpy = aTsqrt.cpu().numpy()
    
    #Duplicate indexes repeating oib to duplicate down and tile to duplicate across array
    oibInds = np.repeat(oibBlock.index,dfLength/oibLength)
    swathInds = np.tile(swathBlock.index,dfLength/swathLength)
    
    df= pd.DataFrame({'distance':aTnumpy,'swathIndex':swathInds,'oibIndex':oibInds})
    print('Memory: {}%, Swap: {}%'.format(psutil.virtual_memory().percent,psutil.swap_memory().percent))

    #Exclude any pairings greater than maxDist
    df = df[df['distance']<=maxDist]
    #df.where('distance<={}'.format(maxDist),inplace=True)
    
    #Removes any failed index matches - will have -1
    df = df[df['swathIndex']>=0]
    #df.where('swathIndex>=0',inplace=True)
        
    return df

def distTupleToNumpy(a,inds):
    ''' convert a distance torch tuple to numpy'''
    dists = a[0].cpu().numpy()
    #Get the index of the mins from the matrix
    matrixInds = a[1].cpu().numpy()
    #Apply these to the actual indices from the pandas df
    actInds = inds.index[matrixInds]
    return np.array([dists,actInds])

def createDistDF(data):
    ''' Takes a distance numpy array and returns a data frame'''
    sqrt = np.sqrt(data[0].transpose())
    df= pd.DataFrame({'distance':sqrt,'swathIndex':data[1].transpose()})
    
    #Removes any failed index matches - will have -1
    df = df[df['swathIndex']>=0]
    df.reset_index(inplace=True)
    df.rename(columns={'index':'oibIndex'},inplace=True)
    return df

#### Start of process #####

#Timer
start = time.time()

#Determine GPU
if torch.cuda.is_available():
    print('GPU available - so it is being used (Approx 4x faster than CPU)')
    print('Config has {}gB memory specified - reduce if have less memory or any issues'.format(memorySizeGB))
    dtype = torch.cuda.FloatTensor
else:
    print('GPU unavailable: using CPU - will be approx 4x slower than with GPU')
    print('Config has {}gB memory specified - reduce if have less memory or any issues'.format(memorySizeGB))
    dtype = torch.FloatTensor

#Load
store = pd.HDFStore(storePath,mode='a',complevel=9, complib='blosc')
if storeKeyMap in store:
    store.remove(storeKeyMap)

swathStorer = store.get_storer('swath')
baseOib = store.get('oib')

nRowsCS = swathStorer.nrows
lenOIB = baseOib.shape[0]

#If want to run on smaller subset for testing
#nRowsCS = 400000
#lenOIB = 20000

outerCSLoopCount = int(np.ceil(nRowsCS*1.0/maxCSLoadSize))
oibLoopCount = len(baseOib['startTime'].unique())
print('\nCS Rows: {:,}'.format(nRowsCS))
print('OIB Rows: {:,}'.format(lenOIB))
print('Outer CS Loops: {}, OIB Loops: {}'.format(outerCSLoopCount,oibLoopCount))

#distDf = pd.DataFrame(columns=['distance','swathIndex','oibIndex'])
distDfExists = False
distDf = None

rowCounterCS = 0
loopCounter = 0
while rowCounterCS < nRowsCS:

    baseSwath = swathStorer.read(start=rowCounterCS,stop=rowCounterCS+maxCSLoadSize)
    lenCS = baseSwath.shape[0]
       
    curTime = baseOib['startTime'][0]
    iTime = 0
    iCS = 0
    iOIB = 0
    loopCounter += 1
    print('Outer Loop {} of {}, InnerCSLength: {}, Memory: {}%, Swap: {}%'.format(loopCounter,outerCSLoopCount,lenCS,psutil.virtual_memory().percent,psutil.swap_memory().percent))
    print('Oib Loop: ', end='')
    
    #Loop this way to preserve order of OIB - unique list wouldn't do this
    oibLoopCounter = 0
    while iTime <= lenOIB:
        if iTime != lenOIB:
            nextTime = baseOib['startTime'][iTime]
            if curTime == nextTime:
                iTime += 1
                continue  
        
        #OIB
        oibLoopCounter += 1
        sys.stdout.write('{}..'.format(oibLoopCounter))
        sys.stdout.flush()
        thisOib = baseOib[iOIB:iTime]
        iX = torchFromPandas(thisOib,['x'],False).type(dtype)
        iY = torchFromPandas(thisOib,['y'],False).type(dtype)
        
        #Create filters to cut down CS2 data
        xMin = iX.min()-maxDist
        xMax = iX.max()+maxDist
        yMin = iY.min()-maxDist
        yMax = iY.max()+maxDist
        timeMin = curTime - timeWindow
        timeMax = curTime + timeWindow
        
        OIBLength = iX.size(0)
               
        chunkCS = maxChunk//OIBLength #This is set to manage memory at a max of (50000 x 5000)
        print("ChunkCS:{}".format(chunkCS))
        iXo = torch.ones(OIBLength,1).type(dtype)
        iYo = torch.ones(OIBLength,1).type(dtype)   
        
        while iCS < lenCS:
            
            a = baseSwath[iCS:iCS+chunkCS][['x','y','startTime']]
            keep = (a['x']<=xMax) & (a['x']>=xMin) & (a['y']<=yMax) & (a['y']>=yMin) & (a['startTime']<=timeMax) & (a['startTime']>=timeMin)
            cutSwath = a[keep]
            CS2Length = cutSwath.shape[0]
            #print(CS2Length)
            if CS2Length != 0:        
                #CS2
                cX = torchFromPandas(cutSwath,['x'],True).type(dtype)
                cY = torchFromPandas(cutSwath,['y'],True).type(dtype)
                   
                #CS2 - Ones
                cXo = torch.ones(1,CS2Length).type(dtype)
                cYo = torch.ones(1,CS2Length).type(dtype)
                
                #Calc square of distances
                distSquare = ((cX * iXo) - (iX * cXo))**2 + ((cY * iYo) - (iY * cYo))**2
                distSquareDf = distToDF(distSquare,thisOib,cutSwath,maxDist)
                
                if distSquareDf.shape[0]>0:
                    if distDfExists:
                        distDf = distDf.append(distSquareDf)
                    else:
                        distDf = distSquareDf
                        distDfExists = True
                
                #Prevent running out of memory on GPU
                del cX, cY, cXo,cYo,distSquare, distSquareDf, cutSwath
                torch.cuda.empty_cache() 
                
            iCS += chunkCS
            
        #Continue loop
        iCS = 0
        
        #Prevent running out of memory on GPU
        del iX, iY, iXo, iYo, OIBLength,thisOib
        torch.cuda.empty_cache()
        
        #Iterate to next starttime collection
        iOIB = iTime
        curTime = nextTime
        iTime += 1
        gc.collect() # Force garbage collector to run so nothing left that shouldn't be
    
    rowCounterCS += maxCSLoadSize
    gc.collect() # Force garbage collector to run so nothing left that shouldn't be

    
#Clean index
distDf = distDf.reset_index().drop(['index'],axis=1)    

#Add to HDF Store and Close
store.append(storeKeyMap,distDf,index=False,data_columns=True)
store.close()
    
taken = time.time()-start
print("Finished! Time taken: {}".format(taken))



        





