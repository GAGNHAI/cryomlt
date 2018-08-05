#!/usr/bin/env python2

'''
Find a nearest neighbour using a convention loop

Must load hdf data (using standard pd.read_hdf) into baseOib and baseSwath variables first from an AreaStore.
'''

#Library
import numpy as np
import time

start = time.time()

i = 0
oibLength = baseOib.shape[0]
oibLength = 5 #Set Oib length to test

minDists = np.array([[],[]])
swath = baseSwath[['x','y']]
oib = baseOib[['x','y']]

#Find nearest neighbour for each OIB
while i <  oibLength:

    #Filter out data more than 1000m perpendicular distance
    xMin = oib['x'][i]-1000
    xMax = oib['x'][i]+1000
    yMin = oib['y'][i]-1000
    yMax = oib['y'][i]+1000
    a = swath
    keep = (a['x']<=xMax) & (a['x']>=xMin) & (a['y']<=yMax) & (a['y']>=yMin)
    swathCut = a[keep]
       
    ones = np.ones(swathCut.shape[0])
    
    #Calc Euclidean for all points
    dists = (swathCut['x']-oib['x'][i]*ones)**2 + (swathCut['y']-oib['y'][i]*ones)**2
    
    #Select minimum
    minV = min(dists)
    ind = np.argmin(dists)
    minDists = np.append(minDists,[[minV],[ind]],axis=1)
    
    i += 1
    
taken = time.time()-start
print(taken)

#Estimate time to take for running on whole dataset
print("Days to calc: {}".format((801771/oibLength)*taken/(60*60*24)))


