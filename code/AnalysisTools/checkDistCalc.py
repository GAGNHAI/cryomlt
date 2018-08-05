#!/usr/bin/env python2

'''
Quick Check for comparing joined data actual distance with expected distance from the nearest/all join process

Must load data into variable called "join" first to run this using DataStoreHelper.loadJoinedArea()
'''

#Libraries
import pandas as pd
import numpy as np

#Calculated Euclidean distance
distsSquared = (join['x_oib']-join['x_swath'])**2 + (join['y_oib']-join['y_swath'])**2
dists = np.sqrt(distsSquared)

#There will be slight numerical rounding diffs, and we're working with accuracy to the nearest metre
#So round to nearest integer
distDiffs = np.round(abs(join['distance'] - dists))

#Print summary
if max(distDiffs) == 0.0:
    print("Success - Max Diff is 0")
else:
    print("Error - Max Diff is {}".format(max(distDiffs)))