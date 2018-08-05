#!/usr/bin/env python2

''' Method to load a single Matlab file and waverform '''

#Libaries
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt

#### Config ####
fname = '/media/martin/DATA/Data/MatFiles/Swath/2011/CS_LTA__SIR_SIN_1B_20110630T075707_20110630T075812_C001.mat'
wf_number = 347

#### Code ####

#Load data
mat = sio.loadmat(fname)
df = pd.DataFrame({"wf_number": mat['wf_number'][:,0],"sampleNb": mat['sampleNb'][:,0],"demDiff": mat['demDiff'][:,0],"elev": mat['elev'][:,0],"powerScaled": mat['power'][:,0],"lat": mat['lat'][:,0],"lon": mat['lon'][:,0]})

#Filter on Waveform
filteredData = df[df['wf_number']==wf_number]

#Print data and plots
print("DataFrame for Waveform")
print(filteredData)

df[df['wf_number']==wf_number].plot.scatter(x='lon',y='lat')
plt.show()

df[df['wf_number']==wf_number].plot.scatter(x='sampleNb',y='powerScaled')
plt.show()

df[df['wf_number']==wf_number].plot.scatter(x='sampleNb',y='lat')
plt.show()


