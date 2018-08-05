import os.path
import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


fname = '/media/martin/DATA/Data/MatFiles/Swath/test/CS_LTA__SIR_SIN_1B_20110614T113126_20110614T113338_C001-Old.mat'
mat = sio.loadmat(fname)
matA = mat

old = pd.DataFrame({"wf_number": mat['wf_number'][:,0],"sampleNb": mat['sampleNb'][:,0],"demDiff": mat['demDiff'][:,0],"elev": mat['elev'][:,0],"powerScaled": mat['power'][:,0],"lat": mat['lat'][:,0],"lon": mat['lon'][:,0]})

oldFil = old[old['wf_number']==1069]

#1702
'''
df[df['wf_number']==227].plot.scatter(x='lon',y='lat')
plt.show()

df[df['wf_number']==227].plot.scatter(x='sampleNb',y='powerdB')
plt.show()

df[df['wf_number']==227].plot.scatter(x='sampleNb',y='lat')
plt.show()
'''

fname = '/media/martin/DATA/Data/MatFiles/Swath/test/CS_LTA__SIR_SIN_1B_20110614T113126_20110614T113338_C001-New.mat'
mat = sio.loadmat(fname)

new = pd.DataFrame({"wf_number": mat['wf_number'][:,0],"sampleNb": mat['sampleNb'][:,0],"demDiff": mat['demDiff'][:,0],"elev": mat['elev'][:,0],"powerScaled": mat['powerScaled'][:,0],"lat": mat['lat'][:,0],"lon": mat['lon'][:,0]})

newFil = new[new['wf_number']==1069]
'''
df2[df2['wf_number']==227].plot.scatter(x='sampleNb',y='power')
plt.show()

df2[df2['wf_number']==227].plot.scatter(x='sampleNb',y='elev')
plt.show()
'''
# Make time datetime.datetime.fromtimestamp(1302873874).strftime('%Y%m%d_%H%M%S')