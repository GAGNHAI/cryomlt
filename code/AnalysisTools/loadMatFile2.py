import os.path
import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


fname = '/media/martin/DATA/Data/MatFiles/Swath/2011/CS_LTA__SIR_SIN_1B_20110201T065456_20110201T065611_C001.mat'
mat = sio.loadmat(fname)
'''
matA = mat

df = pd.DataFrame({"wf_number": mat['wf_number'][:,0],"sampleNb": mat['sampleNb'][:,0],"elev": mat['elev'][:,0],"phase": mat['phase'][:,0],"coh": mat['coh'][:,0],"lon": mat['lon'][:,0]})

#1702

df[df['wf_number']==3165].plot.scatter(x='lon',y='coh')
plt.show()

df[df['wf_number']==3165].plot.scatter(x='sampleNb',y='phase')
plt.show()

df[df['wf_number']==3165].plot.scatter(x='sampleNb',y='coh')
plt.show()


fname = '/media/martin/DATA/Data/MatFiles/Poca/2011/CS_LTA__SIR_SIN_1B_20110415T142434_20110415T142535_C001_POCA.mat'
mat = sio.loadmat(fname)

df2 = pd.DataFrame({"wf_number": mat['wf_number'][:,0],"sampleNb": mat['sampleNb'][:,0],"elev": mat['elev'][:,0],"power": mat['powerdB'][:,0],"lat": mat['lat'][:,0],"lon": mat['lon'][:,0]})

df2[df2['wf_number']==227].plot.scatter(x='sampleNb',y='power')
plt.show()

df2[df2['wf_number']==227].plot.scatter(x='sampleNb',y='elev')
plt.show()
'''
# Make time datetime.datetime.fromtimestamp(1302873874).strftime('%Y%m%d_%H%M%S')