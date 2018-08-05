import scipy as sp
from scipy.interpolate import griddata 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc


############## Functions #######################

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)

def plotHist(predicted,swath_elev,name,savePath,year):

    
    diffs = predicted-swath_elev
    
    
    m = np.mean(diffs)
    std = np.sqrt(np.mean((diffs)**2))
    
    med = np.median(diffs)
    mad = np.median(np.abs(diffs - np.median(diffs)))
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.hist(diffs,bins=101,range=[-20,20], figure=fig)
    #plt.hist(diffs,bins=101, figure=fig)
    plt.ylabel('Count', figure=fig)
    plt.xlabel('Predicted - Swath Elev (m)', figure=fig)
    ax.text(0.05, 0.8, 'Mean={:.2f}\nStd={:.2f}\nMedian={:.2f}\nMad={:.2f}'.format(m,std,med,mad),transform=ax.transAxes)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    title = '{} Large Dimension Model {}'.format(year,name)
    plt.title(title, figure=fig)
    fig.savefig(savePath+title+'.png',format='png')
    

######### Config ###############
#fname = '/media/martin/FastData/Data/hdf/predictions/Models-nn-5k/BigRunNN_NN_Huber_Adamax_5000_NoScaleY/jak11train/jak11test_Full.h5'

year = 'South East 2012:'
area = 'all11to14'
test = "AllSe12"

fname = '/media/martin/FastData/Data/hdf/predictions/Models-nn-50k/BigRunNN_NN_Huber_Adamax_50000_NoScaleY/{}/{}_Filtered.h5'.format(area,test)
data = pd.read_hdf(fname,key="data")
plotHist(data['Predicted'],data['Elev_Swath'],'(Filtered)', '/media/martin/FastData/Plots/',year)


