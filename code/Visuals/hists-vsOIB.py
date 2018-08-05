import scipy as sp
from scipy.interpolate import griddata 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc


font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)


############## Functions #######################



def plotHist(predicted,swath_elev,name,savePath,year,area,titleLabel,fromRange,toRange):

        
    diffs = predicted-swath_elev
    
    
    m = np.mean(diffs)
    rMSE = np.sqrt(np.mean((diffs)**2))
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.hist(diffs,bins=101,range=[fromRange,toRange], figure=fig)
    #plt.hist(diffs,bins=101, figure=fig)
    plt.ylabel('Count', figure=fig)
    plt.xlabel('{} (m)'.format(titleLabel), figure=fig)
    #ax.text(0.05, 0.95, r'$\mu={:.2f},\ \sigma={:.2f}$'.format(m,std),transform=ax.transAxes)
    ax.text(0.05, 0.90, 'Mean Diff: {:.2f}\nRMSE: {:.2f}'.format(m,rMSE),transform=ax.transAxes)
    title = '{}: {} ({})'.format(area,titleLabel,name)
    plt.title(title, figure=fig)
    fig.savefig(savePath+title+'.png',format='png')
    

######### Config ###############
#fname = '/media/martin/FastData/Data/hdf/predictions/Models-nn-5k/BigRunNN_NN_Huber_Adamax_5000_NoScaleY/jak11train/jak11test_Full.h5'

year = 2015
area = 'jak11'

fromRange = -20
toRange = 20



folder = '/media/martin/FastData/Data/hdf/predictions/plots/BigRunNN_NN_Huber_Adamax_50000_NoScaleY/'

fullOrFil = 'Full'
name = 'Large Dimensions'
fromSet = 'Elev_Oib'
toSet = 'Predicted'
titleLabel = 'Predicted - OIB Elev'
subFolder = '{}{}train/'.format(folder,area)
fname = '{}{}test_{}.h5'.format(subFolder,area,fullOrFil)
data = pd.read_hdf(fname,key="data")
plotHist(data[toSet],data[fromSet],fullOrFil,subFolder,year,name, titleLabel,fromRange,toRange)

fullOrFil = 'Full'
name = 'Unadjusted'
fromSet = 'Elev_Oib'
toSet = 'Elev_Swath'
titleLabel = 'CS Swath - OIB Elev'
subFolder = '{}{}train/'.format(folder,area)
fname = '{}{}test_{}.h5'.format(subFolder,area,fullOrFil)
data = pd.read_hdf(fname,key="data")
plotHist(data[toSet],data[fromSet],fullOrFil,subFolder,year,name, titleLabel,fromRange,toRange)

fullOrFil = 'Filtered'
name = 'Large Dimensions'
fromSet = 'Elev_Oib'
toSet = 'Predicted'
titleLabel = 'Predicted - OIB Elev'
subFolder = '{}{}train/'.format(folder,area)
fname = '{}{}test_{}.h5'.format(subFolder,area,fullOrFil)
data = pd.read_hdf(fname,key="data")
plotHist(data[toSet],data[fromSet],fullOrFil,subFolder,year,name, titleLabel,fromRange,toRange)

fullOrFil = 'Filtered'
name = 'Unadjusted'
fromSet = 'Elev_Oib'
toSet = 'Elev_Swath'
titleLabel = 'CS Swath - OIB Elev'
subFolder = '{}{}train/'.format(folder,area)
fname = '{}{}test_{}.h5'.format(subFolder,area,fullOrFil)
data = pd.read_hdf(fname,key="data")
plotHist(data[toSet],data[fromSet],fullOrFil,subFolder,year,name, titleLabel,fromRange,toRange)

folder = '/media/martin/FastData/Data/hdf/predictions/plots/HSmallRun_NN_L1_Adamax_50000_ScaleY/'

fullOrFil = 'Full'
name = 'Small Dimensions'
fromSet = 'Elev_Oib'
toSet = 'Predicted'
titleLabel = 'Predicted - OIB Elev'
subFolder = '{}{}train/'.format(folder,area)
fname = '{}{}test_{}.h5'.format(subFolder,area,fullOrFil)
data = pd.read_hdf(fname,key="data")
plotHist(data[toSet],data[fromSet],fullOrFil,subFolder,year,name, titleLabel,fromRange,toRange)

fullOrFil = 'Filtered'
name = 'Small Dimensions'
fromSet = 'Elev_Oib'
toSet = 'Predicted'
titleLabel = 'Predicted - OIB Elev'
subFolder = '{}{}train/'.format(folder,area)
fname = '{}{}test_{}.h5'.format(subFolder,area,fullOrFil)
data = pd.read_hdf(fname,key="data")
plotHist(data[toSet],data[fromSet],fullOrFil,subFolder,year,name, titleLabel,fromRange,toRange)